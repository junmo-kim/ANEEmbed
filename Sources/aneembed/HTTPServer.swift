// The only listening socket: HTTP/1.1 over Network.framework with keep-alive. Parsing and routing live
// in EmbedCore (HTTPParser, Router); this file moves bytes.
import EmbedCore
import Foundation
import Network

final class HTTPServer: @unchecked Sendable {
    let host: String
    let port: Int
    let maxBody: Int
    let router: Router
    static let idleTimeout: Duration = .seconds(60)
    private var listener: NWListener?
    private let stopped = AsyncStream<Int32>.makeStream()

    init(host: String, port: Int, maxBody: Int, router: Router) {
        self.host = host
        self.port = port
        self.maxBody = maxBody
        self.router = router
    }

    func start() throws(ServiceError) {
        let parameters = NWParameters.tcp
        guard let endpointPort = NWEndpoint.Port(rawValue: UInt16(port)) else {
            throw .badRequest("invalid_option", "invalid port \(port)", param: "port")
        }
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: endpointPort)
        parameters.allowLocalEndpointReuse = true
        let listener: NWListener
        do { listener = try NWListener(using: parameters) } catch {
            throw ServiceError(.unavailable, "listen_failed", "cannot listen on \(host):\(port): \(error)")
        }
        listener.newConnectionHandler = { [self] connection in
            connection.start(queue: .global())
            Task { await self.serve(connection) }
        }
        listener.stateUpdateHandler = { [self] update in
            switch update {
            case .ready: Log.info("listening on http://\(host):\(port)")
            case .failed(let error):
                Log.info("listener failed: \(error)")
                stopped.continuation.yield(ExitCode.unavailable.rawValue)
            default: break
            }
        }
        listener.start(queue: .global())
        self.listener = listener
    }

    /// Returns when the listener fails.
    func wait() async -> ExitCode {
        for await code in stopped.stream { return ExitCode(rawValue: code) ?? .runtimeError }
        return .runtimeError
    }

    // MARK: Connections

    private func receive(_ connection: NWConnection) async -> Data? {
        await withCheckedContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { data, _, isComplete, error in
                if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                } else if isComplete || error != nil {
                    continuation.resume(returning: nil)
                } else {
                    continuation.resume(returning: Data())
                }
            }
        }
    }

    private func send(_ data: Data, on connection: NWConnection) async -> Bool {
        await withCheckedContinuation { continuation in
            connection.send(content: data, completion: .contentProcessed { error in continuation.resume(returning: error == nil) })
        }
    }

    private func serve(_ connection: NWConnection) async {
        var buffer = Data()
        let idle = IdleTimer(connection: connection, timeout: Self.idleTimeout)
        defer {
            idle.stop()
            connection.cancel()
        }
        while true {
            // Head.
            idle.reset()
            var head: HTTPHead?
            do {
                while true {
                    if let parsed = try HTTPParser.parseHead(buffer, maxBody: maxBody) {
                        head = parsed
                        break
                    }
                    guard let chunk = await receive(connection) else { return }
                    buffer.append(chunk)
                }
            } catch {
                let failure = ServiceError(error.status == 413 ? .payloadTooLarge : .badRequest, "bad_http", error.message)
                _ = await send(HTTPResponse.openAIError(failure).with(status: error.status).serialized(keepAlive: false),
                               on: connection)
                return
            }
            guard var head else { return }
            // Body.
            buffer.removeSubrange(buffer.startIndex..<(buffer.startIndex + head.headLength))
            while buffer.count < head.contentLength {
                guard let chunk = await receive(connection) else { return }
                buffer.append(chunk)
            }
            head.request.body = Data(buffer.prefix(head.contentLength))
            buffer.removeSubrange(buffer.startIndex..<(buffer.startIndex + head.contentLength))
            buffer = Data(buffer)  // rebase indices

            idle.stop()
            let response = await router.handle(head.request)
            let keepAlive = head.request.keepAlive
            guard await send(response.serialized(keepAlive: keepAlive), on: connection), keepAlive else { return }
        }
    }
}

extension HTTPResponse {
    func with(status: Int) -> HTTPResponse {
        var copy = self
        copy.status = status
        return copy
    }
}

/// Cancels a connection that stays idle (between requests) longer than `timeout`.
final class IdleTimer: @unchecked Sendable {
    private let connection: NWConnection
    private let timeout: Duration
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    init(connection: NWConnection, timeout: Duration) {
        self.connection = connection
        self.timeout = timeout
    }

    func reset() {
        lock.withLock {
            task?.cancel()
            task = Task { [connection, timeout] in
                try? await Task.sleep(for: timeout)
                if !Task.isCancelled { connection.cancel() }
            }
        }
    }

    func stop() {
        lock.withLock {
            task?.cancel()
            task = nil
        }
    }
}
