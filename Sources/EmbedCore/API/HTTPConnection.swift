// One HTTP/1.1 connection: frames requests from the byte stream (keep-alive, pipelining), answers each through a
// handler and enforces the connection's time limits. Independent of the socket layer: `aneembed serve` passes a
// transport over its Network.framework connection, the tests an in-memory one.
import Foundation

/// The byte stream of one connection.
public protocol HTTPTransport: Sendable {
    /// The next bytes from the peer; nil once the peer closed or the connection was cancelled.
    func receive() async -> Data?
    /// Sends `data`; false when it could not be sent.
    func send(_ data: Data) async -> Bool
    /// Closes the connection; a pending `receive` returns nil.
    func cancel()
}

public struct HTTPConnectionLimits: Sendable {
    public var maxBody: Int
    /// Waiting for the next request and receiving its head: one deadline that progress does not extend, so a head
    /// trickled byte by byte (or an idle kept-alive connection) cannot hold the connection longer.
    public var headTimeout: Duration
    /// While a body arrives: the longest pause between received bytes. Every chunk restarts it, so a slow upload
    /// that keeps arriving is not cut, and a stalled one is.
    public var bodyIdleTimeout: Duration

    public init(maxBody: Int, headTimeout: Duration = .seconds(60), bodyIdleTimeout: Duration = .seconds(60)) {
        self.maxBody = maxBody
        self.headTimeout = headTimeout
        self.bodyIdleTimeout = bodyIdleTimeout
    }
}

/// Request framing over one connection's bytes: a head, then a Content-Length body; pipelined requests stay buffered.
public struct HTTPRequestReader: Sendable {
    public let maxBody: Int
    private var buffer = Data()
    private var head: HTTPHead?
    /// Bytes of `buffer` already searched for the end of the head (minus 3, for a terminator split across reads).
    private var scanned = 0
    /// Total bytes searched for heads (the cost of slow, byte-by-byte heads stays linear).
    private(set) var searchedBytes = 0

    public init(maxBody: Int) {
        self.maxBody = maxBody
    }

    /// Bytes received and not yet returned as part of a request.
    public var bufferedBytes: Int { buffer.count }

    /// The head of the next request has arrived and its body has not (yet) completely.
    public var isReadingBody: Bool { head != nil }

    public mutating func append(_ data: Data) {
        buffer.append(data)
    }

    /// The next complete request; nil while more bytes are needed.
    public mutating func next() throws(HTTPParseError) -> HTTPRequest? {
        if head == nil {
            searchedBytes += buffer.count - scanned
            guard let parsed = try HTTPParser.parseHead(buffer, maxBody: maxBody, scanFrom: scanned) else {
                scanned = max(buffer.count - 3, 0)
                return nil
            }
            head = parsed
            scanned = 0
        }
        guard let current = head, buffer.count >= current.headLength + current.contentLength else { return nil }
        var request = current.request
        let end = current.headLength + current.contentLength
        if buffer.count == end {
            // The usual case: hand the buffer over as the body (no copy of up to --max-body bytes) and keep nothing.
            var whole = buffer
            buffer = Data()
            whole.removeSubrange(whole.startIndex..<(whole.startIndex + current.headLength))
            request.body = whole
        } else {  // pipelined bytes follow
            request.body = buffer.subdata(in: (buffer.startIndex + current.headLength)..<(buffer.startIndex + end))
            buffer = buffer.subdata(in: (buffer.startIndex + end)..<buffer.endIndex)
        }
        head = nil
        return request
    }
}

/// Counts open connections against a limit.
public final class ConnectionLimiter: @unchecked Sendable {
    public let limit: Int
    private let lock = NSLock()
    private var count = 0

    public init(limit: Int) {
        self.limit = limit
    }

    public var active: Int { lock.withLock { count } }

    /// Takes a slot; false when all `limit` slots are taken.
    public func tryAcquire() -> Bool {
        lock.withLock {
            guard count < limit else { return false }
            count += 1
            return true
        }
    }

    public func release() {
        lock.withLock { count -= 1 }
    }
}

public enum HTTPConnection {
    /// Serves `transport` while fewer than `limiter.limit` connections are open; beyond that answers
    /// `503 too_many_connections` (Retry-After: 1) and closes, so open connections and their buffers stay bounded.
    public static func accept(_ transport: some HTTPTransport, limiter: ConnectionLimiter, limits: HTTPConnectionLimits,
                              handler: @Sendable (HTTPRequest) async -> HTTPResponse) async
    {
        guard limiter.tryAcquire() else {
            var response = HTTPResponse.openAIError(
                ServiceError(.unavailable, "too_many_connections",
                             "the server has \(limiter.limit) open connections, its limit; retry shortly"))
            response.headers.append(("Retry-After", "1"))
            _ = await transport.send(response.serialized(keepAlive: false))
            transport.cancel()
            return
        }
        defer { limiter.release() }
        await serve(transport, limits: limits, handler: handler)
    }

    /// Serves requests on `transport` until the peer closes, a request asks to close, or a limit is hit.
    /// `sleep` waits out a timeout; tests pass a clock they advance by hand.
    public static func serve(_ transport: some HTTPTransport, limits: HTTPConnectionLimits,
                             sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
                             handler: @Sendable (HTTPRequest) async -> HTTPResponse) async
    {
        var reader = HTTPRequestReader(maxBody: limits.maxBody)
        let timer = ConnectionTimer(transport, sleep: sleep)
        defer {
            timer.stop()
            transport.cancel()
        }
        while true {
            timer.start(limits.headTimeout)
            let request: HTTPRequest
            do {
                while true {
                    if let next = try reader.next() {
                        request = next
                        break
                    }
                    // Head: the deadline set above. Body: restarted on every chunk.
                    if reader.isReadingBody { timer.start(limits.bodyIdleTimeout) }
                    guard let chunk = await transport.receive() else { return }
                    reader.append(chunk)
                }
            } catch {
                let failure = ServiceError(error.status == 413 ? .payloadTooLarge : .badRequest, "bad_http", error.message)
                _ = await transport.send(HTTPResponse.openAIError(failure).with(status: error.status).serialized(keepAlive: false))
                return
            }
            timer.stop()
            let response = await handler(request)
            let keepAlive = request.keepAlive
            let bytes = response.serialized(keepAlive: keepAlive, includeBody: request.method != "HEAD")
            guard await transport.send(bytes), keepAlive else { return }
        }
    }
}

/// Cancels a connection when a deadline passes; `start` replaces the running deadline.
final class ConnectionTimer: @unchecked Sendable {
    private let transport: any HTTPTransport
    private let sleep: @Sendable (Duration) async throws -> Void
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    init(_ transport: any HTTPTransport, sleep: @escaping @Sendable (Duration) async throws -> Void) {
        self.transport = transport
        self.sleep = sleep
    }

    func start(_ timeout: Duration) {
        lock.withLock {
            task?.cancel()
            task = Task { [transport, sleep] in
                try? await sleep(timeout)
                if !Task.isCancelled { transport.cancel() }
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
