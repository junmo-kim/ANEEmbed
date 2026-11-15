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
    /// Received bytes that are not part of a body: the head being received, then whatever follows the body.
    private var buffer = Data()
    private var head: HTTPHead?
    /// The body of `head`: one allocation of its Content-Length that the received bytes are copied into.
    private var body: BodyBuffer?
    /// Bytes of `buffer` already searched for the end of the head (minus 3, for a terminator split across reads).
    private var scanned = 0
    /// The size of `buffer` at the last search that did not find the end of the head.
    private var searchedLength = 0
    /// Total bytes searched for heads (the cost of slow, byte-by-byte heads stays linear).
    private(set) var searchedBytes = 0

    public init(maxBody: Int) {
        self.maxBody = maxBody
    }

    /// Bytes received and not yet returned as part of a request, outside the body being filled.
    public var bufferedBytes: Int { buffer.count }

    /// The head of the next request has arrived and its body has not (yet) completely.
    public var isReadingBody: Bool { head != nil }

    public mutating func append(_ data: Data) {
        guard let body, !body.isFull else {
            buffer.append(data)
            return
        }
        let taken = body.fill(from: data)
        if taken < data.count { buffer.append(data[(data.startIndex + taken)...]) }
    }

    /// The next complete request; nil while more bytes are needed.
    public mutating func next() throws(HTTPParseError) -> HTTPRequest? {
        try parseHead()
        guard let current = head else { return nil }
        var request = current.request
        if current.contentLength > 0 {
            if body == nil { startBody(current) }
            guard let body, body.isFull else { return nil }
            request.body = body.handOver()
            self.body = nil
        } else {
            buffer = buffer.subdata(in: (buffer.startIndex + current.headLength)..<buffer.endIndex)
        }
        head = nil
        return request
    }

    private mutating func parseHead() throws(HTTPParseError) {
        guard head == nil, buffer.count > searchedLength else { return }  // nothing new since the last search
        searchedBytes += buffer.count - scanned
        guard let parsed = try HTTPParser.parseHead(buffer, maxBody: maxBody, scanFrom: scanned) else {
            scanned = max(buffer.count - 3, 0)
            searchedLength = buffer.count
            return
        }
        head = parsed
        scanned = 0
        searchedLength = 0
    }

    /// Allocates the body and moves the body bytes that arrived with the head into it; what follows them stays.
    private mutating func startBody(_ current: HTTPHead) {
        let body = BodyBuffer(count: current.contentLength)
        let from = buffer.startIndex + current.headLength
        let taken = body.fill(from: buffer[from..<buffer.endIndex])
        buffer = buffer.subdata(in: (from + taken)..<buffer.endIndex)
        self.body = body
    }
}

/// One request body, allocated at its Content-Length (no regrowing while it arrives). The bytes become a Data that
/// frees them when the last reference goes; if the body is never completed, they are freed with the buffer.
final class BodyBuffer: @unchecked Sendable {
    let count: Int
    private(set) var filled = 0
    private let bytes: UnsafeMutableRawPointer
    private var handedOver = false

    init(count: Int) {
        self.count = count
        bytes = .allocate(byteCount: count, alignment: 16)
    }

    var isFull: Bool { filled == count }

    /// Copies as much of `data` as still fits; returns how many bytes that was.
    func fill(from data: Data) -> Int {
        let taken = min(count - filled, data.count)
        data.copyBytes(to: (bytes + filled).assumingMemoryBound(to: UInt8.self), count: taken)
        filled += taken
        return taken
    }

    /// The full body as Data, which owns the bytes from now on.
    func handOver() -> Data {
        handedOver = true
        return Data(bytesNoCopy: bytes, count: count, deallocator: .custom { pointer, _ in pointer.deallocate() })
    }

    deinit {
        if !handedOver { bytes.deallocate() }
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
                              handler: @Sendable (consuming HTTPRequest) async -> HTTPResponse) async
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
                             handler: @Sendable (consuming HTTPRequest) async -> HTTPResponse) async
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
            let keepAlive = request.keepAlive, includeBody = request.method != "HEAD"
            // The handler owns the request from here, so it can free the body before it is done.
            let response = await handler(consume request)
            let bytes = response.serialized(keepAlive: keepAlive, includeBody: includeBody)
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
