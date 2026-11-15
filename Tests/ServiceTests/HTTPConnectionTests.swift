import Foundation
import Testing

@testable import EmbedCore

/// An in-memory connection: the test writes the client's bytes and reads what the server sent.
final class MemoryTransport: HTTPTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var incoming: [Data] = []
    private var waiting: CheckedContinuation<Data?, Never>?
    private var closedByClient = false
    private var output = Data()
    private(set) var cancelled = false
    private var cancelWaiters: [CheckedContinuation<Void, Never>] = []

    /// Bytes from the client.
    func write(_ data: Data) {
        let resumed = lock.withLock { () -> CheckedContinuation<Data?, Never>? in
            guard !cancelled else { return nil }
            if let waiting {
                self.waiting = nil
                return waiting
            }
            incoming.append(data)
            return nil
        }
        resumed?.resume(returning: data)
    }

    func write(_ text: String) { write(Data(text.utf8)) }

    /// The client half-closes: receive returns nil once the buffered bytes are read.
    func closeInput() {
        let resumed = lock.withLock { () -> CheckedContinuation<Data?, Never>? in
            closedByClient = true
            defer { waiting = nil }
            return waiting
        }
        resumed?.resume(returning: nil)
    }

    func receive() async -> Data? {
        await withCheckedContinuation { continuation in
            let result = lock.withLock { () -> Data?? in
                if !incoming.isEmpty { return .some(incoming.removeFirst()) }
                if cancelled || closedByClient { return .some(nil) }
                waiting = continuation
                return .none
            }
            if let result { continuation.resume(returning: result) }
        }
    }

    func send(_ data: Data) async -> Bool {
        lock.withLock {
            guard !cancelled else { return false }
            output.append(data)
            return true
        }
    }

    func cancel() {
        let (resumed, waiters) = lock.withLock { () -> (CheckedContinuation<Data?, Never>?, [CheckedContinuation<Void, Never>]) in
            cancelled = true
            defer {
                waiting = nil
                cancelWaiters = []
            }
            return (waiting, cancelWaiters)
        }
        resumed?.resume(returning: nil)
        waiters.forEach { $0.resume() }
    }

    var sent: Data { lock.withLock { output } }
    var isCancelled: Bool { lock.withLock { cancelled } }
}

/// A clock the test advances by hand, so timeout tests do not depend on how fast the machine schedules tasks.
final class ManualClock: @unchecked Sendable {
    private let lock = NSLock()
    private var now: Duration = .zero
    private var nextID = 0
    private var sleepers: [Int: (deadline: Duration, continuation: CheckedContinuation<Void, Error>)] = [:]
    private var cancelledEarly: Set<Int> = []
    private var armedCount = 0

    /// Sleeps armed so far (waiting for their deadline, woken or cancelled).
    var armed: Int { lock.withLock { armedCount } }

    func sleep(for duration: Duration) async throws {
        let id = lock.withLock { () -> Int in
            nextID += 1
            return nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let cancelled = lock.withLock { () -> Bool in
                    armedCount += 1
                    if cancelledEarly.remove(id) != nil { return true }
                    sleepers[id] = (now + duration, continuation)
                    return false
                }
                if cancelled { continuation.resume(throwing: CancellationError()) }
            }
        } onCancel: {
            let continuation = lock.withLock { () -> CheckedContinuation<Void, Error>? in
                if let sleeper = sleepers.removeValue(forKey: id) { return sleeper.continuation }
                cancelledEarly.insert(id)
                return nil
            }
            continuation?.resume(throwing: CancellationError())
        }
    }

    /// Moves time forward and wakes every sleep whose deadline has passed.
    func advance(by duration: Duration) {
        let due = lock.withLock { () -> [CheckedContinuation<Void, Error>] in
            now += duration
            let ids = sleepers.filter { $0.value.deadline <= now }.map(\.key)
            return ids.compactMap { sleepers.removeValue(forKey: $0)?.continuation }
        }
        due.forEach { $0.resume() }
    }

    /// Waits until more than `count` sleeps are armed. Polling only paces the wait; no result depends on its timing.
    func waitUntilArmed(moreThan count: Int) async throws {
        while armed <= count { try await Task.sleep(for: .milliseconds(1)) }
    }
}

/// A client-side reading of the server's bytes: responses in order, with HEAD responses carrying no body.
struct ParsedResponse {
    var status: Int
    var headers: [String: String]
    var body: Data
}

func parseResponses(_ data: Data, methods: [String]) -> (responses: [ParsedResponse], rest: Data) {
    var rest = data
    var responses: [ParsedResponse] = []
    for method in methods {
        guard let end = rest.range(of: Data("\r\n\r\n".utf8)) else { break }
        let head = String(decoding: rest[rest.startIndex..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
        let statusLine = head[0].split(separator: " ")
        guard statusLine.count >= 2, statusLine[0] == "HTTP/1.1", let status = Int(statusLine[1]) else { break }
        var headers: [String: String] = [:]
        for line in head.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = method == "HEAD" ? 0 : Int(headers["content-length"] ?? "0") ?? 0
        guard rest.distance(from: end.upperBound, to: rest.endIndex) >= length else { break }
        let bodyEnd = rest.index(end.upperBound, offsetBy: length)
        responses.append(ParsedResponse(status: status, headers: headers, body: Data(rest[end.upperBound..<bodyEnd])))
        rest = Data(rest[bodyEnd...])
    }
    return (responses, rest)
}

@Suite("HTTP connection (in-memory transport)")
struct HTTPConnectionTests {
    static let router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "embeddinggemma-2")

    /// Runs a connection until it ends; `client` drives the transport meanwhile.
    func connection(limits: HTTPConnectionLimits = HTTPConnectionLimits(maxBody: 1 << 20), clock: ManualClock? = nil,
                    _ client: @Sendable (MemoryTransport) async throws -> Void) async throws -> MemoryTransport
    {
        let transport = MemoryTransport()
        let sleep: @Sendable (Duration) async throws -> Void
        if let clock {
            sleep = { try await clock.sleep(for: $0) }
        } else {
            sleep = { try await Task.sleep(for: $0) }
        }
        let server = Task { await HTTPConnection.serve(transport, limits: limits, sleep: sleep) { await Self.router.handle($0) } }
        try await client(transport)
        await server.value
        return transport
    }

    @Test func keepAliveAndPipelinedRequests() async throws {
        let body = #"{"input":"hello"}"#
        let transport = try await connection { client in
            client.write("GET /health HTTP/1.1\r\nHost: x\r\n\r\nPOST /v1/embeddings HTTP/1.1\r\nContent-Length: \(body.utf8.count)\r\n\r\n")
            client.write(body + "GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["GET", "POST", "GET"])
        #expect(responses.map(\.status) == [200, 200, 200] && rest.isEmpty)
        #expect(responses[2].headers["connection"] == "close")
        #expect(try JSONValue.parse(responses[1].body)["data"]?.arrayValue?.count == 1)
    }

    @Test func aBadRequestIsAnsweredAndClosed() async throws {
        let transport = try await connection { client in
            client.write("GARBAGE\r\n\r\nGET /health HTTP/1.1\r\n\r\n")
        }
        let (responses, _) = parseResponses(transport.sent, methods: ["GET", "GET"])
        #expect(responses.map(\.status) == [400] && responses[0].headers["connection"] == "close")
        #expect(transport.isCancelled)
    }

    @Test func thePeerClosingEndsTheConnection() async throws {
        let transport = try await connection { client in
            client.write("GET /health HTTP/1.1\r\n")
            client.closeInput()
        }
        #expect(transport.sent.isEmpty && transport.isCancelled)
    }

    /// A HEAD response has the GET response's headers (Content-Length included) and no body, so the next
    /// response on the connection starts where the client expects it.
    @Test func headResponsesCarryNoBody() async throws {
        let transport = try await connection { client in
            client.write("HEAD /health HTTP/1.1\r\n\r\nHEAD /v1/embeddings HTTP/1.1\r\n\r\nHEAD /nope HTTP/1.1\r\n\r\n")
            client.write("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["HEAD", "HEAD", "HEAD", "GET"])
        try #require(responses.map(\.status) == [200, 405, 404, 200])
        #expect(rest.isEmpty, "bytes after the last response: \(String(decoding: rest, as: UTF8.self))")
        #expect(responses[0].headers["content-length"] == "15" && responses[0].headers["content-type"] == "application/json")
        #expect(try JSONValue.parse(responses[3].body) == ["status": "ok"])
    }

    /// A CL.CL request whose second length would smuggle a request inside the body gets one 400 and a close.
    @Test func conflictingContentLengthsCloseTheConnection() async throws {
        let smuggled = "GET /health HTTP/1.1\r\nHost: x\r\n\r\n"
        let transport = try await connection { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: \(smuggled.utf8.count)\r\n\r\n"
                         + smuggled)
            client.closeInput()
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["POST", "GET"])
        #expect(responses.map(\.status) == [400] && rest.isEmpty)
        #expect(responses.first?.headers["connection"] == "close" && transport.isCancelled)
    }

    /// A head that arrives one byte at a time is searched for its end in linear, not quadratic, time.
    @Test func headSearchResumesWhereItStopped() throws {
        var reader = HTTPRequestReader(maxBody: 10)
        let head = "GET /health HTTP/1.1\r\nX-Pad: " + String(repeating: "a", count: 32_000) + "\r\n\r\n"
        var requests: [HTTPRequest] = []
        for byte in head.utf8 {
            reader.append(Data([byte]))
            if let request = try reader.next() { requests.append(request) }
        }
        #expect(requests.count == 1 && requests.first?.path == "/health")
        #expect(reader.searchedBytes < 5 * head.utf8.count, "searched \(reader.searchedBytes) bytes for a \(head.utf8.count)-byte head")
    }

    static let shortTimeouts = HTTPConnectionLimits(maxBody: 1 << 20, headTimeout: .seconds(1), bodyIdleTimeout: .seconds(1))

    /// A body that keeps arriving, slowly, is not cut by the timeout, however long it takes in total.
    /// These tests run on a ManualClock: the timeouts fire when the test moves time, not when a busy machine gets
    /// around to it. A connection that should end but does not hangs until the time limit fails the test.
    @Test(.timeLimit(.minutes(1))) func aSlowButActiveUploadIsNotCut() async throws {
        let body = Array(#"{"input":"slow"}"#.utf8)
        let clock = ManualClock()
        let transport = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
            try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the body's idle timeout
            for (index, byte) in body.enumerated() {
                // 16 bytes 0.9 s apart: 14.4 s in total, far past the 1 s head deadline, never 1 s idle.
                clock.advance(by: .milliseconds(900))
                let armed = clock.armed
                client.write(Data([byte]))
                if index < body.count - 1 { try await clock.waitUntilArmed(moreThan: armed) }
            }
        }
        let (responses, _) = parseResponses(transport.sent, methods: ["POST"])
        #expect(responses.map(\.status) == [200])
    }

    @Test(.timeLimit(.minutes(1))) func aStalledBodyTimesOut() async throws {
        let clock = ManualClock()
        let transport = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"input\"")
            try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the body's idle timeout
            clock.advance(by: .milliseconds(999))
            #expect(!client.isCancelled)
            clock.advance(by: .milliseconds(1))
        }
        #expect(transport.isCancelled && transport.sent.isEmpty)
    }

    /// The head has one deadline that progress does not extend: a head trickled byte by byte, or a kept-alive
    /// connection that sends nothing, cannot hold the connection for longer.
    @Test(.timeLimit(.minutes(1))) func aTrickledHeadAndAnIdleConnectionTimeOut() async throws {
        let head = Array("GET /health HTTP/1.1\r\nX-Slow: abcdefghijklmnopqrstuvwxyz\r\n\r\n".utf8)
        let clock = ManualClock()
        let trickled = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            try await clock.waitUntilArmed(moreThan: 0)  // the head deadline
            for byte in head.prefix(9) {  // a byte every 0.1 s keeps arriving until the deadline passes
                client.write(Data([byte]))
                clock.advance(by: .milliseconds(100))
            }
            #expect(!client.isCancelled)
            clock.advance(by: .milliseconds(100))
            while !client.isCancelled { try await Task.sleep(for: .milliseconds(1)) }
            client.write(Data(head.dropFirst(9)))  // too late: the connection is closed
        }
        #expect(trickled.isCancelled && trickled.sent.isEmpty)

        let idleClock = ManualClock()
        let idle = try await connection(limits: Self.shortTimeouts, clock: idleClock) { client in
            client.write("GET /health HTTP/1.1\r\n\r\n")
            try await idleClock.waitUntilArmed(moreThan: 1)  // the first request's head deadline, then the next one's
            idleClock.advance(by: .seconds(1))
        }
        #expect(idle.isCancelled && parseResponses(idle.sent, methods: ["GET"]).responses.map(\.status) == [200])
    }

    @Test func readerKeepsPipelinedBytes() throws {
        var reader = HTTPRequestReader(maxBody: 100)
        reader.append(Data("POST /a HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcGET /b HTTP/1.1\r\n\r\nGET /c".utf8))
        let first = try #require(try reader.next())
        #expect(first.path == "/a" && first.body == Data("abc".utf8))
        #expect(try reader.next()?.path == "/b")
        #expect(try reader.next() == nil)
        reader.append(Data(" HTTP/1.1\r\n\r\n".utf8))
        #expect(try reader.next()?.path == "/c")
        #expect(reader.bufferedBytes == 0)
    }
}
