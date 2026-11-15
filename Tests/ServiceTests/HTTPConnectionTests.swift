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
    func connection(limits: HTTPConnectionLimits = HTTPConnectionLimits(maxBody: 1 << 20),
                    _ client: @Sendable (MemoryTransport) async throws -> Void) async throws -> MemoryTransport
    {
        let transport = MemoryTransport()
        let server = Task { await HTTPConnection.serve(transport, limits: limits) { await Self.router.handle($0) } }
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
