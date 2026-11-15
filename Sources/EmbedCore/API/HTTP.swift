// HTTP/1.1 message parsing and serialization, independent of the socket layer so it can be tested.
// Requests need Content-Length when they carry a body (chunked uploads are refused with 411).
import Foundation

public struct HTTPRequest: Sendable {
    public var method: String
    /// Path without the query string, percent-decoded.
    public var path: String
    public var query: [(name: String, value: String)]
    public var headers: [(name: String, value: String)]
    public var body: Data
    public var version: String

    public init(method: String, path: String, query: [(name: String, value: String)] = [],
                headers: [(name: String, value: String)] = [], body: Data = Data(), version: String = "HTTP/1.1")
    {
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body
        self.version = version
    }

    public func header(_ name: String) -> String? {
        headers.last { $0.name.caseInsensitiveCompare(name) == .orderedSame }?.value
    }

    public func queryValue(_ name: String) -> String? {
        query.last { $0.name == name }?.value
    }

    /// HTTP/1.1 keeps the connection unless told otherwise; HTTP/1.0 only when asked.
    public var keepAlive: Bool {
        let connection = header("Connection")?.lowercased() ?? ""
        if version == "HTTP/1.0" { return connection.contains("keep-alive") }
        return !connection.contains("close")
    }
}

public struct HTTPResponse: Sendable {
    public var status: Int
    public var headers: [(name: String, value: String)]
    public var body: Data

    public init(status: Int, headers: [(name: String, value: String)] = [], body: Data = Data()) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public static func json(_ status: Int, _ value: JSONValue) -> HTTPResponse {
        HTTPResponse(status: status, headers: [("Content-Type", "application/json")], body: value.jsonData())
    }

    public static func json(_ status: Int, raw: Data) -> HTTPResponse {
        HTTPResponse(status: status, headers: [("Content-Type", "application/json")], body: raw)
    }

    public static func openAIError(_ error: ServiceError) -> HTTPResponse {
        .json(error.httpStatus, error.openAIBody)
    }

    public func header(_ name: String) -> String? {
        headers.last { $0.name.caseInsensitiveCompare(name) == .orderedSame }?.value
    }

    public static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 202: return "Accepted"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 408: return "Request Timeout"
        case 411: return "Length Required"
        case 413: return "Content Too Large"
        case 415: return "Unsupported Media Type"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    /// The response on the wire. `includeBody: false` answers a HEAD request: the same headers, Content-Length of
    /// the body a GET would get included, and no body (RFC 9110 section 9.3.2), so the next response on a kept-alive
    /// connection starts where the client expects it.
    public func serialized(keepAlive: Bool, includeBody: Bool = true) -> Data {
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        for (name, value) in headers { head += "\(name): \(value)\r\n" }
        head += "Content-Length: \(body.count)\r\n"
        head += keepAlive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n"
        var data = Data(head.utf8)
        if includeBody { data.append(body) }
        return data
    }
}

public enum HTTPParseError: Error, Equatable, Sendable {
    case malformed(String)
    case headerTooLarge
    case bodyTooLarge(Int)
    case lengthRequired

    public var status: Int {
        switch self {
        case .malformed: return 400
        case .headerTooLarge: return 431
        case .bodyTooLarge: return 413
        case .lengthRequired: return 411
        }
    }

    public var message: String {
        switch self {
        case .malformed(let detail): return "malformed request: \(detail)"
        case .headerTooLarge: return "request headers are too large"
        case .bodyTooLarge(let limit): return "request body exceeds \(limit) bytes"
        case .lengthRequired: return "Content-Length is required (chunked bodies are not supported)"
        }
    }
}

/// The parsed request line and headers, before the body has arrived.
public struct HTTPHead: Sendable {
    public var request: HTTPRequest
    /// Bytes of the head including the blank line.
    public var headLength: Int
    public var contentLength: Int
}

public enum HTTPParser {
    public static let maxHeadBytes = 64 << 10

    /// Parses the head once `buffer` holds it; nil while more bytes are needed.
    public static func parseHead(_ buffer: Data, maxBody: Int) throws(HTTPParseError) -> HTTPHead? {
        guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            if buffer.count > maxHeadBytes { throw .headerTooLarge }
            return nil
        }
        let headLength = end.upperBound - buffer.startIndex
        guard headLength <= maxHeadBytes + 4 else { throw .headerTooLarge }
        guard let head = String(data: buffer[buffer.startIndex..<end.lowerBound], encoding: .utf8) else {
            throw .malformed("head is not UTF-8")
        }
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: false)
        guard requestLine.count == 3, !requestLine[0].isEmpty, requestLine[2].hasPrefix("HTTP/1.") else {
            throw .malformed("bad request line")
        }
        var headers: [(name: String, value: String)] = []
        for line in lines {
            guard let colon = line.firstIndex(of: ":"), colon != line.startIndex else {
                throw .malformed("bad header line")
            }
            let name = String(line[..<colon])
            guard !name.contains(" ") else { throw .malformed("bad header name") }
            headers.append((name: name, value: line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)))
        }
        var request = HTTPRequest(method: String(requestLine[0]), path: "/", headers: headers,
                                  version: String(requestLine[2]))
        let target = String(requestLine[1])
        let (path, query) = splitTarget(target)
        guard path.hasPrefix("/") else { throw .malformed("request target must be a path") }
        request.path = path
        request.query = query

        if let encoding = request.header("Transfer-Encoding"), encoding.lowercased() != "identity" {
            throw .lengthRequired
        }
        var length = 0
        if let raw = request.header("Content-Length") {
            guard let value = Int(raw), value >= 0 else { throw .malformed("bad Content-Length") }
            length = value
        }
        guard length <= maxBody else { throw .bodyTooLarge(maxBody) }
        return HTTPHead(request: request, headLength: headLength, contentLength: length)
    }

    /// Path and decoded query pairs of a request target.
    public static func splitTarget(_ target: String) -> (String, [(name: String, value: String)]) {
        let parts = target.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        let path = String(parts[0]).removingPercentEncoding ?? String(parts[0])
        guard parts.count == 2 else { return (path, []) }
        let query = parts[1].split(separator: "&").map { pair -> (name: String, value: String) in
            let kv = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            func decode(_ s: Substring) -> String {
                String(s).replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? String(s)
            }
            return (name: decode(kv[0]), value: kv.count > 1 ? decode(kv[1]) : "")
        }
        return (path, query)
    }
}
