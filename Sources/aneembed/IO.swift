import EmbedCore
import Foundation

/// stdout carries data only; everything else goes to stderr.
enum Output {
    private static let lock = NSLock()

    static func data(_ text: String) {
        write(text.hasSuffix("\n") ? text : text + "\n", to: FileHandle.standardOutput)
    }

    static func data(_ value: JSONValue, format: OutputFormat) {
        data(value.render(as: format))
    }

    static func write(_ text: String, to handle: FileHandle) {
        write(Data(text.utf8), to: handle)
    }

    /// POSIX write(2) under one lock, so concurrent writers (MCP responses) never interleave.
    static func write(_ bytes: Data, to handle: FileHandle) {
        let descriptor = handle.fileDescriptor
        lock.withLock {
            bytes.withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    let written = Darwin.write(descriptor, raw.baseAddress! + offset, raw.count - offset)
                    if written < 0 {
                        if errno == EINTR { continue }
                        return
                    }
                    offset += written
                }
            }
        }
    }
}

enum Log {
    static func info(_ message: String) {
        let stamp = Date().formatted(.iso8601)
        Output.write("\(stamp) \(message)\n", to: FileHandle.standardError)
    }
}

/// Reads stdin as raw bytes and yields lines without decoding (so a U+FEFF inside a JSON string, or
/// any other byte sequence, reaches the JSON parser unchanged). Blocking reads run on their own thread.
func standardInputLines() -> AsyncStream<Data> {
    let (stream, continuation) = AsyncStream.makeStream(of: Data.self, bufferingPolicy: .unbounded)
    let thread = Thread {
        var pending: [UInt8] = []
        var start = 0
        var buffer = [UInt8](repeating: 0, count: 1 << 16)
        func emit(_ range: Range<Int>) {
            var end = range.upperBound
            if end > range.lowerBound && pending[end - 1] == UInt8(ascii: "\r") { end -= 1 }
            // A copy of just this line: a slice would keep the whole read buffer alive.
            continuation.yield(pending.withUnsafeBytes { Data(bytes: $0.baseAddress! + range.lowerBound, count: end - range.lowerBound) })
        }
        while true {
            let count = read(0, &buffer, buffer.count)
            if count <= 0 { break }
            pending.append(contentsOf: buffer[0..<count])
            while let newline = pending[start...].firstIndex(of: UInt8(ascii: "\n")) {
                emit(start..<newline)
                start = newline + 1
            }
            if start > 0 {
                pending.removeFirst(start)
                start = 0
            }
        }
        if start < pending.count { emit(start..<pending.count) }
        continuation.finish()
    }
    thread.stackSize = 1 << 20
    thread.start()
    return stream
}

/// All of stdin as bytes.
func readStandardInput() -> Data {
    FileHandle.standardInput.readDataToEndOfFile()
}
