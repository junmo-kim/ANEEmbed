// The load-time self-test: every loaded function runs once on golden inputs from the model package
// (selftest.json + selftest.bin, fp32 expectations from the original model) before it may serve.
//
// Why: the Neural Engine compiler can silently miscompile a graph (a constant attention band once came
// out wrong with no error), and the compile happens on the user's Mac, per macOS build. A function whose
// output does not match is not used; doctor shows which one and why.
//
// Inputs are rebuilt from the package's own files as selftest.json's "procedure" field describes:
//   bucket embed_S     the listed sequences packed into one call: cosine per row, unused rows ~0
//   winp winp{S}_b512  the same through the band buffers
//   winl winl{S}_b512  the first doc_rows document ids as one text: cosine of the embedding
//   vision_{B}         a procedural image (integer formula): digest of the image tokens
// A digest is D = rows @ R with R [512, 8] from selftest.bin; metrics are cosine and |Da - Dr| / |Dr|.
import Accelerate
import Foundation

public struct SelfTestResult: Sendable, Equatable {
    public var function: String
    public var kind: String
    public var passed: Bool
    public var cosine: Double
    public var relativeError: Double?
    public var unusedRowsMaxAbs: Double?
    public var error: String?

    public var json: JSONValue {
        var object: JSONObject = ["function": .string(function), "pass": .bool(passed),
                                  "cos": .number((cosine * 1e6).rounded() / 1e6)]
        if let relativeError { object["rel_err"] = .number((relativeError * 1e5).rounded() / 1e5) }
        if let unusedRowsMaxAbs { object["unused_max_abs"] = .number(unusedRowsMaxAbs) }
        if let error { object["error"] = .string(error) }
        return .object(object)
    }
}

/// What the last load found, kept next to the model for `doctor`.
public struct SelfTestReport: Sendable, Equatable {
    public var osBuild: String
    public var ranAt: Date
    public var seconds: Double
    public var results: [SelfTestResult]
    /// Whether the package carried self-test data at all.
    public var available: Bool

    public var failed: [SelfTestResult] { results.filter { !$0.passed } }

    public init(osBuild: String, ranAt: Date, seconds: Double, results: [SelfTestResult], available: Bool = true) {
        self.osBuild = osBuild
        self.ranAt = ranAt
        self.seconds = seconds
        self.results = results
        self.available = available
    }

    public var summary: String {
        guard available else { return "unavailable: the package has no self-test data" }
        let failed = failed
        return failed.isEmpty
            ? "\(results.count)/\(results.count) passed"
            : "\(results.count - failed.count)/\(results.count) passed; not served: \(failed.map(\.function).joined(separator: ", "))"
    }

    public var json: JSONValue {
        [
            "os_build": .string(osBuild), "ran_at": .string(ISO8601DateFormatter().string(from: ranAt)),
            "seconds": .number((seconds * 100).rounded() / 100), "summary": .string(summary),
            "available": .bool(available), "results": .array(results.map(\.json)),
        ]
    }

    public func save(to directory: URL) throws {
        try (json.render(as: .json) + "\n").write(to: directory.appendingPathComponent(ModelLayout.selfTestState),
                                                  atomically: true, encoding: .utf8)
    }

    public static func load(from directory: URL) -> SelfTestReport? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(ModelLayout.selfTestState)),
            let root = try? JSONValue.parse(data), let build = root["os_build"]?.stringValue
        else { return nil }
        let results = (root["results"]?.arrayValue ?? []).compactMap { item -> SelfTestResult? in
            guard let function = item["function"]?.stringValue, let passed = item["pass"]?.boolValue else { return nil }
            return SelfTestResult(function: function, kind: "", passed: passed, cosine: item["cos"]?.doubleValue ?? 0,
                                  relativeError: item["rel_err"]?.doubleValue, unusedRowsMaxAbs: item["unused_max_abs"]?.doubleValue,
                                  error: item["error"]?.stringValue)
        }
        return SelfTestReport(osBuild: build,
                              ranAt: root["ran_at"]?.stringValue.flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast,
                              seconds: root["seconds"]?.doubleValue ?? 0, results: results,
                              available: root["available"]?.boolValue ?? true)
    }
}

public enum SelfTestError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let message): return "selftest.json: \(message)" }
    }
}

/// selftest.json + selftest.bin of a model package.
public struct SelfTestSuite: Sendable {
    public struct Case: Sendable {
        public var function: String
        public var kind: String
        public var sequences: [[Int32]]
        public var documentRows: Int
        public var grid: (columns: Int, rows: Int)
        public var expectedOffset: Int
        public var expectedShape: [Int]
    }

    public struct Tolerance: Sendable {
        public var minCosine: Double
        public var maxRelativeError: Double?
        public var unusedRowsMaxAbs: Double?
    }

    public let cases: [Case]
    public let documentIDs: [Int32]
    public let tolerance: [String: Tolerance]
    let blob: Data
    let digest: [Float]  // [512, 8]
    static let digestWidth = 8

    public static func load(from directory: URL) throws -> SelfTestSuite? {
        let jsonURL = directory.appendingPathComponent(ModelLayout.selfTest)
        let binURL = directory.appendingPathComponent(ModelLayout.selfTestData)
        guard FileManager.default.fileExists(atPath: jsonURL.path), FileManager.default.fileExists(atPath: binURL.path) else {
            return nil
        }
        return try SelfTestSuite(json: try Data(contentsOf: jsonURL), blob: try Data(contentsOf: binURL, options: .alwaysMapped))
    }

    public init(json: Data, blob: Data) throws {
        let root = try JSONValue.parse(json)
        guard root["format"]?.stringValue == "aneembed-selftest/1" else { throw SelfTestError.invalid("unknown format") }
        self.blob = blob
        func block(_ value: JSONValue?) throws -> (Int, [Int]) {
            guard let offset = value?["offset"]?.intValue, let shape = value?["shape"]?.arrayValue?.compactMap(\.intValue),
                offset >= 0, offset + shape.reduce(4, *) <= blob.count
            else { throw SelfTestError.invalid("bad data block") }
            return (offset, shape)
        }
        let (digestOffset, digestShape) = try block(root["digest"])
        guard digestShape == [BucketContract.hidden, Self.digestWidth] else { throw SelfTestError.invalid("digest shape \(digestShape)") }
        digest = Self.floats(blob, offset: digestOffset, count: digestShape.reduce(1, *))
        documentIDs = (root["doc_ids"]?.arrayValue ?? []).compactMap { $0.intValue.map(Int32.init) }
        var tolerance: [String: Tolerance] = [:]
        for (kind, value) in root["tolerance"]?.objectValue?.pairs ?? [] {
            guard let minCosine = value["min_cos"]?.doubleValue else { continue }
            tolerance[kind] = Tolerance(minCosine: minCosine, maxRelativeError: value["max_rel_err"]?.doubleValue,
                                        unusedRowsMaxAbs: value["zero_rows_max_abs"]?.doubleValue)
        }
        self.tolerance = tolerance
        var cases: [Case] = []
        for test in root["tests"]?.arrayValue ?? [] {
            guard let function = test["function"]?.stringValue, let kind = test["kind"]?.stringValue else {
                throw SelfTestError.invalid("test without function or kind")
            }
            let (offset, shape) = try block(test["expected"])
            let grid = test["grid"]?.arrayValue?.compactMap(\.intValue) ?? []
            cases.append(Case(
                function: function, kind: kind,
                sequences: (test["sequences"]?.arrayValue ?? []).map { ($0.arrayValue ?? []).compactMap { $0.intValue.map(Int32.init) } },
                documentRows: test["doc_rows"]?.intValue ?? 0,
                grid: grid.count == 2 ? (grid[0], grid[1]) : (0, 0), expectedOffset: offset, expectedShape: shape))
        }
        self.cases = cases
    }

    static func floats(_ data: Data, offset: Int, count: Int) -> [Float] {
        data.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + $0 * 4, as: UInt32.self))) }
        }
    }

    func expected(_ test: Case) -> [Float] {
        Self.floats(blob, offset: test.expectedOffset, count: test.expectedShape.reduce(1, *))
    }

    /// rows [count, 512] @ digest [512, 8]
    func digest(of rows: [Float], count: Int) -> [Float] {
        var output = [Float](repeating: 0, count: count * Self.digestWidth)
        vDSP_mmul(rows, 1, digest, 1, &output, 1, vDSP_Length(count), vDSP_Length(Self.digestWidth), vDSP_Length(BucketContract.hidden))
        return output
    }

    static func metrics(_ actual: [Float], _ reference: [Float]) -> (cosine: Double, relativeError: Double) {
        var dot = 0.0, aa = 0.0, bb = 0.0, diff = 0.0
        for (a, b) in zip(actual, reference) {
            let x = Double(a), y = Double(b)
            dot += x * y
            aa += x * x
            bb += y * y
            diff += (x - y) * (x - y)
        }
        let cosine = aa > 0 && bb > 0 ? dot / (aa.squareRoot() * bb.squareRoot()) : 0
        return (cosine.isFinite ? cosine : 0, bb > 0 ? (diff / bb).squareRoot() : .infinity)
    }

    /// The procedural test image: value(X, Y, c) = (5X + 3Y + 70c + 90 * ((X / 24 + Y / 24) % 2)) % 256.
    public static func pattern(columns: Int, rows: Int) -> PreparedImage {
        let width = columns * 16, height = rows * 16
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    rgb[(y * width + x) * 3 + c] = UInt8((5 * x + 3 * y + 70 * c + 90 * ((x / 24 + y / 24) % 2)) % 256)
                }
            }
        }
        return PreparedImage(rgb: rgb, width: width, height: height)
    }

    /// Runs the case of every function that is loaded. Functions without a case are not reported.
    public func run(buckets: (any BucketFunctionRunner)?, long: (any LongFunctionRunner)?, vision: (any VisionEncoder)?,
                    visionBudgets: [Int] = [], rows: any TokenRowSource) async -> [SelfTestResult]
    {
        var results: [SelfTestResult] = []
        for test in cases {
            let tolerance = self.tolerance[test.kind] ?? Tolerance(minCosine: 0.999, maxRelativeError: 0.05)
            var result = SelfTestResult(function: test.function, kind: test.kind, passed: false, cosine: 0)
            do {
                switch test.kind {
                case "bucket", "winp":
                    guard let buckets, let length = buckets.lengths.first(where: {
                        BucketContract.functionName(forLength: $0) == test.function
                    }) else { continue }
                    try await runBucket(test, length: length, runner: buckets, rows: rows, into: &result)
                case "winl":
                    guard let long, let length = long.lengths.first(where: { WindowContract.longName($0) == test.function })
                    else { continue }
                    try await runLong(test, length: length, runner: long, rows: rows, into: &result)
                case "vision":
                    guard let vision, let budget = Int(test.function.dropFirst("vision_".count)),
                        visionBudgets.contains(budget)
                    else { continue }
                    let image = Self.pattern(columns: test.grid.columns, rows: test.grid.rows)
                    let tokens = try await vision.encode(image)
                    let valid = test.expectedShape[0]
                    guard tokens.count >= valid * BucketContract.hidden else { throw RuntimeError.model("too few image tokens") }
                    let actual = digest(of: tokens.prefix(valid * BucketContract.hidden).map(Float.init), count: valid)
                    (result.cosine, result.relativeError) = Self.metrics(actual, expected(test))
                default:
                    continue
                }
                result.passed = Self.passes(result, tolerance)
            } catch {
                result.error = "\(error)"
            }
            results.append(result)
        }
        return results
    }

    /// Every metric must be finite and inside the tolerance: a NaN or an infinity anywhere fails, never passes
    /// through a comparison that is false for NaN.
    static func passes(_ result: SelfTestResult, _ tolerance: Tolerance) -> Bool {
        guard result.cosine.isFinite, result.cosine >= tolerance.minCosine else { return false }
        if let error = result.relativeError, !error.isFinite { return false }
        if let maxAbs = result.unusedRowsMaxAbs, !maxAbs.isFinite { return false }
        if let limit = tolerance.maxRelativeError {
            guard let error = result.relativeError, error <= limit else { return false }
        }
        if let limit = tolerance.unusedRowsMaxAbs {
            guard let maxAbs = result.unusedRowsMaxAbs, maxAbs <= limit else { return false }
        }
        return true
    }

    /// The largest |value|, or infinity when any value is NaN or infinite (`max()` skips NaN).
    static func maxAbs(_ rows: some Sequence<[Float]>) -> Double {
        var largest: Float = 0
        for row in rows {
            for value in row {
                guard value.isFinite else { return .infinity }
                largest = Swift.max(largest, abs(value))
            }
        }
        return Double(largest)
    }

    private func runBucket(_ test: Case, length: Int, runner: any BucketFunctionRunner, rows: any TokenRowSource,
                           into result: inout SelfTestResult) async throws
    {
        var offset = 0
        let segments = test.sequences.enumerated().map { index, ids -> PackedSegment in
            defer { offset += ids.count }
            return PackedSegment(item: index, offset: offset, length: ids.count)
        }
        let buffers = BucketBuffers(length: length)
        buffers.prepare(PackedCall(length: length, segments: segments), ids: test.sequences, rows: rows)
        let output = try await runner.run(buffers)
        let reference = expected(test)
        var worst = 1.0
        for row in test.sequences.indices {
            let cosine = Self.metrics(output[row], Array(reference[(row * 768)..<((row + 1) * 768)])).cosine
            if !(cosine >= worst) { worst = cosine }  // keeps a NaN (Swift's min(1.0, .nan) is 1.0)
        }
        result.cosine = worst
        result.unusedRowsMaxAbs = Self.maxAbs(output.dropFirst(test.sequences.count))
    }

    private func runLong(_ test: Case, length: Int, runner: any LongFunctionRunner, rows: any TokenRowSource,
                         into result: inout SelfTestResult) async throws
    {
        let n = test.documentRows
        guard n > 0, n <= documentIDs.count, n <= length else { throw SelfTestError.invalid("doc_rows \(n)") }
        let buffers = LongBuffers(length: length)
        buffers.prepare(ids: Array(documentIDs.prefix(n)), rows: rows)
        let embedding = try await runner.run(buffers)
        result.cosine = Self.metrics(embedding, expected(test)).cosine
    }
}
