import Foundation
import Testing

@testable import EmbedCore

/// A bucket runner that returns fixed rows (lengths: embed_64 only by default).
struct StubBucketRunner: BucketFunctionRunner {
    var lengths: [Int] = [64]
    let rows: @Sendable () -> [[Float]]

    init(lengths: [Int] = [64], rows: @escaping @Sendable () -> [[Float]]) {
        self.lengths = lengths
        self.rows = rows
    }

    func run(_ buffers: BucketBuffers) async throws -> [[Float]] { rows() }
    func warm() async throws {}
}

@Suite("Self-test data")
struct SelfTestTests {
    /// A two-case suite: little-endian float32 blocks addressed by offset, as the package's selftest.bin stores them.
    func suite() throws -> SelfTestSuite {
        var blob = Data()
        func put(_ values: [Float]) -> JSONValue {
            let offset = blob.count
            for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { blob.append(contentsOf: $0) } }
            return ["offset": .int(offset), "shape": .array([.int(values.count / 8), 8])]
        }
        let digest = put((0..<(512 * 8)).map { Float($0 % 8 == 0 ? 1 : 0) })
        let expected = put([Float](repeating: 0.5, count: 16))
        let json: JSONValue = [
            "format": "aneembed-selftest/1", "digest": digest, "doc_ids": [2, 10, 11, 1],
            "tolerance": ["swa": ["min_cos": 0.999, "max_rel_err": 0.05], "bucket": ["min_cos": 0.999, "zero_rows_max_abs": 0.001]],
            "tests": [["function": "long_swa_l0", "kind": "swa", "doc_rows": 4, "chunk": 1, "valid_rows": 2, "expected": expected]],
        ]
        return try SelfTestSuite(json: json.jsonData(), blob: blob)
    }

    @Test func parsesCasesTolerancesAndBlocks() throws {
        let suite = try suite()
        #expect(suite.cases.count == 1 && suite.cases[0].function == "long_swa_l0" && suite.cases[0].documentRows == 4)
        #expect(suite.documentIDs == [2, 10, 11, 1])
        #expect(suite.tolerance["swa"]?.maxRelativeError == 0.05 && suite.tolerance["bucket"]?.unusedRowsMaxAbs == 0.001)
        #expect(suite.expected(suite.cases[0]) == [Float](repeating: 0.5, count: 16))
        // digest: column 0 of R is all ones, so D[:, 0] is the row sum
        let rows = [Float](repeating: 2, count: 512) + [Float](repeating: 1, count: 512)
        let digest = suite.digest(of: rows, count: 2)
        #expect(digest[0] == 1024 && digest[8] == 512 && digest[1] == 0)
    }

    @Test func rejectsBlocksOutsideTheData() throws {
        let json: JSONValue = ["format": "aneembed-selftest/1", "digest": ["offset": 0, "shape": [512, 8]], "tests": []]
        #expect(throws: (any Error).self) { try SelfTestSuite(json: json.jsonData(), blob: Data(count: 16)) }
    }

    @Test func metrics() {
        let (cosine, error) = SelfTestSuite.metrics([1, 0, 0], [1, 0, 0])
        #expect(cosine == 1 && error == 0)
        let (opposite, _) = SelfTestSuite.metrics([-1, 0], [1, 0])
        #expect(opposite == -1)
    }

    /// The procedural image is exact integer arithmetic: spot values from the formula.
    @Test func patternImage() {
        let image = SelfTestSuite.pattern(columns: 27, rows: 18)
        #expect(image.width == 432 && image.height == 288 && image.tokens == 54)
        func value(_ x: Int, _ y: Int, _ c: Int) -> UInt8 { image.rgb[(y * image.width + x) * 3 + c] }
        #expect(value(0, 0, 0) == 0 && value(0, 0, 2) == 140)
        #expect(value(24, 0, 0) == UInt8((5 * 24 + 90) % 256))
        let corner: Int = (5 * 431 + 3 * 287 + 70 + 90 * ((17 + 11) % 2)) % 256
        #expect(value(431, 287, 1) == UInt8(corner))
    }

    /// One embed_64 case: `rows` sequences with expectation rows `reference` (768 values each).
    func bucketSuite(sequences: [[Int32]], reference: [[Float]]) throws -> SelfTestSuite {
        var blob = Data()
        func put(_ values: [Float], shape: [Int]) -> JSONValue {
            let offset = blob.count
            for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { blob.append(contentsOf: $0) } }
            return ["offset": .int(offset), "shape": .array(shape.map { .int($0) })]
        }
        let digest = put([Float](repeating: 0, count: 512 * 8), shape: [512, 8])
        let expected = put(reference.flatMap { $0 }, shape: [reference.count, 768])
        let json: JSONValue = [
            "format": "aneembed-selftest/1", "digest": digest,
            "tolerance": ["bucket": ["min_cos": 0.999, "zero_rows_max_abs": 0.001]],
            "tests": [["function": "embed_64", "kind": "bucket",
                       "sequences": .array(sequences.map { .array($0.map { .int(Int($0)) }) }), "expected": expected]],
        ]
        return try SelfTestSuite(json: json.jsonData(), blob: blob)
    }

    static let unit: [Float] = [1] + [Float](repeating: 0, count: 767)

    /// A function whose rows are NaN (used rows, then unused rows) must fail, never pass on a NaN comparison.
    @Test func nonFiniteOutputFails() async throws {
        let suite = try bucketSuite(sequences: [[2, 10, 1]], reference: [Self.unit])
        let nan = [Float](repeating: .nan, count: 768)
        let zero = [Float](repeating: 0, count: 768)
        let good = await suite.run(buckets: StubBucketRunner { [Self.unit] + [[Float]](repeating: zero, count: 7) }, long: nil,
                                   vision: nil, rows: HashedTokenRows())
        #expect(good.map(\.passed) == [true])
        let allNaN = await suite.run(buckets: StubBucketRunner { [[Float]](repeating: nan, count: 8) }, long: nil, vision: nil,
                                     rows: HashedTokenRows())
        #expect(allNaN.map(\.passed) == [false])
        var unusedNaN = [Self.unit] + [[Float]](repeating: zero, count: 7)
        unusedNaN[3][1] = .nan
        let unused = await suite.run(buckets: StubBucketRunner { [unusedNaN] in unusedNaN }, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(unused.map(\.passed) == [false])
        var unusedInfinity = [Self.unit] + [[Float]](repeating: zero, count: 7)
        unusedInfinity[7][0] = -.infinity
        let infinite = await suite.run(buckets: StubBucketRunner { [unusedInfinity] in unusedInfinity }, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(infinite.map(\.passed) == [false])
    }

    @Test func reportRoundTripsAndSummarizes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-selftest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let report = SelfTestReport(osBuild: "26A434", ranAt: Date(timeIntervalSince1970: 1_800_000_000), seconds: 1.25, results: [
            SelfTestResult(function: "embed_64", kind: "bucket", passed: true, cosine: 0.99997),
            SelfTestResult(function: "long_swa_l3", kind: "swa", passed: false, cosine: 0.55, relativeError: 0.9),
        ])
        try report.save(to: directory)
        let loaded = try #require(SelfTestReport.load(from: directory))
        #expect(loaded.results.map(\.function) == ["embed_64", "long_swa_l3"] && loaded.failed.map(\.function) == ["long_swa_l3"])
        #expect(loaded.summary == "1/2 passed; not served: long_swa_l3")
    }
}
