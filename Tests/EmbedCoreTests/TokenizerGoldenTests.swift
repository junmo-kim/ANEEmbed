import Foundation
import Testing

@testable import EmbedCore

/// The real EmbeddingGemma 2 tokenizer, loaded once.
enum RealTokenizer {
    static let shared: GemmaTokenizer? = try? GemmaTokenizer(data: Fixtures.data("tokenizer.json.xz"))
}

@Suite("Tokenizer golden (Hugging Face tokenizers)")
struct TokenizerGoldenTests {
    @Test func loadsTheRealTokenizer() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        #expect(tokenizer.vocabularySize == 262_144)
        #expect(tokenizer.mergeCount == 514_906)
        #expect(tokenizer.prefixIDs == [2])
        #expect(tokenizer.suffixIDs == [1])
        #expect(tokenizer.encode("hello world") == [2, 23391, 1902, 1])
    }

    @Test func matchesEveryGoldenText() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let lines = try Fixtures.data("tokenizer-golden.jsonl.xz").split(separator: UInt8(ascii: "\n"))
        var checked = 0, ids = 0
        var mismatches: [String] = []
        var scratch = BPEScratch()
        var output: [Int32] = []
        for line in lines where !line.isEmpty {
            let record = try JSONValue.parse(Data(line))
            let text = try #require(record["text"]?.stringValue)
            let expected = try #require(record["ids"]?.arrayValue).map { Int32($0.intValue!) }
            output.removeAll(keepingCapacity: true)
            tokenizer.encode(text, into: &output, scratch: &scratch)
            if output != expected, mismatches.count < 10 {
                mismatches.append("\(text.debugDescription.prefix(80)): got \(output.prefix(12)) want \(expected.prefix(12))")
            }
            checked += 1
            ids += expected.count
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
        #expect(checked > 10_000)
        print("tokenizer golden: \(checked) texts, \(ids) ids, \(mismatches.count == 0 ? "all equal" : "MISMATCH")")
    }

    /// Optional large local corpus (not stored in the repository), e.g. from
    /// `make_tokenizer_golden.py --lines`: ANEEMBED_EXTRA_GOLDEN=/path/to/golden.jsonl[.xz]
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ANEEMBED_EXTRA_GOLDEN"] != nil))
    func matchesExtraGoldenCorpus() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let path = ProcessInfo.processInfo.environment["ANEEMBED_EXTRA_GOLDEN"]!
        var data = try Data(contentsOf: URL(fileURLWithPath: path))
        if path.hasSuffix(".xz") { data = try (data as NSData).decompressed(using: .lzma) as Data }
        var checked = 0, failed = 0, ids = 0
        for line in data.split(separator: UInt8(ascii: "\n")) where !line.isEmpty {
            let record = try JSONValue.parse(Data(line))
            let expected = record["ids"]!.arrayValue!.map { Int32($0.intValue!) }
            if tokenizer.encode(record["text"]!.stringValue!) != expected {
                failed += 1
                if failed <= 5 { Issue.record("mismatch: \(record["text"]!.stringValue!.debugDescription.prefix(100))") }
            }
            checked += 1
            ids += expected.count
        }
        print("extra tokenizer golden: \(checked) texts, \(ids) ids, \(failed) mismatches")
        #expect(failed == 0)
    }
}
