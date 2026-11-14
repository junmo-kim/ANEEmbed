import Foundation
import Testing

@testable import EmbedCore

@Suite("Embedding service (fake runtime)")
struct ServiceTests {
    @Test func aTextGetsTheSameVectorAloneOrPacked() async throws {
        let service = TestSupport.service()
        let texts = (0..<30).map { "text number \($0) " + String(repeating: "x", count: $0 * 7) }
        let packed = try await service.embedTexts(texts)
        for (index, text) in texts.enumerated() {
            let alone = try await service.embedTexts([text])
            #expect(alone.vectors[0] == packed.vectors[index])
        }
        #expect(packed.vectors.allSatisfy { abs(TestSupport.norm($0) - 1) < 1e-4 && $0.count == 768 })
    }

    @Test func longAndShortTextsKeepTheirOrder() async throws {
        let service = TestSupport.service()
        let texts = ["short", String(repeating: "long ", count: 300), "also short", String(repeating: "y", count: 2000)]
        let result = try await service.embedTexts(texts)
        #expect(result.tokenCounts == texts.map { $0.utf8.count + 2 })
        for (index, text) in texts.enumerated() {
            #expect(try await service.embedTexts([text]).vectors[0] == result.vectors[index])
        }
    }

    @Test func rejectsTextsOverTheLimitWithoutTruncating() async {
        let service = TestSupport.service(maxTokens: 512)
        await #expect(throws: ServiceError.self) { try await service.embedTexts(["ok", String(repeating: "z", count: 600)]) }
        do {
            _ = try await service.embedTexts([String(repeating: "z", count: 600)])
        } catch {
            #expect(error.reason == "context_length_exceeded")
            #expect(error.httpStatus == 400)
            #expect(error.exitCode == .dataError)
            #expect(error.message.contains("512"))
        }
    }

    /// A tokenizer without <bos>/<eos> framing (post_processor null) turns "" into no ids: a 400, not a trap.
    @Test func anEmptyTokenSequenceIsRejected() async throws {
        struct Unframed: TextTokenizer { func encode(_ text: String) -> [Int32] { text.utf8.map { Int32($0) + 3 } } }
        let service = EmbeddingService(tokenizer: Unframed(), runtime: FakeRuntime.make())
        do {
            _ = try await service.embedTexts(["ok", ""])
            Issue.record("an empty token sequence was embedded")
        } catch {
            #expect(error.httpStatus == 400)
        }
        await #expect(throws: RuntimeError.self) { _ = try await FakeRuntime.make().embed(tokenized: [[2, 1], []]) }
    }

    /// The fake runtime loads the same functions as the Core ML one: every bucket up to the smallest that holds
    /// --max-tokens, so a limit between bucket sizes accepts texts up to the limit itself.
    @Test func fakeRuntimeLoadsTheBucketsCoreMLWould() async throws {
        for maxTokens in [2, 64, 65, 100, 1300, 2048, 8192] {
            #expect(FakeRuntime.make(maxTokens: maxTokens).info.bucketLengths
                    == CoreMLBucketRunner.lengths(forMaxTokens: maxTokens))
        }
        let service = TestSupport.service(maxTokens: 100)
        #expect(service.maxTokens == 100)
        let result = try await service.embedTexts([String(repeating: "a", count: 90)])
        #expect(result.tokenCounts == [92])
    }

    @Test func rejectsEmptyAndOversizedBatches() async {
        var limits = ServiceLimits()
        limits.maxInputs = 3
        let service = TestSupport.service(limits: limits)
        await #expect(throws: ServiceError.badRequest("empty_input", "input must not be empty", param: "input")) {
            try await service.embedTexts([])
        }
        await #expect(throws: ServiceError.self) { try await service.embedTexts(["a", "b", "c", "d"]) }
    }

    @Test func inputTypeAppliesThePrefix() async throws {
        let service = TestSupport.service()
        let prefixed = try await service.embedTexts(["hello"], inputType: .query)
        let manual = try await service.embedTexts(["task: search result | query: hello"])
        #expect(prefixed == manual)
    }

    @Test func imagesUseTheOfficialCanvasAndCountTokens() async throws {
        let service = TestSupport.service()
        let small = TestSupport.png(width: 320, height: 240)
        let big = TestSupport.png(width: 1600, height: 1200)
        let tiny = TestSupport.png(width: 1, height: 1)
        let strip = TestSupport.png(width: 3000, height: 10)
        let result = try await service.embedImages([small, big, tiny, strip])
        #expect(result.tokenCounts == [266 + 4, 266 + 4, 256 + 4, 280 + 4])
        #expect(Vectors.cosine(result.vectors[0], result.vectors[1]) > 0.999)
    }

    @Test func rejectsBadImagesBeforeAnyModelCall() async {
        let service = TestSupport.service()
        do {
            _ = try await service.embedImages([TestSupport.png(width: 4, height: 4), Data([0, 0, 0])])
            Issue.record("expected an error")
        } catch {
            #expect(error.reason == "image_decode_failed")
            #expect(error.message.hasPrefix("input[1]"))
        }
        let many = [Data](repeating: TestSupport.png(width: 2, height: 2), count: 9)
        await #expect(throws: ServiceError.self) { try await service.embedImages(many) }
        let textOnly = TestSupport.service(images: false)
        await #expect(throws: ServiceError.self) { try await textOnly.embedImages([TestSupport.png(width: 2, height: 2)]) }
    }

    @Test func handlesManyConcurrentRequests() async throws {
        let service = TestSupport.service()
        try await withThrowingTaskGroup(of: Int.self) { group in
            for index in 0..<64 {
                group.addTask { try await service.embedTexts(["query \(index)"]).vectors.count }
            }
            var total = 0
            for try await count in group { total += count }
            #expect(total == 64)
        }
    }
}
