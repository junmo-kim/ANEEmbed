import Foundation
import Testing

@testable import EmbedCore

/// Real-model checks on the Neural Engine. They run only when a model package exists
/// (ANEEMBED_TEST_MODEL_DIR, or the installed current revision) and never on CI runners
/// (GitHub's virtualized macOS runners have no Neural Engine).
enum ModelGate {
    static var directory: URL? {
        let environment = ProcessInfo.processInfo.environment
        guard environment["CI"] == nil else { return nil }
        let url: URL
        if let path = environment["ANEEMBED_TEST_MODEL_DIR"] {
            url = URL(fileURLWithPath: path)
        } else {
            url = ModelStore(dataDirectory: Config.defaultDataDirectory).currentDirectory
        }
        return ModelLayout.model(ModelLayout.textModel, in: url) != nil ? url : nil
    }

    static var enabled: Bool { directory != nil }

    static func service(maxTokens: Int = 2048) async throws -> EmbeddingService {
        let config = try Config(flags: ["model-dir": directory!.path, "max-tokens": String(maxTokens)], environment: [:])
        return try await ModelLoader.makeService(config)
    }
}

@Suite("Model on the Neural Engine", .serialized, .enabled(if: ModelGate.enabled, "needs a model package and no CI"))
struct ModelTests {
    @Test func everyLoadedFunctionPassesTheSelfTest() async throws {
        let config = try Config(flags: ["model-dir": ModelGate.directory!.path], environment: [:])
        let (service, report) = try await ModelLoader.load(config)
        let selfTest = try #require(report.selfTest)
        #expect(selfTest.available && selfTest.failed.isEmpty, "\(selfTest.summary)")
        #expect(Set(selfTest.results.map(\.function)) == Set(service.runtime.info.functions))
        #expect(service.runtime.info.functions.count == 9)  // six buckets, three vision budgets
    }

    @Test func longTextsRunThroughTheLongPath() async throws {
        let service = try await ModelGate.service(maxTokens: 8192)
        #expect(service.maxTokens == 8192 && service.runtime.info.functions.count == 37)
        let sentence = "The Neural Engine runs fixed-shape graphs, so long documents are split into chunks. "
        let texts = [String(repeating: sentence, count: 130), String(repeating: sentence, count: 330)]
        let result = try await service.embedTexts(texts)
        #expect(result.tokenCounts[0] > 2048 && result.tokenCounts[0] <= 4096)
        #expect(result.tokenCounts[1] > 4096 && result.tokenCounts[1] <= 8192)
        for vector in result.vectors { #expect(abs(vector.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-3) }
        // Same text again (reused buffers and cached biases) gives the same vector.
        let again = try await service.embedTexts([texts[1]])
        #expect(Vectors.cosine(again.vectors[0], result.vectors[1]) > 0.99999)
        // Without --max-tokens above 2048 the long functions are not loaded and such texts are rejected.
        await #expect(throws: ServiceError.self) {
            _ = try await ModelGate.service(maxTokens: 2048).embedTexts([texts[0]])
        }
    }

    @Test func embedsUnitVectors() async throws {
        let service = try await ModelGate.service()
        let result = try await service.embedTexts(["hello world", "task: search result | query: 서울 날씨"])
        for vector in result.vectors {
            #expect(vector.count == 768)
            #expect(abs(vector.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-3)
        }
    }

    @Test func packingDoesNotChangeVectors() async throws {
        let service = try await ModelGate.service()
        let texts = (0..<12).map { "packed query number \($0) about topic \($0 * 7)" }
        let packed = try await service.embedTexts(texts)
        for (index, text) in texts.enumerated() {
            let alone = try await service.embedTexts([text])
            #expect(Vectors.cosine(alone.vectors[0], packed.vectors[index]) > 0.9999)
        }
    }

    @Test func longTextsUseTheLocalBand() async throws {
        let service = try await ModelGate.service()
        let long = String(repeating: "The Neural Engine runs fixed-shape graphs. ", count: 150)
        let result = try await service.embedTexts([long])
        #expect(result.tokenCounts[0] > 1024)
        #expect(abs(result.vectors[0].reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-3)
    }

    @Test(.enabled(if: ModelGate.directory.map { ModelLayout.model(ModelLayout.visionModel, in: $0) != nil } ?? false))
    func embedsImages() async throws {
        let service = try await ModelGate.service()
        let result = try await service.embedImages([try Data(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/images/up_320x240.png"))])
        #expect(result.tokenCounts == [266 + 4])
    }
}
