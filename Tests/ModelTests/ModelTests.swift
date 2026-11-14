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
