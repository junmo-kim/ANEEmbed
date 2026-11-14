// Builds an EmbeddingService from the configuration: the Core ML bundle in the model directory, or
// the fake runtime (--runtime fake) for testing integrations without a model.
import Foundation

public enum ModelLoader {
    public static func makeService(_ config: Config, log: @Sendable (String) -> Void = { _ in }) async throws(ServiceError)
        -> EmbeddingService
    {
        let directory = config.modelDirectory
        let tokenizerURL = directory.appendingPathComponent(ModelLayout.tokenizer)
        if config.useFakeRuntime {
            let tokenizer: any TextTokenizer
            if FileManager.default.fileExists(atPath: tokenizerURL.path), let real = try? GemmaTokenizer(contentsOf: tokenizerURL) {
                tokenizer = real
            } else {
                tokenizer = ByteTokenizer()
            }
            log("fake runtime (no model): vectors are deterministic stand-ins, not embeddings")
            return EmbeddingService(tokenizer: tokenizer, runtime: FakeRuntime.make(maxTokens: min(config.maxTokens, 2048)),
                                    limits: config.limits, modelName: config.modelName)
        }

        guard NeuralEngine.device.available else {
            throw ServiceError(.unavailable, "no_neural_engine", "Core ML reports no Neural Engine on this Mac",
                               hint: "ANEEmbed runs only on Apple silicon with a Neural Engine")
        }
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw ServiceError(.unavailable, "model_missing", "no model directory at \(directory.path)",
                               hint: "run `aneembed models install`, or pass --model-dir")
        }
        guard let textModel = ModelLayout.model(ModelLayout.textModel, in: directory) else {
            throw ServiceError(.unavailable, "model_missing", "no \(ModelLayout.textModel) model in \(directory.path)")
        }
        do {
            let tokenizer = try GemmaTokenizer(contentsOf: tokenizerURL)
            let table = try TokenTable(contentsOf: directory.appendingPathComponent(ModelLayout.tokenTable))
            let runner = try await CoreMLBucketRunner.load(from: textModel, maxTokens: config.maxTokens)
            var vision: (any VisionEncoder)?
            if let visionModel = ModelLayout.model(ModelLayout.visionModel, in: directory) {
                vision = try await CoreMLVisionEncoder.load(
                    model: visionModel, positionTable: directory.appendingPathComponent(ModelLayout.positionTable))
            } else {
                log("no vision model: image inputs are rejected")
            }
            // TODO(model): LongTextRunner for 2049...8192 tokens once the per-layer contract is delivered.
            if config.maxTokens > runner.lengths.last! {
                log("texts are limited to \(runner.lengths.last!) tokens: long-text functions are not available yet")
            }
            let runtime = BucketTextRuntime(runner: runner, rows: table, vision: vision,
                                            maxIdleBuffers: max(2, config.limits.queryConcurrency), backend: "coreml-ane")
            return EmbeddingService(tokenizer: tokenizer, runtime: runtime, limits: config.limits, modelName: config.modelName)
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "model_load_failed", "\(error)")
        }
    }
}
