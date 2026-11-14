// Builds an EmbeddingService from the configuration: the Core ML packages in the model directory, or
// the fake runtime (--runtime fake) for testing integrations without a model.
//
// Loading = compile the packages if needed (once, normally at install) -> load the functions the
// configured --max-tokens needs -> run the package's self-test on each of them -> serve only those that
// pass. The self-test also pages every function in, so it doubles as the warm-up.
import Foundation

public struct LoadReport: Sendable {
    public var functions: [String] = []
    public var selfTest: SelfTestReport?
    public var seconds = 0.0
}

public enum ModelLoader {
    public static func makeService(_ config: Config, log: @Sendable (String) -> Void = { _ in }) async throws(ServiceError)
        -> EmbeddingService
    {
        try await load(config, log: log).service
    }

    public static func load(_ config: Config, log: @Sendable (String) -> Void = { _ in }) async throws(ServiceError)
        -> (service: EmbeddingService, report: LoadReport)
    {
        let started = Date()
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
            let service = EmbeddingService(tokenizer: tokenizer, runtime: FakeRuntime.make(maxTokens: min(config.maxTokens, 2048)),
                                           limits: config.limits, modelName: config.modelName)
            return (service, LoadReport())
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
        if let manifest = try? Manifest.load(from: directory) { try manifest.checkContract() }
        do {
            let tokenizer = try GemmaTokenizer(contentsOf: tokenizerURL)
            let table = try TokenTable(contentsOf: directory.appendingPathComponent(ModelLayout.tokenTable))
            let compiledText = try await compiledModelURL(textModel)
            var buckets = try await CoreMLBucketRunner.load(from: compiledText, maxTokens: config.maxTokens)
            var long: CoreMLLongFunctions?
            let longLengths = CoreMLLongFunctions.lengths(forMaxTokens: config.maxTokens)
            if !longLengths.isEmpty {
                log("loading the long-text functions for up to \(longLengths.last!) tokens")
                long = try await CoreMLLongFunctions.load(from: compiledText, lengths: longLengths)
            }
            var vision: CoreMLVisionEncoder?
            if let visionModel = ModelLayout.model(ModelLayout.visionModel, in: directory) {
                vision = try await CoreMLVisionEncoder.load(
                    model: visionModel, positionTable: directory.appendingPathComponent(ModelLayout.positionTable))
            } else {
                log("no vision model: image inputs are rejected")
            }

            var report = LoadReport()
            if let suite = try SelfTestSuite.load(from: directory) {
                let testStarted = Date()
                let results = await suite.run(buckets: buckets, long: long, vision: vision,
                                              visionBudgets: vision?.budgets ?? [], rows: table)
                let selfTest = SelfTestReport(osBuild: operatingSystemBuild, ranAt: Date(),
                                              seconds: Date().timeIntervalSince(testStarted), results: results)
                let failed = Set(selfTest.failed.map(\.function))
                for result in selfTest.failed {
                    log("self-test failed, not serving \(result.function): cos \(result.cosine)"
                        + (result.relativeError.map { ", rel err \($0)" } ?? "") + (result.error.map { ", \($0)" } ?? ""))
                }
                log(String(format: "self-test: %@ in %.1fs", selfTest.summary, selfTest.seconds))
                try? selfTest.save(to: directory)
                report.selfTest = selfTest
                if !failed.isEmpty {
                    buckets = buckets.excluding(failed)
                    long = long?.excluding(failed)
                    vision = vision?.excluding(failed)
                }
                guard !buckets.lengths.isEmpty else {
                    throw ServiceError(.unavailable, "selftest_failed", "every text function failed the self-test on this Mac",
                                       hint: "see `aneembed doctor`; please report the macOS build")
                }
            } else {
                log("the package has no self-test data: functions are not checked")
                report.selfTest = SelfTestReport(osBuild: operatingSystemBuild, ranAt: Date(), seconds: 0, results: [],
                                                 available: false)
            }
            if let long, long.lengths.isEmpty {
                log("texts are limited to \(buckets.lengths.last!) tokens: the long-text functions failed the self-test")
            }
            let longRunner = long.flatMap { $0.lengths.isEmpty ? nil : LongTextEngine(functions: $0, rows: table) }
            let visionEncoder = vision.flatMap { $0.budgets.isEmpty ? nil : $0 }
            report.functions = buckets.functionNames
                + CoreMLLongFunctions.functionNames(lengths: long?.lengths ?? []) + (visionEncoder?.functionNames ?? [])
            let runtime = BucketTextRuntime(runner: buckets, rows: table, long: longRunner, vision: visionEncoder,
                                            maxIdleBuffers: max(2, config.limits.queryConcurrency), backend: "coreml-ane",
                                            functions: report.functions, warmed: report.selfTest?.available ?? false)
            report.seconds = Date().timeIntervalSince(started)
            let service = EmbeddingService(tokenizer: tokenizer, runtime: runtime, limits: config.limits, modelName: config.modelName)
            return (service, report)
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "model_load_failed", "\(error)")
        }
    }
}
