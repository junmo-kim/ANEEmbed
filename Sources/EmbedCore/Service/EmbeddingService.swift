// The one place requests are validated and scheduled; CLI, HTTP and MCP all call this.
//
// Scheduling: three lanes with separate concurrency limits, so a burst of long documents or images
// cannot make short queries wait behind them (the prototype's lesson from a shared CPU worker queue).
//   query: texts of at most 512 tokens   image: decode + vision + text   long: texts over 512 tokens
import Foundation

public struct ServiceLimits: Sendable, Equatable {
    /// Longest text accepted (tokens including <bos>/<eos>); capped by the runtime.
    public var maxTokens = 2048
    public var maxInputs = 2048
    public var maxImages = 8
    public var maxImageBytes = 8 << 20
    public var queryConcurrency = 8
    public var imageConcurrency = 2
    public var longConcurrency = 2
    /// Texts up to this many tokens use the query lane.
    public var shortTokens = 512

    public init() {}
}

public struct EmbeddingResult: Sendable, Equatable {
    public var vectors: [[Float]]
    /// Per input: tokens the model saw (texts: with <bos>/<eos>; images: soft tokens + 4).
    public var tokenCounts: [Int]
    public var totalTokens: Int { tokenCounts.reduce(0, +) }

    public init(vectors: [[Float]], tokenCounts: [Int]) {
        self.vectors = vectors
        self.tokenCounts = tokenCounts
    }
}

public final class EmbeddingService: Sendable {
    public let tokenizer: any TextTokenizer
    public let runtime: any EmbeddingRuntime
    public let limits: ServiceLimits
    public let modelName: String
    private let queryLane: AsyncSemaphore
    private let imageLane: AsyncSemaphore
    private let longLane: AsyncSemaphore

    public init(tokenizer: any TextTokenizer, runtime: any EmbeddingRuntime, limits: ServiceLimits = ServiceLimits(),
                modelName: String = "embeddinggemma-2")
    {
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.limits = limits
        self.modelName = modelName
        queryLane = AsyncSemaphore(max(1, limits.queryConcurrency))
        imageLane = AsyncSemaphore(max(1, limits.imageConcurrency))
        longLane = AsyncSemaphore(max(1, limits.longConcurrency))
    }

    public var maxTokens: Int { min(limits.maxTokens, runtime.info.maxTokens) }
    public var maxImageTokens: Int { runtime.info.maxImageTokens }

    public func tokenize(_ text: String, inputType: InputType? = nil) -> [Int32] {
        tokenizer.encode(inputType.map { $0.apply(text) } ?? text)
    }

    // MARK: Texts

    public func embedTexts(_ texts: [String], inputType: InputType? = nil) async throws(ServiceError) -> EmbeddingResult {
        guard !texts.isEmpty else { throw .badRequest("empty_input", "input must not be empty", param: "input") }
        guard texts.count <= limits.maxInputs else {
            throw .badRequest("too_many_inputs", "at most \(limits.maxInputs) inputs per request", param: "input")
        }
        let tokenized = texts.map { tokenize($0, inputType: inputType) }
        let limit = maxTokens
        if let over = tokenized.firstIndex(where: { $0.count > limit }) {
            throw ServiceError(
                .invalidInput, "context_length_exceeded",
                "input[\(over)] has \(tokenized[over].count) tokens; this model's maximum context length is \(limit) tokens",
                param: "input",
                hint: limit < runtime.info.maxTokens ? "raise --max-tokens (up to \(runtime.info.maxTokens))" : nil)
        }
        let short = tokenized.indices.filter { tokenized[$0].count <= limits.shortTokens }
        let long = tokenized.indices.filter { tokenized[$0].count > limits.shortTokens }
        var vectors = [[Float]](repeating: [], count: texts.count)
        do {
            try await withThrowingTaskGroup(of: [(Int, [Float])].self) { group in
                let runtime = self.runtime
                if !short.isEmpty {
                    let batch = short.map { tokenized[$0] }
                    group.addTask { [queryLane] in
                        let result = try await queryLane.withPermit { try await runtime.embed(tokenized: batch) }
                        return Array(zip(short, result))
                    }
                }
                for index in long {
                    let ids = tokenized[index]
                    group.addTask { [longLane] in
                        let result = try await longLane.withPermit { try await runtime.embed(tokenized: [ids]) }
                        return [(index, result[0])]
                    }
                }
                for try await pairs in group {
                    for (index, vector) in pairs { vectors[index] = vector }
                }
            }
        } catch {
            throw Self.map(error)
        }
        return EmbeddingResult(vectors: vectors, tokenCounts: tokenized.map(\.count))
    }

    // MARK: Images

    /// Images as encoded bytes (PNG, JPEG, anything ImageIO reads), at most `limits.maxImages`.
    public func embedImages(_ images: [Data]) async throws(ServiceError) -> EmbeddingResult {
        guard !images.isEmpty else { throw .badRequest("empty_input", "input must not be empty", param: "input") }
        guard images.count <= limits.maxImages else {
            throw .badRequest("too_many_images", "at most \(limits.maxImages) image items per request", param: "input")
        }
        guard maxImageTokens > 0 else {
            throw ServiceError(.invalidInput, "images_unsupported",
                               "image inputs are not supported: the model has no vision encoder", param: "input")
        }
        if let big = images.firstIndex(where: { $0.count > limits.maxImageBytes }) {
            throw ServiceError(.payloadTooLarge, "image_too_large",
                               "input[\(big)] is \(images[big].count) bytes; the limit is \(limits.maxImageBytes)",
                               param: "input")
        }
        // Decode and resize everything first, so a bad image fails the request before any model call.
        var prepared = [PreparedImage?](repeating: nil, count: images.count)
        var failure: ServiceError?
        await withTaskGroup(of: (Int, Result<PreparedImage, Error>).self) { group in
            for (index, data) in images.enumerated() {
                group.addTask { [imageLane] in
                    await imageLane.wait()
                    let result = Result { try ImagePrep.prepare(data) }
                    await imageLane.signal()
                    return (index, result)
                }
            }
            for await (index, result) in group {
                switch result {
                case .success(let image): prepared[index] = image
                case .failure(let error):
                    if failure == nil {
                        failure = ServiceError(.invalidInput, "image_decode_failed",
                                               "input[\(index)] cannot be decoded as an image (\(error))", param: "input")
                    }
                }
            }
        }
        if let failure { throw failure }
        let canvases = prepared.map { $0! }
        do {
            let runtime = self.runtime
            let vectors = try await imageLane.withPermit { try await runtime.embed(images: canvases) }
            // <bos> <|image> soft tokens <image|> <eos>, as llama-server counts them
            return EmbeddingResult(vectors: vectors, tokenCounts: canvases.map { $0.tokens + 4 })
        } catch {
            throw Self.map(error)
        }
    }

    static func map(_ error: Error) -> ServiceError {
        switch error {
        case let error as ServiceError: return error
        case RuntimeError.tooLong(let tokens, let max):
            return ServiceError(.invalidInput, "context_length_exceeded",
                                "\(tokens) tokens; this model's maximum context length is \(max) tokens", param: "input")
        case RuntimeError.unsupported(let message):
            return ServiceError(.invalidInput, "unsupported_input", message, param: "input")
        default:
            return ServiceError(.internal, "embedding_failed", "embedding failed: \(error)")
        }
    }
}
