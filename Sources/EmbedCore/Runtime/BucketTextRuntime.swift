// Texts and images through the uniform bucket functions: plan -> fill reusable buffers -> run -> pick
// each text's pooled row. The runner is the only model-specific part.
import Foundation

public final class BucketTextRuntime: EmbeddingRuntime {
    public let planner: PackingPlanner
    private let runner: any BucketFunctionRunner
    private let rows: any TokenRowSource
    private let long: (any LongTextRunner)?
    private let vision: (any VisionEncoder)?
    private let buffers: BucketBufferPool
    public let backend: String

    public init(runner: any BucketFunctionRunner, rows: any TokenRowSource, long: (any LongTextRunner)? = nil,
                vision: (any VisionEncoder)? = nil, costs: [Int: Double] = PackingPlanner.defaultCosts,
                maxIdleBuffers: Int = 4, backend: String)
    {
        planner = PackingPlanner(lengths: runner.lengths, costs: costs)
        self.runner = runner
        self.rows = rows
        self.long = long
        self.vision = vision
        buffers = BucketBufferPool(maxIdlePerLength: maxIdleBuffers)
        self.backend = backend
    }

    public var info: RuntimeInfo {
        RuntimeInfo(maxTokens: max(planner.maxLength, long?.maxTokens ?? 0), maxImageTokens: vision?.maxTokens ?? 0,
                    bucketLengths: planner.lengths, backend: backend)
    }

    public func warm() async throws {
        try await runner.warm()
        try await long?.warm()
        try await vision?.warm()
    }

    public func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] {
        var vectors = [[Float]](repeating: [], count: texts.count)
        let bucketed = texts.indices.filter { texts[$0].count <= planner.maxLength }
        let longer = texts.indices.filter { texts[$0].count > planner.maxLength }
        if !longer.isEmpty {
            guard let long else {
                throw RuntimeError.tooLong(tokens: texts[longer[0]].count, max: planner.maxLength)
            }
            for index in longer {
                guard texts[index].count <= long.maxTokens else {
                    throw RuntimeError.tooLong(tokens: texts[index].count, max: long.maxTokens)
                }
                vectors[index] = try await long.embed(ids: texts[index])
            }
        }
        let sequences = bucketed.map { ModelSequence(ids: texts[$0]) }
        let results = try await embed(sequences: sequences)
        for (position, index) in bucketed.enumerated() { vectors[index] = results[position] }
        return vectors
    }

    public func embed(images: [PreparedImage]) async throws -> [[Float]] {
        guard let vision else { throw RuntimeError.unsupported("this model has no vision encoder") }
        var sequences: [ModelSequence] = []
        for image in images {
            guard image.tokens <= vision.maxTokens else {
                throw RuntimeError.unsupported("\(image.tokens) image tokens exceed \(vision.maxTokens)")
            }
            sequences.append(.image(rows: try await vision.encode(image)))
        }
        return try await embed(sequences: sequences)
    }

    /// Plans the sequences into calls and runs them concurrently (the ANE serializes the work itself;
    /// overlapping lets one call's host-side filling hide behind another's execution).
    func embed(sequences: [ModelSequence]) async throws -> [[Float]] {
        guard !sequences.isEmpty else { return [] }
        let counts = sequences.map(\.count)
        if let over = counts.first(where: { $0 > planner.maxLength }) {
            throw RuntimeError.tooLong(tokens: over, max: planner.maxLength)
        }
        let calls = planner.plan(counts)
        var vectors = [[Float]](repeating: [], count: sequences.count)
        try await withThrowingTaskGroup(of: [(Int, [Float])].self) { group in
            for call in calls {
                group.addTask { [buffers, runner, rows] in
                    let buffer = buffers.checkout(length: call.length)
                    defer { buffers.checkin(buffer) }
                    buffer.prepare(call, sequences: sequences, rows: rows)
                    let output = try await runner.run(buffer)
                    guard output.count >= call.segments.count else {
                        throw RuntimeError.model("embed_\(call.length) returned \(output.count) rows")
                    }
                    return call.segments.enumerated().map { slot, segment in (segment.item, output[slot]) }
                }
            }
            for try await pairs in group {
                for (item, vector) in pairs { vectors[item] = vector }
            }
        }
        return vectors
    }
}
