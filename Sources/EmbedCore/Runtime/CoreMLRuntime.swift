// Core ML backends: the embed_S text functions and the vision encoder, on the Neural Engine
// (MLComputeUnits.cpuAndNeuralEngine; the GPU is never used, so a local LLM keeps it to itself).
//
// Inputs are MLMultiArrays over the BucketBuffers memory (no copy on our side).
@preconcurrency import CoreML
import Foundation

public enum NeuralEngine {
    /// Whether Core ML reports a Neural Engine on this Mac, and its core count.
    public static var device: (available: Bool, cores: Int) {
        for device in MLComputeDevice.allComputeDevices {
            if case .neuralEngine(let engine) = device { return (true, engine.totalCoreCount) }
        }
        return (false, 0)
    }
}

/// Loads an .mlmodelc, compiling an .mlpackage next to it first if needed (once, at install/warm time).
func compiledModelURL(_ url: URL) async throws -> URL {
    guard url.pathExtension == "mlpackage" else { return url }
    let target = url.deletingPathExtension().appendingPathExtension("mlmodelc")
    if FileManager.default.fileExists(atPath: target.path) { return target }
    let compiled = try await MLModel.compileModel(at: url)
    try? FileManager.default.removeItem(at: target)
    try FileManager.default.moveItem(at: compiled, to: target)
    return target
}

func loadFunction(_ url: URL, name: String, computeUnits: MLComputeUnits) async throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    configuration.functionName = name
    do {
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    } catch {
        throw RuntimeError.model("cannot load \(name) from \(url.lastPathComponent): \(error.localizedDescription)")
    }
}

/// Copies rows of a 2-D (or batched) fp16/fp32 MLMultiArray into Float arrays.
func rows(of array: MLMultiArray, count: Int, width: Int) throws -> [[Float]] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    guard shape.count >= 2, shape[shape.count - 1] == width, shape[shape.count - 2] >= count else {
        throw RuntimeError.model("unexpected output shape \(shape), want [\(count), \(width)]")
    }
    let rowStride = strides[strides.count - 2], columnStride = strides[strides.count - 1]
    var output: [[Float]] = []
    output.reserveCapacity(count)
    switch array.dataType {
    case .float32:
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for row in 0..<count { output.append((0..<width).map { pointer[row * rowStride + $0 * columnStride] }) }
    case .float16:
        let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for row in 0..<count { output.append((0..<width).map { Float(pointer[row * rowStride + $0 * columnStride]) }) }
    default:
        throw RuntimeError.model("unexpected output type \(array.dataType.rawValue)")
    }
    return output
}

public final class CoreMLBucketRunner: BucketFunctionRunner, @unchecked Sendable {
    public let lengths: [Int]
    private let models: [Int: MLModel]

    private init(models: [Int: MLModel]) {
        self.models = models
        lengths = models.keys.sorted()
    }

    /// The same runner without the functions named in `failed`.
    public func excluding(_ failed: Set<String>) -> CoreMLBucketRunner {
        CoreMLBucketRunner(models: models.filter { !failed.contains(BucketContract.functionName(forLength: $0.key)) })
    }

    public var functionNames: [String] { lengths.map(BucketContract.functionName(forLength:)) }

    /// Loads embed_S for every bucket needed up to `maxTokens` (at most the largest bucket).
    public static func load(from url: URL, maxTokens: Int, computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLBucketRunner
    {
        let compiled = try await compiledModelURL(url)
        let cap = BucketContract.lengths.first { $0 >= min(maxTokens, BucketContract.lengths.last!) }!
        var models: [Int: MLModel] = [:]
        for length in BucketContract.lengths where length <= cap {
            let model = try await loadFunction(compiled, name: BucketContract.functionName(forLength: length),
                                               computeUnits: computeUnits)
            try check(model, length: length)
            models[length] = model
        }
        return CoreMLBucketRunner(models: models)
    }

    /// Fails early with a readable message when the package does not follow the bucket contract.
    static func check(_ model: MLModel, length: Int) throws {
        let inputs = model.modelDescription.inputDescriptionsByName
        let slots = BucketContract.slots(forLength: length)
        let expected: [(String, [Int])] = [
            (BucketContract.inputsEmbeds, [1, length, BucketContract.hidden]),
            (BucketContract.globalBias, [1, 1, length, length]), (BucketContract.localBias, [1, 1, length, length]),
            (BucketContract.pool, [slots, length]),
        ]
        for (name, shape) in expected {
            guard let constraint = inputs[name]?.multiArrayConstraint else {
                throw RuntimeError.model("embed_\(length) has no \(name) input (bucket contract)")
            }
            guard constraint.shape.map(\.intValue) == shape else {
                throw RuntimeError.model("embed_\(length).\(name) is \(constraint.shape), want \(shape)")
            }
        }
        guard model.modelDescription.outputDescriptionsByName[BucketContract.embedding] != nil else {
            throw RuntimeError.model("embed_\(length) has no \(BucketContract.embedding) output")
        }
    }

    public func run(_ buffers: BucketBuffers) async throws -> [[Float]] {
        guard let model = models[buffers.length] else {
            throw RuntimeError.model("embed_\(buffers.length) is not loaded")
        }
        let length = buffers.length, hidden = BucketContract.hidden
        func array(_ pointer: UnsafeMutablePointer<Float16>, _ shape: [Int]) throws -> MLMultiArray {
            var strides = [Int](repeating: 1, count: shape.count)
            for index in stride(from: shape.count - 2, through: 0, by: -1) { strides[index] = strides[index + 1] * shape[index + 1] }
            return try MLMultiArray(dataPointer: pointer, shape: shape.map(NSNumber.init(value:)), dataType: .float16,
                                    strides: strides.map(NSNumber.init(value:)), deallocator: nil)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            BucketContract.inputsEmbeds: MLFeatureValue(multiArray: try array(buffers.inputsEmbeds, [1, length, hidden])),
            BucketContract.globalBias: MLFeatureValue(multiArray: try array(buffers.globalBias, [1, 1, length, length])),
            BucketContract.localBias: MLFeatureValue(multiArray: try array(buffers.localBias, [1, 1, length, length])),
            BucketContract.pool: MLFeatureValue(multiArray: try array(buffers.pool, [buffers.slots, length])),
        ])
        let output = try await model.prediction(from: input)
        guard let embedding = output.featureValue(for: BucketContract.embedding)?.multiArrayValue else {
            throw RuntimeError.model("embed_\(length) returned no \(BucketContract.embedding)")
        }
        return try rows(of: embedding, count: buffers.slots, width: BucketContract.dimension)
    }

    /// One call per function so the first request does not pay for paging each one in.
    public func warm() async throws {
        for length in lengths {
            let buffers = BucketBuffers(length: length)
            buffers.setLayout([PackedSegment(item: 0, offset: 0, length: 2)])
            _ = try await run(buffers)
        }
    }
}

/// vision_70 / vision_140 / vision_280 of the vision package (FluidInference/embeddinggemma-2-coreml):
/// 16 px patches scaled to [0, 1], 2-D position embeddings from position_embeddings.f16 (x row + y row),
/// and a 3 x 3 average pool of the patch grid into soft tokens.
public final class CoreMLVisionEncoder: VisionEncoder, @unchecked Sendable {
    public static let allBudgets = [70, 140, 280]
    static let patch = 16
    static let patchWidth = 768  // 16 * 16 * 3
    static let tableRows = 1024
    /// Loaded budgets (soft tokens per function), ascending.
    public let budgets: [Int]
    public var maxTokens: Int { budgets.last ?? 0 }
    private let models: [Int: MLModel]
    private let positionTable: Data

    private init(models: [Int: MLModel], positionTable: Data) {
        self.models = models
        self.positionTable = positionTable
        budgets = models.keys.sorted()
    }

    /// The same encoder without the functions named in `failed`.
    public func excluding(_ failed: Set<String>) -> CoreMLVisionEncoder {
        CoreMLVisionEncoder(models: models.filter { !failed.contains("vision_\($0.key)") }, positionTable: positionTable)
    }

    public var functionNames: [String] { budgets.map { "vision_\($0)" } }

    public static func load(model url: URL, positionTable: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLVisionEncoder
    {
        guard let table = try? Data(contentsOf: positionTable, options: .alwaysMapped),
            table.count == 2 * tableRows * patchWidth * 2
        else { throw RuntimeError.model("missing or malformed \(positionTable.lastPathComponent)") }
        let compiled = try await compiledModelURL(url)
        var models: [Int: MLModel] = [:]
        for budget in allBudgets {
            models[budget] = try await loadFunction(compiled, name: "vision_\(budget)", computeUnits: computeUnits)
        }
        return CoreMLVisionEncoder(models: models, positionTable: table)
    }

    public func warm() async throws {
        for (budget, (columns, rows)) in zip(Self.allBudgets, [(10, 7), (14, 10), (20, 14)]) where models[budget] != nil {
            let image = PreparedImage(rgb: [UInt8](repeating: 0, count: columns * rows * 48 * 48 * 3),
                                      width: columns * 48, height: rows * 48)
            precondition(image.tokens <= budget)
            _ = try await encode(image)
        }
    }

    public func encode(_ image: PreparedImage) async throws -> [Float16] {
        let tokens = image.tokens
        guard let budget = budgets.first(where: { $0 >= tokens }), let model = models[budget] else {
            throw RuntimeError.unsupported("\(tokens) image tokens do not fit")
        }
        let output = try await model.prediction(from: try inputs(image, budget: budget))
        guard let soft = output.featureValue(for: "image_tokens")?.multiArrayValue else {
            throw RuntimeError.model("vision_\(budget) returned no image_tokens")
        }
        return try rows(of: soft, count: tokens, width: BucketContract.hidden).flatMap { $0.map(Float16.init) }
    }

    private func inputs(_ image: PreparedImage, budget: Int) throws -> MLDictionaryFeatureProvider {
        let total = budget * 9
        let width = image.width, columns = width / Self.patch, rows = image.height / Self.patch
        precondition(columns * rows <= total && width % 48 == 0 && image.height % 48 == 0)
        let count = NSNumber(value: total)
        let patches = try MLMultiArray(shape: [1, count, NSNumber(value: Self.patchWidth)], dataType: .float16)
        let positions = try MLMultiArray(shape: [1, count, 2], dataType: .float16)
        let embeddings = try MLMultiArray(shape: [1, count, NSNumber(value: Self.patchWidth)], dataType: .float16)
        let valid = try MLMultiArray(shape: [1, count], dataType: .float16)
        let pool = try MLMultiArray(shape: [NSNumber(value: budget), count], dataType: .float16)
        let pp = patches.dataPointer.assumingMemoryBound(to: Float16.self)
        let po = positions.dataPointer.assumingMemoryBound(to: Float16.self)
        let pe = embeddings.dataPointer.assumingMemoryBound(to: Float16.self)
        let pv = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let pl = pool.dataPointer.assumingMemoryBound(to: Float16.self)
        pp.update(repeating: 0, count: total * Self.patchWidth)
        pe.update(repeating: 0, count: total * Self.patchWidth)
        po.update(repeating: -1, count: total * 2)
        pv.update(repeating: 0, count: total)
        pl.update(repeating: 0, count: budget * total)
        image.rgb.withUnsafeBufferPointer { rgb in
            positionTable.withUnsafeBytes { raw in
                let table = raw.bindMemory(to: Float16.self)
                for row in 0..<rows {
                    for column in 0..<columns {
                        let patch = row * columns + column, base = patch * Self.patchWidth
                        for y in 0..<16 {
                            for x in 0..<16 {
                                let pixel = ((row * 16 + y) * width + column * 16 + x) * 3
                                let offset = base + (y * 16 + x) * 3
                                for c in 0..<3 { pp[offset + c] = Float16(Float(rgb[pixel + c]) / 255) }
                            }
                        }
                        po[patch * 2] = Float16(column)
                        po[patch * 2 + 1] = Float16(row)
                        pv[patch] = 1
                        let xRow = column * Self.patchWidth, yRow = (Self.tableRows + row) * Self.patchWidth
                        for c in 0..<Self.patchWidth { pe[base + c] = Float16(Float(table[xRow + c]) + Float(table[yRow + c])) }
                        let group = (column / 3) + (columns / 3) * (row / 3)
                        pl[group * total + patch] = Float16(1.0 / 9.0)
                    }
                }
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: patches), "positions": MLFeatureValue(multiArray: positions),
            "position_embeddings": MLFeatureValue(multiArray: embeddings), "valid": MLFeatureValue(multiArray: valid),
            "pool": MLFeatureValue(multiArray: pool),
        ])
    }
}
