// The long-path functions of the text package on the Neural Engine: long_swa_l{i} (20 sliding layers,
// shared by both lengths), long{S}_glb_l{5,11,17} and long{S}_head for S in 4096 / 8192.
@preconcurrency import CoreML
import Foundation

/// An fp16 MLMultiArray over memory we own (no copy on our side).
func halfArray(_ pointer: UnsafeMutablePointer<Float16>, _ shape: [Int]) throws -> MLMultiArray {
    var strides = [Int](repeating: 1, count: shape.count)
    for index in stride(from: shape.count - 2, through: 0, by: -1) { strides[index] = strides[index + 1] * shape[index + 1] }
    return try MLMultiArray(dataPointer: pointer, shape: shape.map(NSNumber.init(value:)), dataType: .float16,
                            strides: strides.map(NSNumber.init(value:)), deallocator: nil)
}

/// Copies an fp16 [1, rows, width] (or [rows, width]) output into contiguous rows, honouring its strides.
func copyHalfRows(_ array: MLMultiArray, rows: Int, width: Int, into target: UnsafeMutablePointer<Float16>) throws {
    let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
    guard array.dataType == .float16, shape.count >= 2, shape[shape.count - 1] == width, shape[shape.count - 2] >= rows
    else { throw RuntimeError.model("unexpected output \(shape) \(array.dataType.rawValue), want [\(rows), \(width)] fp16") }
    let rowStride = strides[strides.count - 2], columnStride = strides[strides.count - 1]
    array.withUnsafeBufferPointer(ofType: Float16.self) { source in
        let base = source.baseAddress!
        if columnStride == 1 && rowStride == width {
            target.update(from: base, count: rows * width)
        } else {
            for row in 0..<rows {
                if columnStride == 1 {
                    (target + row * width).update(from: base + row * rowStride, count: width)
                } else {
                    for column in 0..<width { target[row * width + column] = base[row * rowStride + column * columnStride] }
                }
            }
        }
    }
}

public final class CoreMLLongFunctions: LongFunctions, @unchecked Sendable {
    public let lengths: [Int]
    private let sliding: [Int: MLModel]
    private let global: [Int: [Int: MLModel]]
    private let heads: [Int: MLModel]

    init(lengths: [Int], sliding: [Int: MLModel], global: [Int: [Int: MLModel]], heads: [Int: MLModel]) {
        self.lengths = lengths
        self.sliding = sliding
        self.global = global
        self.heads = heads
    }

    /// The lengths a `maxTokens` limit needs: none up to 2048, 4096 up to 4096, else both.
    public static func lengths(forMaxTokens maxTokens: Int) -> [Int] {
        guard maxTokens > BucketContract.lengths.last!, let top = LongContract.length(forTokens: maxTokens) else { return [] }
        return LongContract.lengths.filter { $0 <= top }
    }

    /// Every function name `lengths` loads.
    public static func functionNames(lengths: [Int]) -> [String] {
        guard !lengths.isEmpty else { return [] }
        return LongContract.slidingLayers.map { LongContract.slidingName(layer: $0) }
            + lengths.flatMap { length in
                LongContract.globalLayers.map { LongContract.globalName(length: length, layer: $0) }
                    + [LongContract.headName(length: length)]
            }
    }

    public static func load(from compiled: URL, lengths: [Int], computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLLongFunctions
    {
        var sliding: [Int: MLModel] = [:]
        for layer in LongContract.slidingLayers {
            sliding[layer] = try await loadFunction(compiled, name: LongContract.slidingName(layer: layer), computeUnits: computeUnits)
        }
        var global: [Int: [Int: MLModel]] = [:]
        var heads: [Int: MLModel] = [:]
        for length in lengths {
            for layer in LongContract.globalLayers {
                global[length, default: [:]][layer] = try await loadFunction(
                    compiled, name: LongContract.globalName(length: length, layer: layer), computeUnits: computeUnits)
            }
            heads[length] = try await loadFunction(compiled, name: LongContract.headName(length: length), computeUnits: computeUnits)
        }
        return CoreMLLongFunctions(lengths: lengths, sliding: sliding, global: global, heads: heads)
    }

    /// The same functions without the lengths that `failed` names break (a failed sliding layer breaks both).
    public func excluding(_ failed: Set<String>) -> CoreMLLongFunctions {
        let slidingFailed = LongContract.slidingLayers.contains { failed.contains(LongContract.slidingName(layer: $0)) }
        let kept = slidingFailed ? [] : lengths.filter { length in
            !LongContract.functionNames(length: length).contains(where: failed.contains)
        }
        return CoreMLLongFunctions(lengths: kept, sliding: kept.isEmpty ? [:] : sliding,
                                   global: global.filter { kept.contains($0.key) }, heads: heads.filter { kept.contains($0.key) })
    }

    public func sliding(layer: Int, window: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                        bias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    {
        guard let model = sliding[layer] else { throw RuntimeError.model("\(LongContract.slidingName(layer: layer)) is not loaded") }
        let hidden = LongContract.hidden, chunk = LongContract.chunk, rows = LongContract.windowRows
        let input = try MLDictionaryFeatureProvider(dictionary: [
            LongContract.hiddenWindow: MLFeatureValue(multiArray: try halfArray(window, [1, rows, hidden])),
            LongContract.inputsEmbeds: MLFeatureValue(multiArray: try halfArray(embeds, [1, chunk, hidden])),
            LongContract.bias: MLFeatureValue(multiArray: try halfArray(bias, [1, 1, chunk, rows])),
        ])
        let result = try await model.prediction(from: input)
        guard let out = result.featureValue(for: LongContract.hiddenOut)?.multiArrayValue else {
            throw RuntimeError.model("\(LongContract.slidingName(layer: layer)) returned no \(LongContract.hiddenOut)")
        }
        try copyHalfRows(out, rows: chunk, width: hidden, into: output)
    }

    public func global(length: Int, layer: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                       keyBias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    {
        guard let model = global[length]?[layer] else {
            throw RuntimeError.model("\(LongContract.globalName(length: length, layer: layer)) is not loaded")
        }
        let width = LongContract.hidden
        let input = try MLDictionaryFeatureProvider(dictionary: [
            LongContract.hiddenInput: MLFeatureValue(multiArray: try halfArray(hidden, [1, length, width])),
            LongContract.inputsEmbeds: MLFeatureValue(multiArray: try halfArray(embeds, [1, length, width])),
            LongContract.keyBias: MLFeatureValue(multiArray: try halfArray(keyBias, [1, 1, 1, length])),
        ])
        let result = try await model.prediction(from: input)
        guard let out = result.featureValue(for: LongContract.hiddenOut)?.multiArrayValue else {
            throw RuntimeError.model("\(LongContract.globalName(length: length, layer: layer)) returned no \(LongContract.hiddenOut)")
        }
        try copyHalfRows(out, rows: length, width: width, into: output)
    }

    public func head(length: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                     keyBias: UnsafeMutablePointer<Float16>, pool: UnsafeMutablePointer<Float16>) async throws -> [Float]
    {
        guard let model = heads[length] else { throw RuntimeError.model("\(LongContract.headName(length: length)) is not loaded") }
        let width = LongContract.hidden
        let input = try MLDictionaryFeatureProvider(dictionary: [
            LongContract.hiddenInput: MLFeatureValue(multiArray: try halfArray(hidden, [1, length, width])),
            LongContract.inputsEmbeds: MLFeatureValue(multiArray: try halfArray(embeds, [1, length, width])),
            LongContract.keyBias: MLFeatureValue(multiArray: try halfArray(keyBias, [1, 1, 1, length])),
            LongContract.pool: MLFeatureValue(multiArray: try halfArray(pool, [1, length])),
        ])
        let result = try await model.prediction(from: input)
        guard let out = result.featureValue(for: LongContract.embedding)?.multiArrayValue else {
            throw RuntimeError.model("\(LongContract.headName(length: length)) returned no \(LongContract.embedding)")
        }
        return try rows(of: out, count: 1, width: BucketContract.dimension)[0]
    }
}
