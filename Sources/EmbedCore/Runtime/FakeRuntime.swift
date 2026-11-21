// A model-free runtime for tests and for running the CLI/HTTP/MCP surface without a model package
// (ANEEMBED_RUNTIME=fake). It goes through the real planner and buffers, and its "model" reads the
// inputs the way the contract says the real one does: each output row is the pool-weighted mean of the
// input rows. So a text gets the same vector alone or packed with others, which tests can rely on.
import Foundation

public enum FakeRuntime {
    /// A runtime over the real planner/buffers with deterministic fake token rows and functions.
    public static func make(maxTokens: Int = 2048, images: Bool = true, callDelay: Duration = .zero)
        -> BucketTextRuntime
    {
        // The same functions the Core ML runtime loads for this limit.
        let lengths = CoreMLBucketRunner.lengths(forMaxTokens: maxTokens)
        return BucketTextRuntime(
            runner: PoolingFakeRunner(lengths: lengths, delay: callDelay),
            rows: HashedTokenRows(), vision: images ? FakeVisionEncoder() : nil, backend: "fake")
    }
}

/// Pseudo-random but fixed token rows (no 256 MB table needed).
public struct HashedTokenRows: TokenRowSource {
    public init() {}

    public func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
        for (position, id) in ids.enumerated() {
            let row = destination + position * BucketContract.hidden
            for column in 0..<BucketContract.hidden {
                row[column] = Float16(Self.value(id: id, column: column))
            }
        }
    }

    /// Uniform in [-1, 1) from SplitMix64(id, column).
    static func value(id: Int32, column: Int) -> Float {
        var z = UInt64(bitPattern: Int64(id)) &* 0x9E37_79B9_7F4A_7C15 &+ UInt64(column) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 23) - 1
    }
}

/// Each slot's output = normalize(pool row · inputs_embeds), widened from 512 to 768 values.
public struct PoolingFakeRunner: BucketFunctionRunner {
    public let lengths: [Int]
    public let delay: Duration

    public init(lengths: [Int] = BucketContract.lengths, delay: Duration = .zero) {
        self.lengths = lengths
        self.delay = delay
    }

    public func warm() async throws {}

    public func run(_ buffers: BucketBuffers) async throws -> [[Float]] {
        if delay > .zero { try await Task.sleep(for: delay) }
        let length = buffers.length, hidden = BucketContract.hidden
        var output: [[Float]] = []
        output.reserveCapacity(buffers.slots)
        for slot in 0..<buffers.slots {
            var pooled = [Float](repeating: 0, count: hidden)
            for position in 0..<length {
                let weight = Float(buffers.pool[slot * length + position])
                guard weight != 0 else { continue }
                let row = buffers.embedsRow(position)
                for column in 0..<hidden { pooled[column] += weight * Float(row[column]) }
            }
            output.append(Vectors.l2Normalized(pooled + pooled.prefix(BucketContract.dimension - hidden)))
        }
        return output
    }
}

/// Soft tokens from the mean colour of each 48 x 48 cell, so the same picture at two resolutions gives
/// nearly the same vector, and different pictures differ.
public struct FakeVisionEncoder: VisionEncoder {
    public let maxTokens = ImagePrep.maxSoftTokens

    public init() {}

    public func warm() async throws {}

    public func encode(_ image: PreparedImage) async throws -> [Float16] {
        let columns = image.width / 48, rows = image.height / 48, hidden = BucketContract.hidden
        var output = [Float16](repeating: 0, count: columns * rows * hidden)
        for cellY in 0..<rows {
            for cellX in 0..<columns {
                var mean: [Float] = [0, 0, 0]
                for y in (cellY * 48)..<(cellY * 48 + 48) {
                    for x in (cellX * 48)..<(cellX * 48 + 48) {
                        let pixel = (y * image.width + x) * 3
                        for c in 0..<3 { mean[c] += Float(image.rgb[pixel + c]) }
                    }
                }
                let token = cellY * columns + cellX
                for column in 0..<hidden {
                    var value: Float = 0
                    for c in 0..<3 {
                        value += mean[c] / (48 * 48 * 255) * HashedTokenRows.value(id: Int32(c), column: column)
                    }
                    output[token * hidden + column] = Float16(value)
                }
            }
        }
        return output
    }
}

/// Stand-in tokenizer: <bos>, one id per UTF-8 byte, <eos>. Used only when no tokenizer.json exists.
public struct ByteTokenizer: TextTokenizer {
    public init() {}

    public func encode(_ text: String) -> [Int32] {
        [SpecialTokens.bos] + text.utf8.map { Int32($0) + 3 } + [SpecialTokens.eos]
    }

    public var maxBytesPerToken: Int? { 1 }
}
