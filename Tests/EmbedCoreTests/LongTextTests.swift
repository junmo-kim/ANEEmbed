import Foundation
import Testing

@testable import EmbedCore

/// Token rows whose first two values identify the id; the rest are zero.
private struct TinyRows: TokenRowSource {
    func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
        for (position, id) in ids.enumerated() {
            let row = destination + position * LongContract.hidden
            row.update(repeating: 0, count: LongContract.hidden)
            row[0] = Float16(Float(id % 97) / 97)
            row[1] = Float16(Float(id % 13) / 13 - 0.5)
        }
    }
}

/// Stand-in layers that use the inputs the way the real ones must: a sliding layer averages the window
/// rows its bias opens, a global layer adds half the mean of the unmasked rows, the head sums pool * x.
/// Only the first two hidden values are computed, so the test stays fast.
private final class AveragingLayers: LongFunctions, @unchecked Sendable {
    let lengths: [Int]
    let compute: Bool
    let lock = NSLock()
    var calls: [String] = []

    init(lengths: [Int], compute: Bool = true) {
        self.lengths = lengths
        self.compute = compute
    }

    func record(_ name: String) { lock.withLock { calls.append(name) } }

    func sliding(layer: Int, window: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                 bias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    {
        record(LongContract.slidingName(layer: layer))
        guard compute else { return }
        let hidden = LongContract.hidden, columns = LongContract.windowRows, halo = LongContract.halo
        // Prefix sums over the window, then per row the open run inside the band (the bias test checks
        // that the band is the only open run besides the diagonal).
        var prefix = [(Double, Double)](repeating: (0, 0), count: columns + 1)
        for c in 0..<columns {
            prefix[c + 1] = (prefix[c].0 + Double(window[c * hidden]), prefix[c].1 + Double(window[c * hidden + 1]))
        }
        for r in 0..<LongContract.chunk {
            let row = bias + r * columns
            var lower = r, upper = r + 2 * halo  // inclusive band [r, r + 1024]
            while lower <= upper && row[lower] != 0 { lower += 1 }
            while upper >= lower && row[upper] != 0 { upper -= 1 }
            var sum = (0.0, 0.0), count = 0.0
            if lower <= upper {
                sum = (prefix[upper + 1].0 - prefix[lower].0, prefix[upper + 1].1 - prefix[lower].1)
                count = Double(upper - lower + 1)
            }
            let diagonal = r + halo
            if diagonal < lower || diagonal > upper {
                sum.0 += Double(window[diagonal * hidden])
                sum.1 += Double(window[diagonal * hidden + 1])
                count += 1
            }
            output[r * hidden] = Float16(sum.0 / count + Double(embeds[r * hidden]) * 0.01)
            output[r * hidden + 1] = Float16(sum.1 / count)
        }
    }

    func global(length: Int, layer: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                keyBias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    {
        record(LongContract.globalName(length: length, layer: layer))
        guard compute else { return }
        let width = LongContract.hidden
        var mean = (0.0, 0.0), count = 0.0
        for k in 0..<length where keyBias[k] == 0 {
            mean.0 += Double(hidden[k * width])
            mean.1 += Double(hidden[k * width + 1])
            count += 1
        }
        for q in 0..<length {
            output[q * width] = Float16(Double(hidden[q * width]) + 0.5 * mean.0 / count)
            output[q * width + 1] = Float16(Double(hidden[q * width + 1]) + 0.5 * mean.1 / count)
        }
    }

    func head(length: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
              keyBias: UnsafeMutablePointer<Float16>, pool: UnsafeMutablePointer<Float16>) async throws -> [Float]
    {
        record(LongContract.headName(length: length))
        let width = LongContract.hidden
        var output = [Float](repeating: 0, count: 768)
        for row in 0..<length {
            output[0] += Float(pool[row]) * Float(hidden[row * width])
            output[1] += Float(pool[row]) * Float(hidden[row * width + 1])
        }
        return output
    }
}

/// The same model computed directly on the whole sequence, with the contract's masks.
private func reference(ids: [Int32]) -> [Float] {
    let n = ids.count, length = LongContract.length(forTokens: n)!, width = LongContract.hidden
    var x0 = [Float16](repeating: 0, count: length * width)
    x0.withUnsafeMutableBufferPointer { TinyRows().gather(ids, into: $0.baseAddress!) }
    var x = x0
    let live = LongContract.liveChunks(tokens: n) * LongContract.chunk
    for layer in 0..<23 {
        var next = x
        if LongContract.globalLayers.contains(layer) {
            var mean = (0.0, 0.0)
            for k in 0..<n {
                mean.0 += Double(x[k * width])
                mean.1 += Double(x[k * width + 1])
            }
            for q in 0..<length {
                next[q * width] = Float16(Double(x[q * width]) + 0.5 * mean.0 / Double(n))
                next[q * width + 1] = Float16(Double(x[q * width + 1]) + 0.5 * mean.1 / Double(n))
            }
        } else {
            for q in 0..<live {
                var keys = Array(stride(from: max(0, q - 512), through: min(n - 1, q + 512), by: 1))
                if q >= n { keys.append(q) }
                var sum = (0.0, 0.0)
                for k in keys {
                    sum.0 += Double(x[k * width])
                    sum.1 += Double(x[k * width + 1])
                }
                next[q * width] = Float16(sum.0 / Double(keys.count) + Double(x0[q * width]) * 0.01)
                next[q * width + 1] = Float16(sum.1 / Double(keys.count))
            }
        }
        x = next
    }
    var output = [Float](repeating: 0, count: 768)
    for row in 0..<n {
        output[0] += Float(LongContract.poolWeight) * Float(x[row * width])
        output[1] += Float(LongContract.poolWeight) * Float(x[row * width + 1])
    }
    return output
}

@Suite("Long text path")
struct LongTextTests {
    @Test func chunkBiasFollowsTheContract() {
        for (b, n) in [(0, 2049), (1, 2049), (2, 2049), (3, 3333), (7, 8192), (4, 4097)] {
            let bias = ChunkBias(ChunkBiasKey(chunk: b, tokens: n))
            for r in stride(from: 0, to: 1024, by: 37) {
                for c in 0..<2048 {
                    let q = b * 1024 + r, k = b * 1024 - 512 + c
                    let open = (abs(q - k) <= 512 && k >= 0 && k < n) || k == q
                    #expect(bias[r, c] == (open ? 0 : -10_000), "b=\(b) n=\(n) r=\(r) c=\(c)")
                }
            }
        }
    }

    @Test func fewDistinctBiasesPerText() {
        let cache = ChunkBiasCache()
        let n = 8000
        let keys = Set((0..<LongContract.liveChunks(tokens: n)).map { ChunkBiasKey(chunk: $0, tokens: n) })
        #expect(keys.count == 3)  // first chunk, full-window middle chunks, last chunk
        for b in 0..<LongContract.liveChunks(tokens: n) { _ = cache.bias(chunk: b, tokens: n) }
        #expect(cache.builds == 3)
        for b in 0..<LongContract.liveChunks(tokens: 5000) { _ = cache.bias(chunk: b, tokens: 5000) }
        #expect(cache.builds == 4)  // only the new tail
    }

    @Test func callOrderAndCount() async throws {
        let layers = AveragingLayers(lengths: [4096, 8192], compute: false)
        let engine = LongTextEngine(functions: layers, rows: TinyRows())
        _ = try await engine.embed(ids: (0..<2100).map { Int32($0) })
        #expect(layers.calls.count == LongContract.calls(tokens: 2100))
        #expect(layers.calls.count == 20 * 3 + 4)
        #expect(layers.calls.prefix(3) == ["long_swa_l0", "long_swa_l0", "long_swa_l0"])
        #expect(layers.calls[5 * 3] == "long4096_glb_l5")
        #expect(layers.calls.last == "long4096_head")
        let layers8192 = AveragingLayers(lengths: [4096, 8192], compute: false)
        _ = try await LongTextEngine(functions: layers8192, rows: TinyRows()).embed(ids: [Int32](repeating: 5, count: 4097))
        #expect(layers8192.calls.last == "long8192_head" && layers8192.calls.count == 20 * 5 + 4)
    }

    @Test(arguments: [2049, 4097])
    func chunkedLayersEqualTheWholeSequence(tokens: Int) async throws {
        let ids = (0..<tokens).map { Int32(($0 * 7919) % 262_144) }
        let engine = LongTextEngine(functions: AveragingLayers(lengths: [4096, 8192]), rows: TinyRows())
        let chunked = try await engine.embed(ids: ids)
        let whole = reference(ids: ids)
        #expect(abs(chunked[0] - whole[0]) < 1e-5 && abs(chunked[1] - whole[1]) < 1e-5, "\(chunked.prefix(2)) vs \(whole.prefix(2))")
        // Buffers are reused: a shorter text after a longer one must give the same result as alone.
        let shorter = Array(ids.prefix(2049))
        let again = try await engine.embed(ids: shorter)
        let alone = try await LongTextEngine(functions: AveragingLayers(lengths: [4096, 8192]), rows: TinyRows()).embed(ids: shorter)
        #expect(again == alone)
    }

    @Test func lengthsForMaxTokens() {
        #expect(CoreMLLongFunctions.lengths(forMaxTokens: 2048).isEmpty)
        #expect(CoreMLLongFunctions.lengths(forMaxTokens: 2049) == [4096])
        #expect(CoreMLLongFunctions.lengths(forMaxTokens: 4096) == [4096])
        #expect(CoreMLLongFunctions.lengths(forMaxTokens: 8192) == [4096, 8192])
        #expect(CoreMLLongFunctions.functionNames(lengths: [4096, 8192]).count == 28)
        #expect(CoreMLLongFunctions.functionNames(lengths: [4096]).count == 24)
    }

    @Test func tooLongIsRejected() async {
        let engine = LongTextEngine(functions: AveragingLayers(lengths: [4096]), rows: TinyRows())
        await #expect(throws: RuntimeError.self) { _ = try await engine.embed(ids: [Int32](repeating: 1, count: 4097)) }
    }
}
