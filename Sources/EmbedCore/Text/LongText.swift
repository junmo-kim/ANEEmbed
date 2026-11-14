// Texts of 2049...8192 tokens: the long path of the model package, per layer and per chunk.
//
// The bucket functions attend over the whole sequence in one call, which does not fit the Neural Engine
// beyond 2048 rows. The long path splits the 24 layers instead:
//
//   S = 4096 if n <= 4096 else 8192; x0 = token rows (as for buckets), rows >= n zero
//   sliding layer i (20 of them): for each chunk b with b*1024 < n
//       long_swa_l{i}(hidden_window = [512 zero rows; x; 512 zero rows][b*1024 ..< b*1024 + 2048],
//                     inputs_embeds = x0[b*1024 ..< (b+1)*1024], bias = chunk_bias(b, n)) -> rows of chunk b
//       chunks past the text keep their rows; every chunk reads the previous x (double buffer)
//   global layer 5, 11, 17: long{S}_glb_l{i}(hidden = x, inputs_embeds = x0, key_bias) -> x
//   layer 23 + pooling + projection + L2: long{S}_head(hidden = x, inputs_embeds = x0, key_bias, pool)
//
//   chunk_bias(b, n) [1024, 2048]: query q = b*1024 + r, key k = b*1024 - 512 + c;
//                                  0 if (|q - k| <= 512 and 0 <= k < n) or k == q, else -1e4
//   key_bias [S]: 0 for k < n, else -1e4      pool [S]: 2^-10 for rows < n (not 1/n: fp16 underflow)
//
// 20 * ceil(n / 1024) + 4 calls per text, one after another: submitting a layer's chunks concurrently
// and writing outputs through outputBackings were measured and gained nothing (the Neural Engine runs
// one call at a time), while concurrent chunks doubled the footprint. The host does no math on activations: it only gathers token
// rows, slices windows (views into one padded buffer, no copies) and builds biases. A chunk bias depends
// on n only through the window's first and last valid key column, so the few distinct ones are cached
// and shared across texts.
import Foundation

public enum LongContract {
    public static let lengths = [4096, 8192]
    public static let chunk = 1024
    public static let halo = 512
    public static let window = 512
    /// Rows of one hidden window: a chunk plus a halo on each side.
    public static let windowRows = chunk + 2 * halo
    public static let layers = 24
    public static let slidingLayers = (0..<24).filter { $0 % 6 != 5 }
    /// Global layers run as separate functions; the last one (23) is part of the head.
    public static let globalLayers = [5, 11, 17]
    public static let poolWeight: Float16 = 0.000_976_562_5  // 2^-10
    public static let hidden = BucketContract.hidden
    public static let blocked = BucketContract.blocked

    public static func length(forTokens tokens: Int) -> Int? { lengths.first { $0 >= tokens } }
    public static func liveChunks(tokens: Int) -> Int { (tokens + chunk - 1) / chunk }
    public static func calls(tokens: Int) -> Int { slidingLayers.count * liveChunks(tokens: tokens) + globalLayers.count + 1 }

    public static func slidingName(layer: Int) -> String { "long_swa_l\(layer)" }
    public static func globalName(length: Int, layer: Int) -> String { "long\(length)_glb_l\(layer)" }
    public static func headName(length: Int) -> String { "long\(length)_head" }
    /// Every function the long path for `length` calls.
    public static func functionNames(length: Int) -> [String] {
        slidingLayers.map { slidingName(layer: $0) } + globalLayers.map { globalName(length: length, layer: $0) }
            + [headName(length: length)]
    }

    /// Input and output feature names.
    public static let hiddenWindow = "hidden_window"
    public static let inputsEmbeds = "inputs_embeds"
    public static let bias = "bias"
    public static let hiddenInput = "hidden"
    public static let keyBias = "key_bias"
    public static let pool = "pool"
    public static let hiddenOut = "hidden_out"
    public static let embedding = "embedding"
}

/// The valid key columns [lower, upper) of one chunk's window. Equal keys mean equal biases.
public struct ChunkBiasKey: Hashable, Sendable {
    public let lower: Int
    public let upper: Int

    public init(chunk b: Int, tokens n: Int) {
        let start = b * LongContract.chunk - LongContract.halo  // sequence row of window column 0
        lower = max(0, -start)
        upper = max(lower, min(LongContract.windowRows, n - start))
    }
}

/// One chunk bias, [1024, 2048] fp16, page aligned.
public final class ChunkBias: @unchecked Sendable {
    public let key: ChunkBiasKey
    public let values: UnsafeMutablePointer<Float16>

    public init(_ key: ChunkBiasKey) {
        self.key = key
        let rows = LongContract.chunk, columns = LongContract.windowRows, halo = LongContract.halo
        let raw = UnsafeMutableRawPointer.allocate(byteCount: rows * columns * 2, alignment: BucketBuffers.alignment)
        values = raw.bindMemory(to: Float16.self, capacity: rows * columns)
        values.initialize(repeating: LongContract.blocked, count: rows * columns)
        for r in 0..<rows {
            let row = values + r * columns
            // |q - k| <= window  <=>  column in [r + halo - window, r + halo + window]
            let lower = max(key.lower, r + halo - LongContract.window)
            let upper = min(key.upper, r + halo + LongContract.window + 1)
            if upper > lower { (row + lower).update(repeating: 0, count: upper - lower) }
            row[r + halo] = 0  // a row always sees itself, so no row is fully masked
        }
    }

    deinit { values.deallocate() }

    /// The value at (row, column), for tests.
    public subscript(row: Int, column: Int) -> Float16 { values[row * LongContract.windowRows + column] }
}

/// Shared chunk biases. Full-window chunks (all of them but the first and the last one or two of a
/// text) have one of two keys and stay cached; tail keys depend on n and are kept while recently used.
public final class ChunkBiasCache: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [ChunkBiasKey: (bias: ChunkBias, used: UInt64)] = [:]
    private var clock: UInt64 = 0
    private let capacity: Int
    public private(set) var builds = 0

    public init(capacity: Int = 8) {
        self.capacity = max(2, capacity)
    }

    public func bias(chunk b: Int, tokens n: Int) -> ChunkBias {
        let key = ChunkBiasKey(chunk: b, tokens: n)
        if let cached = lock.withLock({ () -> ChunkBias? in
            clock += 1
            guard let entry = entries[key] else { return nil }
            entries[key] = (entry.bias, clock)
            return entry.bias
        }) { return cached }
        let bias = ChunkBias(key)  // built outside the lock; a racing duplicate is harmless
        lock.withLock {
            builds += 1
            clock += 1
            entries[key] = (bias, clock)
            while entries.count > capacity {
                let full = LongContract.windowRows
                let victim = entries.filter { $0.key.upper != full }.min { $0.value.used < $1.value.used }
                    ?? entries.min { $0.value.used < $1.value.used }!
                entries[victim.key] = nil
            }
        }
        return bias
    }

    public var count: Int { lock.withLock { entries.count } }
}

/// The Core ML side of the long path: one call per function, inputs and outputs as fp16 rows.
/// Pointers stay valid for the duration of each call.
public protocol LongFunctions: Sendable {
    var lengths: [Int] { get }
    /// long_swa_l{layer}: window [2048, 512], embeds [1024, 512], bias [1024, 2048] -> output [1024, 512]
    func sliding(layer: Int, window: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                 bias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    /// long{length}_glb_l{layer}: hidden and embeds [length, 512], key bias [length] -> output [length, 512]
    func global(length: Int, layer: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
                keyBias: UnsafeMutablePointer<Float16>, into output: UnsafeMutablePointer<Float16>) async throws
    /// long{length}_head: as global, plus pool [length] -> L2-normalized embedding (768)
    func head(length: Int, hidden: UnsafeMutablePointer<Float16>, embeds: UnsafeMutablePointer<Float16>,
              keyBias: UnsafeMutablePointer<Float16>, pool: UnsafeMutablePointer<Float16>) async throws -> [Float]
}

/// Reusable buffers for one sequence length: x0, the padded double buffer, key bias and pool.
public final class LongBuffers: @unchecked Sendable {
    public let length: Int
    /// Token rows [length, 512].
    public let embeds: UnsafeMutablePointer<Float16>
    /// Two padded hidden buffers [halo + length + halo, 512]; the halos stay zero.
    public let padded: (UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>)
    public let keyBias: UnsafeMutablePointer<Float16>
    public let pool: UnsafeMutablePointer<Float16>
    public private(set) var tokens = 0
    private var dirtyRows = 0

    public init(length: Int) {
        self.length = length
        let hidden = LongContract.hidden, paddedRows = length + 2 * LongContract.halo
        func allocate(_ count: Int, fill: Float16) -> UnsafeMutablePointer<Float16> {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: count * 2, alignment: BucketBuffers.alignment)
            let pointer = raw.bindMemory(to: Float16.self, capacity: count)
            pointer.initialize(repeating: fill, count: count)
            return pointer
        }
        embeds = allocate(length * hidden, fill: 0)
        padded = (allocate(paddedRows * hidden, fill: 0), allocate(paddedRows * hidden, fill: 0))
        keyBias = allocate(length, fill: LongContract.blocked)
        pool = allocate(length, fill: 0)
    }

    deinit {
        embeds.deallocate()
        padded.0.deallocate()
        padded.1.deallocate()
        keyBias.deallocate()
        pool.deallocate()
    }

    /// Row `row` of a padded buffer's sequence (row 0 is the first token, after the leading halo).
    public func sequenceRow(_ buffer: UnsafeMutablePointer<Float16>, _ row: Int) -> UnsafeMutablePointer<Float16> {
        buffer + (LongContract.halo + row) * LongContract.hidden
    }

    /// Gathers the token rows and sets key bias, pool and the first hidden state (x = x0).
    public func prepare(ids: [Int32], rows: any TokenRowSource) {
        let n = ids.count, hidden = LongContract.hidden
        precondition(n >= 1 && n <= length)
        rows.gather(ids, into: embeds)
        if dirtyRows > n { (embeds + n * hidden).update(repeating: 0, count: (dirtyRows - n) * hidden) }
        dirtyRows = n
        if n != tokens {
            keyBias.update(repeating: 0, count: n)
            (keyBias + n).update(repeating: LongContract.blocked, count: length - n)
            pool.update(repeating: LongContract.poolWeight, count: n)
            (pool + n).update(repeating: 0, count: length - n)
            tokens = n
        }
        // Global layers write every row, so the whole sequence is reset, not only the first n rows.
        sequenceRow(padded.0, 0).update(from: embeds, count: length * hidden)
    }
}

/// The long path over `LongFunctions`, with buffers pooled per length.
public final class LongTextEngine: LongTextRunner, @unchecked Sendable {
    public let functions: any LongFunctions
    public let lengths: [Int]
    private let rows: any TokenRowSource
    private let biases: ChunkBiasCache
    private let lock = NSLock()
    private var idle: [Int: [LongBuffers]] = [:]
    private let maxIdlePerLength: Int

    public init(functions: any LongFunctions, rows: any TokenRowSource, maxIdlePerLength: Int = 1,
                biases: ChunkBiasCache = ChunkBiasCache())
    {
        self.functions = functions
        lengths = functions.lengths.sorted()
        self.rows = rows
        self.biases = biases
        self.maxIdlePerLength = maxIdlePerLength
    }

    public var maxTokens: Int { lengths.last ?? 0 }

    public func warm() async throws {}

    public func embed(ids: [Int32]) async throws -> [Float] {
        let n = ids.count
        guard let length = lengths.first(where: { $0 >= n }) else {
            throw RuntimeError.tooLong(tokens: n, max: maxTokens)
        }
        let buffers = checkout(length: length)
        defer { checkin(buffers) }
        buffers.prepare(ids: ids, rows: rows)
        return try await run(buffers)
    }

    /// Runs the 24 layers on prepared buffers.
    func run(_ buffers: LongBuffers) async throws -> [Float] {
        let n = buffers.tokens, length = buffers.length, hidden = LongContract.hidden, chunk = LongContract.chunk
        let live = LongContract.liveChunks(tokens: n), chunks = length / chunk
        let chunkBiases = (0..<live).map { biases.bias(chunk: $0, tokens: n) }
        var (source, target) = buffers.padded
        for layer in 0..<(LongContract.layers - 1) {
            if LongContract.globalLayers.contains(layer) {
                try await functions.global(length: length, layer: layer, hidden: buffers.sequenceRow(source, 0),
                                           embeds: buffers.embeds, keyBias: buffers.keyBias,
                                           into: buffers.sequenceRow(target, 0))
            } else {
                for b in 0..<live {
                    // window rows b*1024 - 512 ..< b*1024 + 1536 of the sequence = padded rows b*1024 ..< +2048
                    try await functions.sliding(layer: layer, window: source + b * chunk * hidden,
                                                embeds: buffers.embeds + b * chunk * hidden,
                                                bias: chunkBiases[b].values,
                                                into: buffers.sequenceRow(target, b * chunk))
                }
                if live < chunks {
                    buffers.sequenceRow(target, live * chunk).update(
                        from: buffers.sequenceRow(source, live * chunk), count: (chunks - live) * chunk * hidden)
                }
            }
            swap(&source, &target)
        }
        return try await functions.head(length: length, hidden: buffers.sequenceRow(source, 0), embeds: buffers.embeds,
                                        keyBias: buffers.keyBias, pool: buffers.pool)
    }

    private func checkout(length: Int) -> LongBuffers {
        if let buffers = lock.withLock({ idle[length]?.popLast() }) { return buffers }
        return LongBuffers(length: length)
    }

    private func checkin(_ buffers: LongBuffers) {
        lock.withLock {
            var list = idle[buffers.length, default: []]
            if list.count < maxIdlePerLength { list.append(buffers) }
            idle[buffers.length] = list
        }
    }
}

