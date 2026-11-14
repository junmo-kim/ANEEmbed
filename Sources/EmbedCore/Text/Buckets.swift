// The uniform text-function contract and the packing planner.
//
// Every text function has the same signature, so single texts, packed batches, long texts and (later)
// images share one code path:
//
//   embed_S, S in {64, 128, 256, 512, 1024, 2048}
//     inputs_embeds [1, S, 512] fp16   token rows (table row * sqrt(512)); padding rows are finite
//     global_bias   [1, 1, S, S] fp16  0 where query i and key j belong to the same text, else -1e4
//     local_bias    [1, 1, S, S] fp16  global_bias AND |i - j| <= 512 (Gemma's sliding window)
//     pool          [K, S] fp16        K = max(8, S / 32); row k = 1/len over text k's tokens, other rows 0
//     -> embedding  [K, 768] fp32      already L2-normalized
//
// Padding rows attend only to themselves, so no row is fully masked. Positions are not an input: RoPE
// is relative, and attention never crosses texts, so a packed text sees the same scores as alone.
// The band is an input rather than a graph constant because a constant band miscompiles on the ANE.
//
// Names may change with the final model package; everything here is behind `BucketContract`.

public enum BucketContract {
    public static let lengths = [64, 128, 256, 512, 1024, 2048]
    public static let hidden = TokenTable.width
    public static let dimension = 768
    public static let window = 512
    public static let blocked: Float16 = -10_000

    public static func slots(forLength length: Int) -> Int { max(8, length / 32) }
    public static func functionName(forLength length: Int) -> String { "embed_\(length)" }

    /// Input and output feature names.
    public static let inputsEmbeds = "inputs_embeds"
    public static let globalBias = "global_bias"
    public static let localBias = "local_bias"
    public static let pool = "pool"
    public static let embedding = "embedding"
}

/// One text placed in a packed call: rows offset..<offset+length, pool row = its index in the call.
public struct PackedSegment: Sendable, Equatable {
    /// Index of the text in the caller's list.
    public let item: Int
    public let offset: Int
    public let length: Int

    public init(item: Int, offset: Int, length: Int) {
        self.item = item
        self.offset = offset
        self.length = length
    }
}

/// One function call: a bucket length and the texts packed into it.
public struct PackedCall: Sendable, Equatable {
    public let length: Int
    public let segments: [PackedSegment]

    public var slots: Int { BucketContract.slots(forLength: length) }
    public var usedTokens: Int { segments.reduce(0) { $0 + $1.length } }

    public init(length: Int, segments: [PackedSegment]) {
        self.length = length
        self.segments = segments
    }
}

/// Splits a list of texts (token counts) into calls of the available buckets, minimizing an
/// estimated total cost.
///
/// For every bucket length C it packs first-fit-decreasing into bins of C tokens and C's slot count
/// (a text longer than C gets a bin of its own), shrinks each bin to the smallest bucket that holds it,
/// and sums the per-call costs; the cheapest C wins. Small C means many cheap calls, large C fewer
/// expensive ones; which is better depends on the measured per-call latency of each bucket.
public struct PackingPlanner: Sendable {
    public let lengths: [Int]
    /// Estimated cost of one call per bucket length (any unit).
    public let costs: [Int: Double]

    /// Per-call latency in ms of each bucket, one text filling it (`aneembed bench --buckets`, Apple M4,
    /// macOS 27). Only the ratios matter to the planner; other chips scale them roughly together.
    public static let defaultCosts: [Int: Double] = [64: 5.0, 128: 6.5, 256: 12.0, 512: 30.6, 1024: 80.7, 2048: 247.3]

    public init(lengths: [Int] = BucketContract.lengths, costs: [Int: Double] = PackingPlanner.defaultCosts) {
        precondition(!lengths.isEmpty)
        self.lengths = lengths.sorted()
        self.costs = costs
    }

    public var maxLength: Int { lengths.last! }

    public func cost(ofLength length: Int) -> Double {
        costs[length] ?? Double(length)
    }

    /// The smallest bucket that holds `tokens` tokens in `texts` slots.
    public func bucket(tokens: Int, texts: Int = 1) -> Int? {
        lengths.first { $0 >= tokens && BucketContract.slots(forLength: $0) >= texts }
    }

    /// Calls covering every text exactly once. Precondition: every count is 1...maxLength.
    public func plan(_ counts: [Int]) -> [PackedCall] {
        guard !counts.isEmpty else { return [] }
        precondition(counts.allSatisfy { $0 >= 1 && $0 <= maxLength }, "token count outside 1...\(maxLength)")
        let order = counts.indices.sorted { counts[$0] != counts[$1] ? counts[$0] > counts[$1] : $0 < $1 }
        var best: [PackedCall] = []
        var bestCost = Double.infinity
        for capacity in lengths {
            let calls = pack(order: order, counts: counts, capacity: capacity)
            let total = calls.reduce(0) { $0 + cost(ofLength: $1.length) }
            if total < bestCost {
                best = calls
                bestCost = total
            }
        }
        return best
    }

    private func pack(order: [Int], counts: [Int], capacity: Int) -> [PackedCall] {
        let slots = BucketContract.slots(forLength: capacity)
        var bins: [[Int]] = []  // item indices
        var used: [Int] = []
        var open: [Int] = []  // bins that can still take texts
        for item in order {
            let count = counts[item]
            if count > capacity {
                bins.append([item])
                used.append(count)
                continue
            }
            var placed = false
            for (position, bin) in open.enumerated() where used[bin] + count <= capacity && bins[bin].count < slots {
                bins[bin].append(item)
                used[bin] += count
                if bins[bin].count == slots || used[bin] == capacity { open.remove(at: position) }
                placed = true
                break
            }
            if !placed {
                bins.append([item])
                used.append(count)
                open.append(bins.count - 1)
            }
        }
        return bins.indices.map { index in
            var offset = 0
            let segments = bins[index].map { item -> PackedSegment in
                defer { offset += counts[item] }
                return PackedSegment(item: item, offset: offset, length: counts[item])
            }
            return PackedCall(length: bucket(tokens: used[index], texts: segments.count)!, segments: segments)
        }
    }
}
