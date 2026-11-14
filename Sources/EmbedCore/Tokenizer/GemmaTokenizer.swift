// The EmbeddingGemma 2 tokenizer, written from the tokenizer.json specification and checked token
// for token against Hugging Face `tokenizers` (Tests/Fixtures/tokenizer-golden.jsonl).
//
// Pipeline (what `tokenizers` does for this file):
//   1. added tokens: the 24 special tokens (<bos>, <|image>, ...) are cut out of the raw text first,
//      leftmost-longest, and become their ids directly (they are not normalized)
//   2. normalizer: Replace " " with "▁" (U+2581) in every other piece
//   3. pre_tokenizer: Split on " " (MergedWithPrevious). After step 2 no " " is left, so each piece is
//      one word; this is why BPE has to stay O(n log n) on a whole document
//   4. BPE: one symbol per Unicode scalar (its vocab id, or <0xXX> byte tokens when the scalar is not in
//      the vocab), then merges applied lowest rank first (leftmost on ties) with a priority queue
//   5. post_processor: <bos> ... <eos>
//
// Vocabulary keys are handled as bytes from end to end. Two traps this avoids: JSONSerialization drops a
// leading U+FEFF from keys (8 vocab entries start with one), and Swift String keys merge canonically
// equivalent entries such as U+F900 and U+8C48, which are different tokens.
import Foundation

public enum TokenizerError: Error, CustomStringConvertible, Sendable {
    case unreadable(String)
    case unsupported(String)
    case malformed(String)

    public var description: String {
        switch self {
        case .unreadable(let message): return "cannot read tokenizer.json: \(message)"
        case .unsupported(let message): return "unsupported tokenizer.json: \(message)"
        case .malformed(let message): return "malformed tokenizer.json: \(message)"
        }
    }
}

public final class GemmaTokenizer: Sendable {
    public let vocabularySize: Int
    public let mergeCount: Int
    /// Ids the post-processor puts before and after every text (<bos> and <eos> for EmbeddingGemma 2).
    public let prefixIDs: [Int32]
    public let suffixIDs: [Int32]
    public let unknownID: Int32?

    /// Single-scalar vocabulary entries: BMP scalars in a flat table, the rest in a dictionary.
    private let bmpIDs: [Int32]
    private let astralIDs: [UInt32: Int32]
    /// <0x00> ... <0xFF>
    private let byteIDs: [Int32]
    /// (left id << 32 | right id) -> (rank << 32 | merged id)
    private let merges: PairTable
    /// Added tokens, longest first within each leading byte.
    private let addedTokens: [[UInt8]]
    private let addedTokenIDs: [Int32]
    private let addedFirstBytes: [Bool]
    /// Token bytes by id (for `tokens(_:)`).
    private let tokenBytes: [[UInt8]]

    public convenience init(contentsOf url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw TokenizerError.unreadable("\(url.path): \(error.localizedDescription)")
        }
        try self.init(data: data)
    }

    public init(data: Data) throws {
        var parsed = try data.withUnsafeBytes { raw in try TokenizerFile.parse(raw) }

        let vocab = parsed.vocab
        vocabularySize = vocab.count
        var bmp = [Int32](repeating: -1, count: 0x10000)
        var astral: [UInt32: Int32] = [:]
        let maxID = (0..<vocab.count).map { Int(vocab.value(at: $0)) }.max() ?? 0
        guard maxID < 16 * vocab.count + 1024 else { throw TokenizerError.malformed("vocab ids are too sparse") }
        var bytesByID = [[UInt8]](repeating: [], count: maxID + 1)
        for entry in 0..<vocab.count {
            let key = Array(vocab.key(at: entry))
            let id = vocab.value(at: entry)
            bytesByID[Int(id)] = key
            if let scalar = singleScalar(key) {
                if scalar < 0x10000 { bmp[Int(scalar)] = id } else { astral[scalar] = id }
            }
        }
        bmpIDs = bmp
        astralIDs = astral
        tokenBytes = bytesByID

        var bytes = [Int32](repeating: -1, count: 256)
        let hex = Array("0123456789ABCDEF".utf8)
        for value in 0..<256 {
            let name: [UInt8] = Array("<0x".utf8) + [hex[value >> 4], hex[value & 15], UInt8(ascii: ">")]
            bytes[value] = vocab.lookup(name) ?? -1
        }
        byteIDs = bytes
        unknownID = parsed.unknownToken.flatMap { vocab.lookup(Array($0.utf8)) }

        // Merges: rank = position in the list; later duplicates overwrite earlier ones.
        let mergeTotal = parsed.mergeCount
        var table = PairTable(minimumCapacity: max(mergeTotal, 1))
        var joined: [UInt8] = []
        try parsed.mergeStorage.withUnsafeBytes { storage in
            for rank in 0..<mergeTotal {
                let left = UnsafeRawBufferPointer(rebasing: storage[parsed.mergeRange(2 * rank)])
                let right = UnsafeRawBufferPointer(rebasing: storage[parsed.mergeRange(2 * rank + 1)])
                guard let leftID = vocab.lookup(left), let rightID = vocab.lookup(right) else {
                    throw TokenizerError.malformed("merge \(rank) refers to a token missing from the vocab")
                }
                joined.removeAll(keepingCapacity: true)
                joined.append(contentsOf: left)
                joined.append(contentsOf: right)
                guard let mergedID = joined.withUnsafeBytes({ vocab.lookup($0) }) else {
                    throw TokenizerError.malformed("merge \(rank) produces a token missing from the vocab")
                }
                let key = UInt64(UInt32(bitPattern: leftID)) << 32 | UInt64(UInt32(bitPattern: rightID))
                table.insert(key, UInt64(UInt32(rank)) << 32 | UInt64(UInt32(bitPattern: mergedID)))
            }
        }
        merges = table
        mergeCount = mergeTotal
        parsed.mergeStorage = []

        let added = parsed.addedTokens.sorted { $0.content.count > $1.content.count }
        addedTokens = added.map(\.content)
        addedTokenIDs = added.map(\.id)
        var first = [Bool](repeating: false, count: 256)
        for token in added where !token.content.isEmpty { first[Int(token.content[0])] = true }
        addedFirstBytes = first
        prefixIDs = parsed.prefixIDs
        suffixIDs = parsed.suffixIDs
    }

    // MARK: - Encoding

    /// Token ids of `text`, with <bos>/<eos> unless `addSpecialTokens` is false.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int32] {
        var output: [Int32] = []
        var scratch = BPEScratch()
        encode(text, addSpecialTokens: addSpecialTokens, into: &output, scratch: &scratch)
        return output
    }

    /// Appends the ids of `text` to `output`, reusing `scratch` between calls.
    public func encode(_ text: String, addSpecialTokens: Bool = true, into output: inout [Int32],
                       scratch: inout BPEScratch)
    {
        output.reserveCapacity(output.count + text.utf8.count / 3 + 2)
        if addSpecialTokens { output.append(contentsOf: prefixIDs) }
        var text = text
        text.withUTF8 { utf8 in
            var pieceStart = 0
            var index = 0
            while index < utf8.count {
                if addedFirstBytes[Int(utf8[index])], let (tokenIndex, length) = matchAddedToken(utf8, at: index) {
                    if index > pieceStart {
                        encodePiece(UnsafeBufferPointer(rebasing: utf8[pieceStart..<index]), into: &output,
                                    scratch: &scratch)
                    }
                    output.append(addedTokenIDs[tokenIndex])
                    index += length
                    pieceStart = index
                } else {
                    index += 1
                }
            }
            if utf8.count > pieceStart {
                encodePiece(UnsafeBufferPointer(rebasing: utf8[pieceStart..<utf8.count]), into: &output,
                            scratch: &scratch)
            }
        }
        if addSpecialTokens { output.append(contentsOf: suffixIDs) }
    }

    /// The number of tokens `encode` would return.
    public func count(_ text: String, addSpecialTokens: Bool = true) -> Int {
        encode(text, addSpecialTokens: addSpecialTokens).count
    }

    /// The vocabulary entry of each id as text (byte tokens stay "<0xXX>").
    public func tokens(_ ids: [Int32]) -> [String] {
        ids.map { id in
            guard id >= 0, Int(id) < tokenBytes.count else { return "<invalid \(id)>" }
            return String(decoding: tokenBytes[Int(id)], as: UTF8.self)
        }
    }

    /// The id of a vocabulary entry, compared byte for byte.
    public func id(of token: String) -> Int32? {
        let bytes = Array(token.utf8)
        // Linear only for the rare debug use; encoding never calls this.
        return tokenBytes.firstIndex(of: bytes).map { Int32($0) }
    }

    /// Leftmost-longest match of an added token at `index` (tokens are sorted longest first).
    @inline(__always)
    private func matchAddedToken(_ utf8: UnsafeBufferPointer<UInt8>, at index: Int) -> (Int, Int)? {
        for (tokenIndex, token) in addedTokens.enumerated() {
            let length = token.count
            guard length > 0, index + length <= utf8.count else { continue }
            var equal = true
            for offset in 0..<length where utf8[index + offset] != token[offset] {
                equal = false
                break
            }
            if equal { return (tokenIndex, length) }
        }
        return nil
    }

    /// Normalizes (" " -> "▁"), splits into initial symbols and runs BPE on one piece.
    private func encodePiece(_ utf8: UnsafeBufferPointer<UInt8>, into output: inout [Int32],
                             scratch: inout BPEScratch)
    {
        scratch.reset()
        var index = 0
        var pendingUnknown = false
        while index < utf8.count {
            let (scalar, length) = decodeScalar(utf8, at: index)
            let normalized: UInt32 = scalar == 0x20 ? 0x2581 : scalar
            let id: Int32
            if normalized < 0x10000 {
                id = bmpIDs[Int(normalized)]
            } else {
                id = astralIDs[normalized] ?? -1
            }
            if id >= 0 {
                pendingUnknown = false
                scratch.append(id)
            } else {
                // byte_fallback: <0xXX> per UTF-8 byte of the (normalized) scalar
                var encoded: [UInt8] = []
                appendUTF8(normalized, to: &encoded)
                if encoded.allSatisfy({ byteIDs[Int($0)] >= 0 }) {
                    pendingUnknown = false
                    for byte in encoded { scratch.append(byteIDs[Int(byte)]) }
                } else if let unknownID {
                    // fuse_unk: consecutive unknowns become one <unk>
                    if !pendingUnknown { scratch.append(unknownID) }
                    pendingUnknown = true
                }
            }
            index += length
        }
        scratch.mergeAll(merges)
        scratch.collect(into: &output)
    }
}

/// Reusable working memory for BPE: a doubly linked list of symbols and a binary min-heap of
/// candidate merges ordered by (rank, position).
public struct BPEScratch: Sendable {
    var ids: [Int32] = []
    var previous: [Int32] = []
    var next: [Int32] = []
    var heap: [HeapEntry] = []

    struct HeapEntry: Sendable {
        /// rank << 32 | position: the heap order
        var order: UInt64
        var mergedID: Int32
    }

    public init() {}

    mutating func reset() {
        ids.removeAll(keepingCapacity: true)
        previous.removeAll(keepingCapacity: true)
        next.removeAll(keepingCapacity: true)
        heap.removeAll(keepingCapacity: true)
    }

    @inline(__always)
    mutating func append(_ id: Int32) {
        let index = Int32(ids.count)
        ids.append(id)
        previous.append(index - 1)
        next.append(index + 1)
    }

    @inline(__always)
    private func pairKey(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }

    @inline(__always)
    private mutating func push(position: Int32, merges: PairTable) {
        let right = next[Int(position)]
        guard right < Int32(ids.count), let found = merges.lookup(pairKey(ids[Int(position)], ids[Int(right)])) else {
            return
        }
        let entry = HeapEntry(order: (found >> 32) << 32 | UInt64(UInt32(position)),
                              mergedID: Int32(bitPattern: UInt32(truncatingIfNeeded: found)))
        heap.append(entry)
        var child = heap.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            if heap[parent].order <= heap[child].order { break }
            heap.swapAt(parent, child)
            child = parent
        }
    }

    private mutating func pop() -> HeapEntry? {
        guard let top = heap.first else { return nil }
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var parent = 0
            while true {
                let left = 2 * parent + 1, right = left + 1
                var smallest = parent
                if left < heap.count, heap[left].order < heap[smallest].order { smallest = left }
                if right < heap.count, heap[right].order < heap[smallest].order { smallest = right }
                if smallest == parent { break }
                heap.swapAt(parent, smallest)
                parent = smallest
            }
        }
        return top
    }

    /// Applies merges until none is left (tokenizers' Word::merge_all without dropout).
    mutating func mergeAll(_ merges: PairTable) {
        let count = Int32(ids.count)
        guard count > 1 else { return }
        for position in 0..<(count - 1) { push(position: position, merges: merges) }
        while let top = pop() {
            let position = Int32(truncatingIfNeeded: top.order & 0xFFFF_FFFF)
            // Skip stale entries: the symbol was merged away, or its pair no longer produces the same
            // token. Like tokenizers, only the produced id is compared, not the rank.
            guard ids[Int(position)] >= 0 else { continue }
            let right = next[Int(position)]
            guard right < count,
                let found = merges.lookup(pairKey(ids[Int(position)], ids[Int(right)])),
                Int32(bitPattern: UInt32(truncatingIfNeeded: found)) == top.mergedID
            else { continue }
            // Merge right into position.
            ids[Int(position)] = top.mergedID
            ids[Int(right)] = -1
            let after = next[Int(right)]
            next[Int(position)] = after
            if after < count { previous[Int(after)] = position }
            let before = previous[Int(position)]
            if before >= 0 { push(position: before, merges: merges) }
            push(position: position, merges: merges)
        }
    }

    func collect(into output: inout [Int32]) {
        guard !ids.isEmpty else { return }
        var index: Int32 = 0
        while index < Int32(ids.count) {
            output.append(ids[Int(index)])
            index = next[Int(index)]
        }
    }
}

// MARK: - Helpers

@inline(__always)
func decodeScalar(_ utf8: UnsafeBufferPointer<UInt8>, at index: Int) -> (UInt32, Int) {
    let lead = UInt32(utf8[index])
    if lead < 0x80 { return (lead, 1) }
    if lead < 0xE0 { return ((lead & 0x1F) << 6 | UInt32(utf8[index + 1] & 0x3F), 2) }
    if lead < 0xF0 {
        return ((lead & 0x0F) << 12 | UInt32(utf8[index + 1] & 0x3F) << 6 | UInt32(utf8[index + 2] & 0x3F), 3)
    }
    return ((lead & 0x07) << 18 | UInt32(utf8[index + 1] & 0x3F) << 12 | UInt32(utf8[index + 2] & 0x3F) << 6
        | UInt32(utf8[index + 3] & 0x3F), 4)
}

/// The scalar value when `bytes` is exactly one well-formed UTF-8 scalar.
func singleScalar(_ bytes: [UInt8]) -> UInt32? {
    guard (1...4).contains(bytes.count) else { return nil }
    var iterator = bytes.makeIterator()
    var decoder = UTF8()
    guard case .scalarValue(let scalar) = decoder.decode(&iterator) else { return nil }
    guard case .emptyInput = decoder.decode(&iterator) else { return nil }
    return scalar.value
}
