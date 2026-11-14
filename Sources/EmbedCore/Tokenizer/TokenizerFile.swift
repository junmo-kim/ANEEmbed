// Streaming reader for tokenizer.json. The vocabulary (262,144 entries) and merges (514,906 pairs) are
// read straight from the bytes into flat buffers; only the small sections become JSONValue trees.
// Anything this tokenizer does not implement is rejected at load time instead of tokenizing wrongly.
import Foundation

struct AddedToken {
    var content: [UInt8]
    var id: Int32
}

struct TokenizerFile {
    var vocab = ByteStringTable(minimumCapacity: 1)
    /// Merge strings back to back: merge i is (2i, 2i + 1).
    var mergeStorage: [UInt8] = []
    var mergeOffsets: [Int32] = [0]
    var addedTokens: [AddedToken] = []
    var prefixIDs: [Int32] = []
    var suffixIDs: [Int32] = []
    var unknownToken: String?

    var mergeCount: Int { (mergeOffsets.count - 1) / 2 }

    func mergeRange(_ string: Int) -> Range<Int> {
        Int(mergeOffsets[string])..<Int(mergeOffsets[string + 1])
    }

    static func parse(_ raw: UnsafeRawBufferPointer) throws -> TokenizerFile {
        var file = TokenizerFile()
        var reader = JSONReader(raw)
        var sawModel = false
        var postProcessor: JSONValue = .null
        do {
            try reader.expect(UInt8(ascii: "{"))
            if !reader.consume(UInt8(ascii: "}")) {
                repeat {
                    reader.skipWhitespace()
                    let key = try reader.readString()
                    try reader.expect(UInt8(ascii: ":"))
                    switch key {
                    case "model":
                        try file.readModel(&reader)
                        sawModel = true
                    case "added_tokens":
                        try file.readAddedTokens(try reader.parseValue())
                    case "normalizer":
                        try checkNormalizer(try reader.parseValue())
                    case "pre_tokenizer":
                        try checkPreTokenizer(try reader.parseValue())
                    case "post_processor":
                        postProcessor = try reader.parseValue()
                    case "truncation", "padding":
                        guard try reader.parseValue().isNull else {
                            throw TokenizerError.unsupported("\(key) must be null")
                        }
                    default:
                        try reader.skipValue()  // version, decoder
                    }
                } while try reader.nextElement(close: UInt8(ascii: "}"))
            }
        } catch let error as JSONParseError {
            throw TokenizerError.malformed(error.description)
        }
        guard sawModel else { throw TokenizerError.malformed("no model section") }
        try file.readPostProcessor(postProcessor)
        return file
    }

    // MARK: Sections

    private mutating func readModel(_ reader: inout JSONReader) throws {
        try reader.expect(UInt8(ascii: "{"))
        if reader.consume(UInt8(ascii: "}")) { throw TokenizerError.malformed("empty model") }
        repeat {
            reader.skipWhitespace()
            let key = try reader.readString()
            try reader.expect(UInt8(ascii: ":"))
            switch key {
            case "type":
                guard try reader.parseValue().stringValue == "BPE" else {
                    throw TokenizerError.unsupported("model.type must be BPE")
                }
            case "byte_fallback":
                guard try reader.parseValue().boolValue == true else {
                    throw TokenizerError.unsupported("model.byte_fallback must be true")
                }
            case "ignore_merges":
                guard try reader.parseValue().boolValue != true else {
                    throw TokenizerError.unsupported("model.ignore_merges is not implemented")
                }
            case "dropout", "continuing_subword_prefix", "end_of_word_suffix":
                let value = try reader.parseValue()
                guard value.isNull || (key == "dropout" && value.doubleValue == 0) else {
                    throw TokenizerError.unsupported("model.\(key) must be null")
                }
            case "unk_token":
                unknownToken = try reader.parseValue().stringValue
            case "vocab":
                try readVocab(&reader)
            case "merges":
                try readMerges(&reader)
            default:
                try reader.skipValue()  // fuse_unk: consecutive unknowns are always fused
            }
        } while try reader.nextElement(close: UInt8(ascii: "}"))
        guard vocab.count > 0 else { throw TokenizerError.malformed("empty vocab") }
    }

    private mutating func readVocab(_ reader: inout JSONReader) throws {
        vocab = ByteStringTable(minimumCapacity: 300_000)
        var key: [UInt8] = []
        try reader.expect(UInt8(ascii: "{"))
        if reader.consume(UInt8(ascii: "}")) { return }
        repeat {
            reader.skipWhitespace()
            key.removeAll(keepingCapacity: true)
            try reader.readString(into: &key)
            try reader.expect(UInt8(ascii: ":"))
            let id = try reader.readNumber()
            guard let id32 = Int32(exactly: id), id32 >= 0 else { throw TokenizerError.malformed("vocab id \(id)") }
            let inserted = key.withUnsafeBytes { vocab.insert($0, value: id32) }
            guard inserted else {
                throw TokenizerError.malformed("duplicate vocab key \(String(decoding: key, as: UTF8.self))")
            }
        } while try reader.nextElement(close: UInt8(ascii: "}"))
    }

    /// Merges as [["a", "b"], ...] (tokenizers >= 0.20) or ["a b", ...] (older files).
    private mutating func readMerges(_ reader: inout JSONReader) throws {
        mergeStorage.reserveCapacity(8 << 20)
        mergeOffsets.reserveCapacity(1_100_000)
        try reader.expect(UInt8(ascii: "["))
        if reader.consume(UInt8(ascii: "]")) { return }
        var pair: [UInt8] = []
        repeat {
            reader.skipWhitespace()
            if reader.peek() == UInt8(ascii: "[") {
                try reader.expect(UInt8(ascii: "["))
                reader.skipWhitespace()
                try reader.readString(into: &mergeStorage)
                mergeOffsets.append(Int32(mergeStorage.count))
                try reader.expect(UInt8(ascii: ","))
                reader.skipWhitespace()
                try reader.readString(into: &mergeStorage)
                mergeOffsets.append(Int32(mergeStorage.count))
                try reader.expect(UInt8(ascii: "]"))
            } else {
                pair.removeAll(keepingCapacity: true)
                try reader.readString(into: &pair)
                guard let space = pair.firstIndex(of: 0x20) else {
                    throw TokenizerError.malformed("merge without a space")
                }
                mergeStorage.append(contentsOf: pair[..<space])
                mergeOffsets.append(Int32(mergeStorage.count))
                mergeStorage.append(contentsOf: pair[(space + 1)...])
                mergeOffsets.append(Int32(mergeStorage.count))
            }
        } while try reader.nextElement(close: UInt8(ascii: "]"))
    }

    private mutating func readAddedTokens(_ value: JSONValue) throws {
        guard let list = value.arrayValue else { throw TokenizerError.malformed("added_tokens is not a list") }
        for item in list {
            guard let content = item["content"]?.stringValue, let id = item["id"]?.intValue,
                let id32 = Int32(exactly: id), id32 >= 0
            else { throw TokenizerError.malformed("added token without content or a valid id") }
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] where item[flag]?.boolValue == true {
                throw TokenizerError.unsupported("added token \(content) with \(flag) = true")
            }
            addedTokens.append(AddedToken(content: Array(content.utf8), id: id32))
        }
    }

    private static func checkNormalizer(_ value: JSONValue) throws {
        guard value["type"]?.stringValue == "Replace", value["pattern"]?["String"]?.stringValue == " ",
            value["content"]?.stringValue == "\u{2581}"
        else { throw TokenizerError.unsupported("normalizer must be Replace \" \" -> \"\u{2581}\"") }
    }

    /// Split on " " is a no-op after the normalizer has replaced every " ".
    private static func checkPreTokenizer(_ value: JSONValue) throws {
        if value.isNull { return }
        guard value["type"]?.stringValue == "Split", value["pattern"]?["String"]?.stringValue == " ",
            value["invert"]?.boolValue != true
        else { throw TokenizerError.unsupported("pre_tokenizer must be null or Split on \" \"") }
    }

    /// TemplateProcessing "single": special tokens before and after the sequence.
    private mutating func readPostProcessor(_ value: JSONValue) throws {
        if value.isNull { return }
        guard value["type"]?.stringValue == "TemplateProcessing", let single = value["single"]?.arrayValue else {
            throw TokenizerError.unsupported("post_processor must be TemplateProcessing")
        }
        var seenSequence = false
        for piece in single {
            if piece["Sequence"] != nil {
                guard !seenSequence else { throw TokenizerError.unsupported("template with two sequences") }
                seenSequence = true
            } else if let name = piece["SpecialToken"]?["id"]?.stringValue {
                guard let ids = value["special_tokens"]?[name]?["ids"]?.arrayValue else {
                    throw TokenizerError.malformed("template token \(name) has no ids")
                }
                let converted = try ids.map { id -> Int32 in
                    guard let int = id.intValue, let id32 = Int32(exactly: int), id32 >= 0 else {
                        throw TokenizerError.malformed("template id")
                    }
                    return id32
                }
                if seenSequence { suffixIDs += converted } else { prefixIDs += converted }
            } else {
                throw TokenizerError.unsupported("template piece \(piece.jsonString())")
            }
        }
    }
}
