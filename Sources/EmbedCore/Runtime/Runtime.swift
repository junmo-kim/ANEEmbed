// Model runtime interfaces. The service, CLI, HTTP and MCP layers only see `EmbeddingRuntime`, so the
// whole surface runs against `FakeRuntime` in tests and on machines without a model or a Neural Engine.
//
//   EmbeddingRuntime        texts (token ids) and images -> L2-normalized 768-d vectors
//   └─ BucketTextRuntime    packing planner + reusable buffers + a BucketFunctionRunner
//        ├─ CoreMLBucketRunner   embed_S and winp functions of the model package (Neural Engine)
//        └─ PoolingFakeRunner    deterministic stand-in that honours the bias/pool contract
//   LongTextRunner          2049...8192 tokens: LongTextEngine over CoreMLLongRunner, one winl call per text
//   VisionEncoder           prepared image -> soft-token rows for the text functions
//   SelfTestSuite           golden inputs from the package; a function that fails is not served
import Foundation

/// Static facts a runtime reports to the API (`/v1/models`, doctor).
public struct RuntimeInfo: Sendable, Equatable {
    public var maxTokens: Int
    /// 0 when images are not supported.
    public var maxImageTokens: Int
    public var bucketLengths: [Int]
    public var backend: String
    /// Model functions in use (empty for runtimes without named functions).
    public var functions: [String]

    public init(maxTokens: Int, maxImageTokens: Int, bucketLengths: [Int], backend: String, functions: [String] = []) {
        self.maxTokens = maxTokens
        self.maxImageTokens = maxImageTokens
        self.bucketLengths = bucketLengths
        self.backend = backend
        self.functions = functions
    }
}

public protocol EmbeddingRuntime: Sendable {
    var info: RuntimeInfo { get }
    /// One vector per text, in order. Every text is token ids with <bos>/<eos>, at most `info.maxTokens`.
    func embed(tokenized texts: [[Int32]]) async throws -> [[Float]]
    /// One vector per image, in order.
    func embed(images: [PreparedImage]) async throws -> [[Float]]
    /// Loads (and on first use compiles) every function this runtime will call.
    func warm() async throws
}

/// Text to token ids. `GemmaTokenizer` is the real one; tests and the fake runtime may use others.
public protocol TextTokenizer: Sendable {
    func encode(_ text: String) -> [Int32]
}

extension GemmaTokenizer: TextTokenizer {
    public func encode(_ text: String) -> [Int32] { encode(text, addSpecialTokens: true) }
}

/// Source of scaled fp16 token rows.
public protocol TokenRowSource: Sendable {
    func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>)
}

extension TokenTable: TokenRowSource {}

/// Runs one prepared bucket call and returns `slots` rows of 768 floats (rows of unused slots are
/// undefined). Implementations must be safe to call concurrently with different buffers.
public protocol BucketFunctionRunner: Sendable {
    var lengths: [Int] { get }
    func run(_ buffers: BucketBuffers) async throws -> [[Float]]
    func warm() async throws
}

/// Texts longer than the largest packing function (2049...8192 tokens).
public protocol LongTextRunner: Sendable {
    var maxTokens: Int { get }
    func embed(ids: [Int32]) async throws -> [Float]
    func warm() async throws
}

/// A prepared canvas -> its soft-token rows (tokens x 512 fp16, unscaled), which the text functions
/// then embed as `<bos> <|image> rows <image|> <eos>`.
public protocol VisionEncoder: Sendable {
    var maxTokens: Int { get }
    func encode(_ image: PreparedImage) async throws -> [Float16]
    func warm() async throws
}

public enum RuntimeError: Error, CustomStringConvertible, Sendable {
    case tooLong(tokens: Int, max: Int)
    case unsupported(String)
    case model(String)

    public var description: String {
        switch self {
        case .tooLong(let tokens, let max): return "\(tokens) tokens exceed the runtime limit of \(max)"
        case .unsupported(let message): return message
        case .model(let message): return message
        }
    }
}

/// Special token ids of EmbeddingGemma 2 used to wrap media rows.
public enum SpecialTokens {
    public static let bos: Int32 = 2
    public static let eos: Int32 = 1
    public static let pad: Int32 = 0
    public static let imageStart: Int32 = 255_999  // <|image>
    public static let imageEnd: Int32 = 258_882  // <image|>
}
