public enum ModelID: String, CaseIterable, Sendable {
    case embeddingGemma2 = "embeddinggemma-2"
    case pplx = "pplx-embed-v1-0.6b"
    case qwen3Embedding = "qwen3-embedding-0.6b"

    public var supportsImages: Bool { self == .embeddingGemma2 }

    public var source: ModelSource {
        switch self {
        case .embeddingGemma2: return .pinned
        case .qwen3Embedding:
            return ModelSource(repository: "junmo-k/qwen3-embedding-0.6b-ane", revision: "e3f5851431a491dd7a76436b82df048f798b2744",
                               manifestSHA256: "69c6571f9a726803f0d0b6edb055d372776e521d44e7b37d2c87ff89c55e4da8")
        case .pplx:
            return ModelSource(repository: "junmo-k/pplx-embed-v1-0.6b-ane", revision: "f07ba85d35ae742b0d9fb93bcbe0ec1384466b12",
                               manifestSHA256: "80a7f0e88dd74e6d7f552ebf68c7b27baaeb636a20bdab88e223242b84dd2017")
        }
    }

    public var tokenRange: ClosedRange<Int> {
        switch self {
        case .embeddingGemma2: return 2...8192
        case .pplx, .qwen3Embedding: return 1...32768
        }
    }

    public static var catalog: String {
        """
        embeddinggemma-2 downloads about 1 GB. First install took about 4–5 minutes on M4.
        pplx-embed-v1-0.6b downloads about 2.4 GB. The first local install at 2,048 tokens took about 1.5 minutes on M4.
        qwen3-embedding-0.6b returns normalized text embeddings (1024 dimensions by default, any width from 32 to 1024).
        Install it with anerun models install --model qwen3-embedding-0.6b.
        """
    }
}
