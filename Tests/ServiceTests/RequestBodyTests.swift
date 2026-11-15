import Foundation
import Testing

@testable import EmbedCore

/// A value several tasks write and the test reads.
final class Shared<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value

    init(_ value: Value) { stored = value }

    var value: Value {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }

    func update(_ change: (inout Value) -> Void) { lock.withLock { change(&stored) } }
}

/// The fake vision encoder, which first calls `observe` for every image it encodes.
struct ObservingVision: VisionEncoder {
    let maxTokens = ImagePrep.maxSoftTokens
    let observe: @Sendable () -> Void

    func warm() async throws {}

    func encode(_ image: PreparedImage) async throws -> [Float16] {
        observe()
        return try await FakeVisionEncoder().encode(image)
    }
}

enum BodyTestSupport {
    /// A router over the fake runtime whose vision encoder calls `observe` for every image.
    static func router(observe: @escaping @Sendable () -> Void) -> Router {
        let runtime = BucketTextRuntime(runner: PoolingFakeRunner(lengths: CoreMLBucketRunner.lengths(forMaxTokens: 2048)),
                                        rows: HashedTokenRows(), vision: ObservingVision(observe: observe), backend: "fake")
        let service = EmbeddingService(tokenizer: ByteTokenizer(), runtime: runtime, limits: ServiceLimits())
        return Router(state: ServerState(.ready(service)), modelName: "embeddinggemma-2")
    }

    /// An embeddings request body with one 8 x 8 PNG.
    static let imageRequest: Data = {
        let item: JSONValue = ["content": [["type": "image_url", "image_url": ["url": .string(TestSupport.dataURL(TestSupport.png(width: 8, height: 8)))]]]]
        return (["input": [item]] as JSONValue).jsonData()
    }()

    /// `bytes` in memory of their own, which calls `onFree` when the last reference to it goes.
    static func tracked(_ bytes: Data, onFree: @escaping @Sendable () -> Void) -> Data {
        let memory = UnsafeMutableRawPointer.allocate(byteCount: bytes.count, alignment: 16)
        bytes.copyBytes(to: memory.assumingMemoryBound(to: UInt8.self), count: bytes.count)
        return Data(bytesNoCopy: memory, count: bytes.count, deallocator: .custom { pointer, _ in
            pointer.deallocate()
            onFree()
        })
    }
}

@Suite("Request bodies")
struct RequestBodyTests {
    /// The raw body of an embeddings request is freed once parsed, before its images are embedded, rather than
    /// after the response.
    @Test func theEmbeddingsBodyIsFreedBeforeTheImagesAreEmbedded() async throws {
        let freed = Shared(false), seen = Shared<[Bool]>([])
        let router = BodyTestSupport.router { seen.update { $0.append(freed.value) } }
        let response = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings",
                                                       body: BodyTestSupport.tracked(BodyTestSupport.imageRequest) { freed.value = true }))
        #expect(response.status == 200, "\(String(decoding: response.body, as: UTF8.self))")
        #expect(seen.value == [true])
    }

    /// The reader allocates a body once, at its Content-Length, and hands exactly those bytes over.
    @Test func aBodyArrivingInPiecesIsHandedOverWhole() throws {
        var reader = HTTPRequestReader(maxBody: 1 << 20)
        let body = Data((0..<5000).map { UInt8(truncatingIfNeeded: $0) })
        reader.append(Data("POST /a HTTP/1.1\r\nContent-Length: 5000\r\n\r\n".utf8) + body.prefix(10))
        #expect(try reader.next() == nil)
        for start in stride(from: 10, to: 5000, by: 997) { reader.append(body[start..<min(start + 997, 5000)]) }
        reader.append(Data("GET /b HTTP/1.1\r\n\r\n".utf8))
        #expect(try reader.next()?.body == body)
        #expect(try reader.next()?.path == "/b" && reader.bufferedBytes == 0)
    }
}
