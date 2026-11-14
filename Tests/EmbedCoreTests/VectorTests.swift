import Foundation
import Testing

@testable import EmbedCore

@Suite("Vectors")
struct VectorTests {
    @Test func truncationIsTheRenormalizedPrefix() {
        let vector = Vectors.l2Normalized((0..<768).map { Float(($0 * 37) % 101) - 50 })
        for dimensions in Vectors.dimensions {
            let truncated = Vectors.truncated(vector, to: dimensions)
            #expect(truncated.count == dimensions)
            let norm = truncated.reduce(0) { $0 + $1 * $1 }.squareRoot()
            #expect(abs(norm - 1) < 1e-5)
            #expect(Vectors.cosine(truncated, Array(vector.prefix(dimensions))) > 0.99999)
        }
        let full = Vectors.truncated(vector, to: 768)
        #expect(zip(full, vector).allSatisfy { abs($0 - $1) < 1e-7 })
    }

    /// The full size is normalized as well: a vector whose norm is off comes out with norm 1, same direction.
    @Test func theFullSizeIsNormalizedToo() {
        let vector = (0..<768).map { Float(($0 * 37) % 101) - 50 }.map { $0 / 30 }
        let full = Vectors.truncated(vector, to: 768)
        #expect(abs(full.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-5)
        #expect(Vectors.cosine(full, vector) > 0.99999)
    }

    @Test func base64IsLittleEndianFloat32() {
        let vector: [Float] = [1, -2.5, 0.1]
        let encoded = Vectors.base64(vector)
        #expect(encoded == "AACAPwAAIMDNzMw9")
        #expect(Vectors.decodeBase64(encoded) == vector)
        #expect(Vectors.decodeBase64("AAA") == nil)
    }

    /// Squares of tiny values underflow and of huge ones overflow in float32; scaling by the largest value first
    /// keeps both normalizable (a Matryoshka prefix can be tiny while the full vector is not).
    @Test func normalizesTinyAndHugeVectors() {
        func norm(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }
        let tiny = Vectors.l2Normalized([Float](repeating: 1e-23, count: 128))
        #expect(abs(norm(tiny) - 1) < 1e-5)
        let huge = Vectors.l2Normalized([3e30, 4e30])
        #expect(abs(huge[0] - 0.6) < 1e-6 && abs(huge[1] - 0.8) < 1e-6)
        let prefix = Vectors.truncated([Float](repeating: 1e-23, count: 128) + [Float](repeating: 1, count: 640), to: 128)
        #expect(abs(norm(prefix) - 1) < 1e-5)
        #expect(Vectors.l2Normalized([0, 0]) == [0, 0])
    }

    @Test func normalizationLeavesZeroVectorsAlone() {
        #expect(Vectors.l2Normalized([0, 0]) == [0, 0])
        #expect(Vectors.l2Normalized([3, 4]) == [0.6, 0.8])
    }

    @Test func inputTypesApplyTheModelPrefixes() {
        #expect(InputType.query.apply("x") == "task: search result | query: x")
        #expect(InputType.document.apply("x") == "title: none | text: x")
    }
}
