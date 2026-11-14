import Foundation
import Testing

@testable import EmbedCore

/// Hand-built JPEGs whose tables or entropy data are hostile. The decoder must reject them with an error
/// (nil from `decode`); a trap would take the whole server down.
enum CraftedJPEG {
    /// An 8 x 8 greyscale baseline JPEG with one-code Huffman tables: the DC code `0` decodes to `dcSymbol`,
    /// the AC code `0` to `acSymbol`. `entropy` is the scan data.
    static func make(dcSymbol: UInt8 = 0, acSymbol: UInt8 = 0x00, entropy: [UInt8] = [0x00], width: Int = 8,
                     height: Int = 8) -> Data
    {
        var bytes: [UInt8] = [0xFF, 0xD8]
        bytes += [0xFF, 0xDB, 0x00, 0x43, 0x00] + [UInt8](repeating: 1, count: 64)  // DQT 0
        bytes += [0xFF, 0xC0, 0x00, 0x0B, 0x08, UInt8(height >> 8), UInt8(height & 255), UInt8(width >> 8),
                  UInt8(width & 255), 0x01, 0x01, 0x11, 0x00]  // SOF0, one component
        let oneCode = [UInt8(1)] + [UInt8](repeating: 0, count: 15)
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x00] + oneCode + [dcSymbol]  // DHT DC 0
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x10] + oneCode + [acSymbol]  // DHT AC 0
        bytes += [0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]  // SOS
        bytes += entropy + [0xFF, 0xD9]
        return Data(bytes)
    }
}

@Suite("JPEG decoder robustness")
struct JPEGRobustnessTests {
    @Test func theCraftedFileDecodes() throws {
        let (rgb, width, height) = try #require(JPEGDecoder.decode(CraftedJPEG.make(), maxPixels: 100))
        #expect(width == 8 && height == 8 && rgb.count == 8 * 8 * 3)
    }

    /// A DC symbol is a bit count; 255, 64 or 63 used to shift past the 64-bit buffer and trap.
    @Test(arguments: [UInt8(12), 15, 16, 17, 63, 64, 65, 255])
    func rejectsDCSymbolsBeyondEightBitPrecision(_ symbol: UInt8) {
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: symbol), maxPixels: 100) == nil)
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: symbol, entropy: [0x00, 0x00, 0x00]), maxPixels: 100) == nil)
    }

    @Test(arguments: [UInt8(0x0B), 0x0F, 0x1B, 0xFF])
    func rejectsACSymbolsBeyondEightBitPrecision(_ symbol: UInt8) {
        #expect(JPEGDecoder.decode(CraftedJPEG.make(acSymbol: symbol), maxPixels: 100) == nil)
    }

    @Test func acceptsTheLargestValidSymbols() {
        // DC size 11 and AC size 10 are the 8-bit maxima; with an all-zero scan they decode to some image.
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: 11, acSymbol: 0x0A, entropy: [UInt8](repeating: 0, count: 64)),
                                   maxPixels: 100) != nil)
    }

    @Test func bitCountsOutsideZeroToSixteenThrow() throws {
        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 64)
        defer { bytes.deallocate() }
        bytes.initialize(repeating: 0xA5)
        var reader = JPEGDecoder.BitReader(bytes: UnsafeBufferPointer(bytes), position: 0)
        #expect(try reader.bits(0) == 0)
        #expect(try reader.bits(16) == 0xA5A5)
        for n in [-1, 17, 32, 63, 64, 65, 255] {
            #expect(throws: JPEGDecoder.DecodeError.self) { try reader.bits(n) }
            #expect(throws: JPEGDecoder.DecodeError.self) { try JPEGDecoder.BitReader.extend(1, n) }
        }
        #expect(try JPEGDecoder.BitReader.extend(0, 11) == -2047)
        #expect(try JPEGDecoder.BitReader.extend(0xFFFF, 16) == 0xFFFF)
    }

    @Test func imagePrepRejectsTheCrashFileWithAnError() {
        #expect(throws: ImageError.self) { try ImagePrep.prepare(CraftedJPEG.make(dcSymbol: 255)) }
    }
}
