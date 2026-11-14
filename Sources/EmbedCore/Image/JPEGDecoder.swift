// JPEG decoder that reproduces libjpeg-turbo's default output, i.e. what PIL (and so Hugging Face) decodes:
// 8-bit Huffman JPEG (baseline, extended sequential, progressive), ISLOW integer IDCT (jidctint.c), fancy
// "triangle" chroma upsampling with replicated edges (jdsample.c, jdmainct.c) and the integer YCbCr -> RGB tables
// (jdcolor.c). ImageIO decodes the same files with different chroma upsampling (differences up to ~50 levels at
// block edges), which moves embeddings by up to ~0.02 cosine, so producer JPEGs are decoded here.
// Returns nil for anything else (arithmetic coding, 12-bit, lossless, CMYK / YCCK, odd sampling, corrupt data);
// the caller then falls back to ImageIO.
import Foundation

public enum JPEGDecoder {
    struct Component {
        let id: Int
        let h: Int
        let v: Int
        let quantTable: Int
        var quant = [Int32](repeating: 0, count: 64)  // natural order, latched at the component's first scan
        var latched = false
        var width = 0  // downsampled size in samples
        var height = 0
        var blocksPerLine = 0  // MCU-padded block grid
        var blocksPerColumn = 0
        var coefficients: [Int16] = []
    }

    struct Huffman {
        var maxcode = [Int32](repeating: -1, count: 18)
        var valptr = [Int32](repeating: 0, count: 17)
        var mincode = [Int32](repeating: 0, count: 17)
        var values: [UInt8] = []

        init?(counts: [Int], values: [UInt8]) {
            guard counts.reduce(0, +) == values.count, values.count <= 256 else { return nil }
            self.values = values
            var code: Int32 = 0, k: Int32 = 0
            for length in 1...16 {
                let n = Int32(counts[length - 1])
                if n > 0 {
                    valptr[length] = k
                    mincode[length] = code
                    code += n
                    k += n
                    maxcode[length] = code - 1
                } else {
                    maxcode[length] = -1
                }
                guard code <= (1 << length) else { return nil }
                code <<= 1
            }
            maxcode[17] = Int32.max
        }
    }

    struct DecodeError: Error {}

    /// zigzag index -> natural index, with 16 extra entries so corrupt run lengths stay in bounds (jpeg_natural_order)
    static let naturalOrder: [Int] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
        28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
        54, 47, 55, 62, 63,
    ] + [Int](repeating: 63, count: 16)

    // MARK: - entropy-coded segment reader

    struct BitReader {
        let bytes: UnsafeBufferPointer<UInt8>
        var position: Int
        var buffer: UInt64 = 0
        var count = 0
        var hitMarker = false

        mutating func fill() {
            while count <= 56 {
                var byte: UInt8 = 0
                if !hitMarker && position < bytes.count {
                    let b = bytes[position]
                    if b == 0xFF {
                        let next = position + 1 < bytes.count ? bytes[position + 1] : 0xD9
                        if next == 0x00 {
                            position += 2
                            byte = 0xFF
                        } else {
                            hitMarker = true  // zeros from here on, like libjpeg
                        }
                    } else {
                        position += 1
                        byte = b
                    }
                }
                buffer |= UInt64(byte) << UInt64(56 - count)
                count += 8
            }
        }

        /// The next `n` bits, 0 <= n <= 16 (the longest code or magnitude an 8-bit JPEG has). Anything else comes
        /// from a corrupt table and is an error: `n` past the 64-bit buffer would trap.
        mutating func bits(_ n: Int) throws -> Int {
            if n == 0 { return 0 }
            guard n > 0, n <= 16 else { throw DecodeError() }
            if count < n { fill() }
            let value = Int(buffer >> UInt64(64 - n))
            buffer <<= UInt64(n)
            count -= n
            return value
        }

        mutating func bit() -> Int {
            if count < 1 { fill() }
            let value = Int(buffer >> 63)
            buffer <<= 1
            count -= 1
            return value
        }

        mutating func decode(_ table: Huffman) throws -> Int {
            if count < 16 { fill() }
            var code = Int32(buffer >> 63)
            var length = 1
            while code > table.maxcode[length] {
                code = (code << 1) | Int32((buffer >> UInt64(63 - length)) & 1)
                length += 1
                if length > 16 { throw DecodeError() }
            }
            buffer <<= UInt64(length)
            count -= length
            let index = Int(table.valptr[length] + code - table.mincode[length])
            guard index >= 0, index < table.values.count else { throw DecodeError() }
            return Int(table.values[index])
        }

        static func extend(_ value: Int, _ size: Int) throws -> Int {
            guard size >= 0, size <= 16 else { throw DecodeError() }
            return size == 0 ? 0 : (value < (1 << (size - 1)) ? value - (1 << size) + 1 : value)
        }

        mutating func received(_ size: Int) throws -> Int { try Self.extend(try bits(size), size) }

        /// Discard buffered bits and consume the RSTn marker that must follow.
        mutating func restart() throws {
            buffer = 0
            count = 0
            hitMarker = false
            while position + 1 < bytes.count {
                if bytes[position] == 0xFF {
                    let m = bytes[position + 1]
                    if m >= 0xD0 && m <= 0xD7 {
                        position += 2
                        return
                    }
                    if m == 0xFF {
                        position += 1
                        continue
                    }
                    throw DecodeError()
                }
                position += 1
            }
            throw DecodeError()
        }
    }

    // MARK: - decode

    public static func decode(_ data: Data, maxPixels: Int) -> ([UInt8], Int, Int)? {
        data.withUnsafeBytes { raw -> ([UInt8], Int, Int)? in
            try? decode(raw.bindMemory(to: UInt8.self), maxPixels: maxPixels)
        }
    }

    private static func decode(_ bytes: UnsafeBufferPointer<UInt8>, maxPixels: Int) throws -> ([UInt8], Int, Int)? {
        guard bytes.count > 4, bytes[0] == 0xFF, bytes[1] == 0xD8 else { return nil }
        var quant = [[Int32]?](repeating: nil, count: 4)
        var dcTables = [Huffman?](repeating: nil, count: 4)
        var acTables = [Huffman?](repeating: nil, count: 4)
        var components: [Component] = []
        var width = 0, height = 0, progressive = false
        var restartInterval = 0
        var sawJFIF = false, adobeTransform: Int? = nil
        var hmax = 1, vmax = 1, mcusX = 0, mcusY = 0
        var position = 2

        func u16(_ p: Int) -> Int { Int(bytes[p]) << 8 | Int(bytes[p + 1]) }

        while true {
            // next marker (skip fill bytes)
            while position < bytes.count && bytes[position] != 0xFF { position += 1 }
            while position < bytes.count && bytes[position] == 0xFF { position += 1 }
            guard position < bytes.count else { break }
            let marker = bytes[position]
            position += 1
            if marker == 0xD9 { break }
            if (0xD0...0xD7).contains(marker) || marker == 0x01 { continue }
            guard position + 2 <= bytes.count else { throw DecodeError() }
            let length = u16(position)
            guard length >= 2, position + length <= bytes.count else { throw DecodeError() }
            let start = position + 2, end = position + length
            position = end
            switch marker {
            case 0xC0, 0xC1, 0xC2:  // SOF0 baseline, SOF1 extended, SOF2 progressive (Huffman)
                guard components.isEmpty, end - start >= 6, bytes[start] == 8 else { return nil }
                progressive = marker == 0xC2
                height = u16(start + 1)
                width = u16(start + 3)
                let n = Int(bytes[start + 5])
                guard width > 0, height > 0, width * height <= maxPixels, n == 1 || n == 3,
                    end - start >= 6 + 3 * n
                else { return nil }
                for i in 0..<n {
                    let p = start + 6 + 3 * i
                    let h = Int(bytes[p + 1] >> 4), v = Int(bytes[p + 1] & 15), q = Int(bytes[p + 2])
                    guard (1...4).contains(h), (1...4).contains(v), q < 4 else { return nil }
                    components.append(Component(id: Int(bytes[p]), h: h, v: v, quantTable: q))
                }
                hmax = components.map(\.h).max()!
                vmax = components.map(\.v).max()!
                mcusX = (width + 8 * hmax - 1) / (8 * hmax)
                mcusY = (height + 8 * vmax - 1) / (8 * vmax)
                for i in components.indices {
                    var c = components[i]
                    guard hmax % c.h == 0, vmax % c.v == 0 else { return nil }
                    c.width = (width * c.h + hmax - 1) / hmax
                    c.height = (height * c.v + vmax - 1) / vmax
                    c.blocksPerLine = mcusX * c.h
                    c.blocksPerColumn = mcusY * c.v
                    c.coefficients = [Int16](repeating: 0, count: c.blocksPerLine * c.blocksPerColumn * 64)
                    components[i] = c
                }
            case 0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF:
                return nil  // lossless, hierarchical, arithmetic
            case 0xC4:  // DHT
                var p = start
                while p < end {
                    guard p + 17 <= end else { throw DecodeError() }
                    let info = Int(bytes[p])
                    let counts = (0..<16).map { Int(bytes[p + 1 + $0]) }
                    let total = counts.reduce(0, +)
                    guard p + 17 + total <= end, info & 15 < 4, info >> 4 <= 1 else { throw DecodeError() }
                    let values = Array(bytes[(p + 17)..<(p + 17 + total)])
                    // Symbols are bit counts: a DC difference has at most 11 bits and an AC coefficient (low nibble
                    // of run/size) at most 10 at 8-bit precision. Larger ones only come from corrupt or hostile files.
                    let valid = info >> 4 == 0 ? values.allSatisfy { $0 <= 11 } : values.allSatisfy { $0 & 15 <= 10 }
                    guard valid, let table = Huffman(counts: counts, values: values) else { throw DecodeError() }
                    if info >> 4 == 0 { dcTables[info & 15] = table } else { acTables[info & 15] = table }
                    p += 17 + total
                }
            case 0xDB:  // DQT
                var p = start
                while p < end {
                    let precision = Int(bytes[p] >> 4), id = Int(bytes[p] & 15)
                    guard id < 4, p + 1 + 64 * (precision + 1) <= end else { throw DecodeError() }
                    var table = [Int32](repeating: 0, count: 64)
                    for k in 0..<64 {
                        let value = precision == 0 ? Int32(bytes[p + 1 + k]) : Int32(u16(p + 1 + 2 * k))
                        table[naturalOrder[k]] = value
                    }
                    quant[id] = table
                    p += 1 + 64 * (precision + 1)
                }
            case 0xDD:  // DRI
                guard end - start >= 2 else { throw DecodeError() }
                restartInterval = u16(start)
            case 0xE0:  // APP0: JFIF
                if end - start >= 14, Array(bytes[start..<(start + 5)]) == [0x4A, 0x46, 0x49, 0x46, 0] { sawJFIF = true }
            case 0xEE:  // APP14: Adobe
                if end - start >= 12, Array(bytes[start..<(start + 5)]) == [0x41, 0x64, 0x6F, 0x62, 0x65] {
                    adobeTransform = Int(bytes[start + 11])
                }
            case 0xDA:  // SOS
                guard !components.isEmpty, end - start >= 1 else { throw DecodeError() }
                let n = Int(bytes[start])
                guard n >= 1, n <= components.count, end - start >= 1 + 2 * n + 3 else { throw DecodeError() }
                var scan: [(Int, Int, Int)] = []  // component index, dc table, ac table
                for i in 0..<n {
                    let id = Int(bytes[start + 1 + 2 * i]), tables = Int(bytes[start + 2 + 2 * i])
                    guard let index = components.firstIndex(where: { $0.id == id }) else { throw DecodeError() }
                    scan.append((index, tables >> 4 & 3, tables & 3))
                }
                let ss = Int(bytes[start + 1 + 2 * n]), se = Int(bytes[start + 2 + 2 * n])
                let ah = Int(bytes[start + 3 + 2 * n] >> 4), al = Int(bytes[start + 3 + 2 * n] & 15)
                if !progressive { guard ss == 0, se == 63, ah == 0, al == 0 else { throw DecodeError() } }
                guard ss <= se, se <= 63, al <= 13 else { throw DecodeError() }
                if progressive, ss > 0, n != 1 { throw DecodeError() }
                for (index, _, _) in scan where !components[index].latched {
                    guard let table = quant[components[index].quantTable] else { throw DecodeError() }
                    components[index].quant = table
                    components[index].latched = true
                }
                position = try decodeScan(bytes: bytes, position: end, components: &components, scan: scan,
                                          dcTables: dcTables, acTables: acTables, progressive: progressive,
                                          ss: ss, se: se, ah: ah, al: al, restartInterval: restartInterval,
                                          mcusX: mcusX, mcusY: mcusY)
            default:
                break
            }
        }
        guard !components.isEmpty, components.allSatisfy(\.latched) else { return nil }
        let rgbInput: Bool
        if components.count == 3 {
            if sawJFIF { rgbInput = false }
            else if let transform = adobeTransform { rgbInput = transform == 0 }
            else { rgbInput = components.map(\.id) == [82, 71, 66] }
        } else {
            rgbInput = false
        }
        return (output(components: components, width: width, height: height, hmax: hmax, vmax: vmax, rgbInput: rgbInput),
                width, height)
    }

    private static func decodeScan(bytes: UnsafeBufferPointer<UInt8>, position: Int, components: inout [Component],
                                   scan: [(Int, Int, Int)], dcTables: [Huffman?], acTables: [Huffman?],
                                   progressive: Bool, ss: Int, se: Int, ah: Int, al: Int, restartInterval: Int,
                                   mcusX: Int, mcusY: Int) throws -> Int {
        var reader = BitReader(bytes: bytes, position: position)
        var eobrun = 0
        var predictors = [Int](repeating: 0, count: components.count)  // last_dc_val, reset per scan
        let needDC = !progressive || ss == 0, needAC = !progressive || se > 0
        func table(_ tables: [Huffman?], _ i: Int) throws -> Huffman {
            guard let t = tables[i] else { throw DecodeError() }
            return t
        }
        let dc = try scan.map { needDC && ah == 0 ? try table(dcTables, $0.1) : nil }
        let ac = try scan.map { needAC && (!progressive || ss > 0) ? try table(acTables, $0.2) : nil }

        func decodeBlock(_ s: Int, _ offset: Int) throws {
            let ci = scan[s].0
            try components[ci].coefficients.withUnsafeMutableBufferPointer { coef in
                if !progressive {
                    let t = try reader.decode(dc[s]!)
                    let diff = try reader.received(t)
                    predictors[ci] += diff
                    coef[offset] = Int16(truncatingIfNeeded: predictors[ci])
                    var k = 1
                    let acTable = ac[s]!
                    while k < 64 {
                        let rs = try reader.decode(acTable)
                        let r = rs >> 4, size = rs & 15
                        if size != 0 {
                            k += r
                            coef[offset + naturalOrder[k]] = Int16(truncatingIfNeeded: try reader.received(size))
                        } else {
                            if r != 15 { break }
                            k += 15
                        }
                        k += 1
                    }
                } else if ss == 0 {
                    if ah == 0 {
                        let t = try reader.decode(dc[s]!)
                        predictors[ci] += try reader.received(t)
                        coef[offset] = Int16(truncatingIfNeeded: predictors[ci] << al)
                    } else if reader.bit() == 1 {
                        coef[offset] |= Int16(truncatingIfNeeded: 1 << al)
                    }
                } else if ah == 0 {
                    if eobrun > 0 {
                        eobrun -= 1
                        return
                    }
                    var k = ss
                    let acTable = ac[s]!
                    while k <= se {
                        let rs = try reader.decode(acTable)
                        let r = rs >> 4, size = rs & 15
                        if size != 0 {
                            k += r
                            coef[offset + naturalOrder[k]] = Int16(truncatingIfNeeded: try reader.received(size) << al)
                        } else if r == 15 {
                            k += 15
                        } else {
                            eobrun = 1 << r
                            if r != 0 { eobrun += try reader.bits(r) }
                            eobrun -= 1
                            break
                        }
                        k += 1
                    }
                } else {
                    let p1 = Int16(truncatingIfNeeded: 1 << al), m1 = Int16(truncatingIfNeeded: -1 << al)
                    var k = ss
                    let acTable = ac[s]!
                    if eobrun == 0 {
                        while k <= se {
                            let rs = try reader.decode(acTable)
                            var r = rs >> 4
                            var value: Int16 = 0
                            if rs & 15 != 0 {
                                value = reader.bit() == 1 ? p1 : m1
                            } else if r != 15 {
                                eobrun = 1 << r
                                if r != 0 { eobrun += try reader.bits(r) }
                                break
                            }
                            repeat {
                                let z = offset + naturalOrder[k]
                                if coef[z] != 0 {
                                    if reader.bit() == 1, coef[z] & p1 == 0 { coef[z] &+= coef[z] >= 0 ? p1 : m1 }
                                } else {
                                    if r == 0 { break }
                                    r -= 1
                                }
                                k += 1
                            } while k <= se
                            if value != 0 { coef[offset + naturalOrder[k]] = value }
                            k += 1
                        }
                    }
                    if eobrun > 0 {
                        while k <= se {
                            let z = offset + naturalOrder[k]
                            if coef[z] != 0, reader.bit() == 1, coef[z] & p1 == 0 { coef[z] &+= coef[z] >= 0 ? p1 : m1 }
                            k += 1
                        }
                        eobrun -= 1
                    }
                }
            }
        }

        func blockOffset(_ ci: Int, _ row: Int, _ column: Int) -> Int {
            (row * components[ci].blocksPerLine + column) * 64
        }

        var unitsLeft = restartInterval
        func restartIfNeeded() throws {
            guard restartInterval > 0 else { return }
            if unitsLeft == 0 {
                try reader.restart()
                for (ci, _, _) in scan { predictors[ci] = 0 }
                eobrun = 0
                unitsLeft = restartInterval
            }
            unitsLeft -= 1
        }

        if scan.count == 1 {
            // non-interleaved: the component's own block grid, not padded to MCUs
            let ci = scan[0].0
            let columns = (components[ci].width + 7) / 8, rows = (components[ci].height + 7) / 8
            for row in 0..<rows {
                for column in 0..<columns {
                    try restartIfNeeded()
                    try decodeBlock(0, blockOffset(ci, row, column))
                }
            }
        } else {
            // Only the sampling factors: a copy of the Component would share its coefficient array, and decodeBlock's
            // write into that array would then copy all of it, for every block.
            let factors = scan.map { (h: components[$0.0].h, v: components[$0.0].v) }
            for my in 0..<mcusY {
                for mx in 0..<mcusX {
                    try restartIfNeeded()
                    for s in scan.indices {
                        let c = factors[s]
                        for v in 0..<c.v {
                            for h in 0..<c.h {
                                try decodeBlock(s, blockOffset(scan[s].0, my * c.v + v, mx * c.h + h))
                            }
                        }
                    }
                }
            }
        }
        // resume marker parsing at the marker that ended the entropy-coded segment
        var p = reader.position
        while p + 1 < bytes.count && !(bytes[p] == 0xFF && bytes[p + 1] != 0x00 && !(0xD0...0xD7).contains(bytes[p + 1])) {
            p += 1
        }
        return p
    }

    // MARK: - IDCT, upsampling, colour conversion

    /// jpeg_idct_islow into `plane` (stride `stride`) at (x, y).
    static func idctIslow(_ coef: UnsafeBufferPointer<Int16>, _ offset: Int, _ quant: [Int32],
                          _ plane: UnsafeMutableBufferPointer<UInt8>, _ stride: Int, _ x: Int, _ y: Int) {
        let constBits = 13, pass1Bits = 2
        func descale(_ v: Int, _ n: Int) -> Int { (v + (1 << (n - 1))) >> n }
        func limit(_ v: Int) -> UInt8 {
            let i = v & 1023
            let s = i < 512 ? i : i - 1024
            return UInt8(min(max(s + 128, 0), 255))
        }
        var ws = [Int](repeating: 0, count: 64)
        for c in 0..<8 {
            func dq(_ r: Int) -> Int { Int(coef[offset + r * 8 + c]) * Int(quant[r * 8 + c]) }
            if coef[offset + 8 + c] == 0 && coef[offset + 16 + c] == 0 && coef[offset + 24 + c] == 0
                && coef[offset + 32 + c] == 0 && coef[offset + 40 + c] == 0 && coef[offset + 48 + c] == 0
                && coef[offset + 56 + c] == 0
            {
                let dcval = dq(0) << pass1Bits
                for r in 0..<8 { ws[r * 8 + c] = dcval }
                continue
            }
            var z2 = dq(2), z3 = dq(6)
            var z1 = (z2 + z3) * 4433
            var tmp2 = z1 + z3 * -15137
            var tmp3 = z1 + z2 * 6270
            z2 = dq(0)
            z3 = dq(4)
            var tmp0 = (z2 + z3) << constBits
            var tmp1 = (z2 - z3) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = dq(7); tmp1 = dq(5); tmp2 = dq(3); tmp3 = dq(1)
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446; tmp1 *= 16819; tmp2 *= 25172; tmp3 *= 12299
            z1 *= -7373; z2 *= -20995; z3 *= -16069; z4 *= -3196
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits - pass1Bits
            ws[0 * 8 + c] = descale(tmp10 + tmp3, n); ws[7 * 8 + c] = descale(tmp10 - tmp3, n)
            ws[1 * 8 + c] = descale(tmp11 + tmp2, n); ws[6 * 8 + c] = descale(tmp11 - tmp2, n)
            ws[2 * 8 + c] = descale(tmp12 + tmp1, n); ws[5 * 8 + c] = descale(tmp12 - tmp1, n)
            ws[3 * 8 + c] = descale(tmp13 + tmp0, n); ws[4 * 8 + c] = descale(tmp13 - tmp0, n)
        }
        for r in 0..<8 {
            let w = r * 8, out = (y + r) * stride + x
            if ws[w + 1] == 0 && ws[w + 2] == 0 && ws[w + 3] == 0 && ws[w + 4] == 0 && ws[w + 5] == 0
                && ws[w + 6] == 0 && ws[w + 7] == 0
            {
                let dcval = limit(descale(ws[w], pass1Bits + 3))
                for k in 0..<8 { plane[out + k] = dcval }
                continue
            }
            var z2 = ws[w + 2], z3 = ws[w + 6]
            var z1 = (z2 + z3) * 4433
            var tmp2 = z1 + z3 * -15137
            var tmp3 = z1 + z2 * 6270
            var tmp0 = (ws[w] + ws[w + 4]) << constBits
            var tmp1 = (ws[w] - ws[w + 4]) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = ws[w + 7]; tmp1 = ws[w + 5]; tmp2 = ws[w + 3]; tmp3 = ws[w + 1]
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446; tmp1 *= 16819; tmp2 *= 25172; tmp3 *= 12299
            z1 *= -7373; z2 *= -20995; z3 *= -16069; z4 *= -3196
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits + pass1Bits + 3
            plane[out + 0] = limit(descale(tmp10 + tmp3, n)); plane[out + 7] = limit(descale(tmp10 - tmp3, n))
            plane[out + 1] = limit(descale(tmp11 + tmp2, n)); plane[out + 6] = limit(descale(tmp11 - tmp2, n))
            plane[out + 2] = limit(descale(tmp12 + tmp1, n)); plane[out + 5] = limit(descale(tmp12 - tmp1, n))
            plane[out + 3] = limit(descale(tmp13 + tmp0, n)); plane[out + 4] = limit(descale(tmp13 - tmp0, n))
        }
    }

    /// Component samples upsampled to width x height (jdsample.c with jdmainct.c's replicated edge rows).
    static func upsampled(_ c: Component, width: Int, height: Int, hmax: Int, vmax: Int) -> [UInt8] {
        let stride = c.blocksPerLine * 8
        var plane = [UInt8](repeating: 0, count: stride * c.blocksPerColumn * 8)
        c.coefficients.withUnsafeBufferPointer { coef in
            plane.withUnsafeMutableBufferPointer { out in
                for row in 0..<c.blocksPerColumn {
                    for column in 0..<c.blocksPerLine {
                        idctIslow(coef, (row * c.blocksPerLine + column) * 64, c.quant, out, stride, column * 8, row * 8)
                    }
                }
            }
        }
        let fh = hmax / c.h, fv = vmax / c.v
        if fh == 1 && fv == 1 {
            if stride == width { return Array(plane[0..<(width * height)]) }
            var out = [UInt8](repeating: 0, count: width * height)
            for y in 0..<height { for x in 0..<width { out[y * width + x] = plane[y * stride + x] } }
            return out
        }
        let w = c.width, h = c.height
        func at(_ x: Int, _ y: Int) -> Int { Int(plane[min(max(y, 0), h - 1) * stride + min(max(x, 0), w - 1)]) }
        var out = [UInt8](repeating: 0, count: width * height)
        if fh == 2 && fv == 1 && w > 2 {  // h2v1_fancy_upsample
            for y in 0..<height {
                for x in 0..<width {
                    let cx = x >> 1, v3 = at(cx, y) * 3
                    out[y * width + x] = UInt8(x & 1 == 0 ? (v3 + at(cx - 1, y) + 1) >> 2 : (v3 + at(cx + 1, y) + 2) >> 2)
                }
            }
        } else if fh == 1 && fv == 2 {  // h1v2_fancy_upsample
            for y in 0..<height {
                let cy = y >> 1
                for x in 0..<width {
                    let v3 = at(x, cy) * 3
                    out[y * width + x] = UInt8(y & 1 == 0 ? (v3 + at(x, cy - 1) + 1) >> 2 : (v3 + at(x, cy + 1) + 2) >> 2)
                }
            }
        } else if fh == 2 && fv == 2 && w > 2 {  // h2v2_fancy_upsample
            for y in 0..<height {
                let cy = y >> 1, ny = y & 1 == 0 ? cy - 1 : cy + 1
                for x in 0..<width {
                    let cx = x >> 1
                    let this = at(cx, cy) * 3 + at(cx, ny)
                    if x & 1 == 0 {
                        out[y * width + x] = UInt8((this * 3 + at(cx - 1, cy) * 3 + at(cx - 1, ny) + 8) >> 4)
                    } else {
                        out[y * width + x] = UInt8((this * 3 + at(cx + 1, cy) * 3 + at(cx + 1, ny) + 7) >> 4)
                    }
                }
            }
        } else {  // h2v1_upsample / h2v2_upsample / int_upsample: replication
            for y in 0..<height { for x in 0..<width { out[y * width + x] = plane[(y / fv) * stride + x / fh] } }
        }
        return out
    }

    static func output(components: [Component], width: Int, height: Int, hmax: Int, vmax: Int, rgbInput: Bool) -> [UInt8] {
        let planes = components.map { upsampled($0, width: width, height: height, hmax: hmax, vmax: vmax) }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        if planes.count == 1 {
            for i in 0..<(width * height) { rgb[i * 3] = planes[0][i]; rgb[i * 3 + 1] = planes[0][i]; rgb[i * 3 + 2] = planes[0][i] }
            return rgb
        }
        if rgbInput {
            for i in 0..<(width * height) { for c in 0..<3 { rgb[i * 3 + c] = planes[c][i] } }
            return rgb
        }
        // jdcolor.c build_ycc_rgb_table, SCALEBITS 16
        let half = 1 << 15
        var crR = [Int](repeating: 0, count: 256), cbB = crR, crG = crR, cbG = crR
        for i in 0..<256 {
            let x = i - 128
            crR[i] = (91881 * x + half) >> 16
            cbB[i] = (116130 * x + half) >> 16
            crG[i] = -46802 * x
            cbG[i] = -22554 * x + half
        }
        func clamp(_ v: Int) -> UInt8 { UInt8(min(max(v, 0), 255)) }
        for i in 0..<(width * height) {
            let y = Int(planes[0][i]), cb = Int(planes[1][i]), cr = Int(planes[2][i])
            rgb[i * 3] = clamp(y + crR[cr])
            rgb[i * 3 + 1] = clamp(y + ((cbG[cb] + crG[cr]) >> 16))
            rgb[i * 3 + 2] = clamp(y + cbB[cb])
        }
        return rgb
    }
}
