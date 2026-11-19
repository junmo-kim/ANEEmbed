#if canImport(CoreAI)
import CoreAI
import CoreGraphics
import Foundation
import ImageIO

struct VisualFeatures { let rows: [[Float]], columns: Int }
@available(macOS 27.0, *)
struct Vision {
  let positions: Tensor?
  let channels: Int, rotaryQuarter: Int
  init(_ weights: Weights?, elephant: Bool = false) throws {
    channels = elephant ? 1152 : 1024
    rotaryQuarter = elephant ? 18 : 16
    if elephant {
      positions = nil
      return
    }
    guard let weights else { throw DecisionError("Missing vision weights") }
    positions = try weights.get((elephant ? "model." : "backbone.") + "visual.pos_embed.weight")
    guard positions?.shape == [2304, channels] else {
      throw DecisionError("Unsupported vision position grid")
    }
  }
  func encode(_ data: Data, using backbone: Backbone) async throws -> VisualFeatures {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = props[kCGImagePropertyPixelWidth] as? Int,
      let height = props[kCGImagePropertyPixelHeight] as? Int,
      width > 0, height > 0, Double(max(width, height)) / Double(min(width, height)) <= 200,
      let asset = backbone.manifest.visionAsset
    else { throw DecisionError("Unsupported image", input: true) }
    let area = Double(width) * Double(height)
    var w = max(32, Int((Double(width) / 32).rounded(.toNearestOrEven)) * 32)
    var h = max(32, Int((Double(height) / 32).rounded(.toNearestOrEven)) * 32)
    if Double(w) * Double(h) > 262144 {
      if backbone.manifest.elephant {
        throw DecisionError("Image exceeds bundle patch capacity", input: true)
      }
      let scale = sqrt(262144 / area)
      w = max(32, Int(floor(Double(width) * scale / 32)) * 32)
      h = max(32, Int(floor(Double(height) * scale / 32)) * 32)
    } else if w * h < 65536 {
      let scale = sqrt(65536 / area)
      w = max(32, Int(ceil(Double(width) * scale / 32)) * 32)
      h = max(32, Int(ceil(Double(height) * scale / 32)) * 32)
    }
    let gw = w / 16
    let gh = h / 16
    let patches = gw * gh
    let capacity = 1024
    guard patches <= capacity,
      let image = CGImageSourceCreateThumbnailAtIndex(
        source, 0,
        [
          kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceThumbnailMaxPixelSize: max(w, h),
        ] as CFDictionary)
    else { throw DecisionError("Image exceeds the exported patch budget", input: true) }
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let color = CGColorSpace(name: CGColorSpace.sRGB) else {
      throw DecisionError("Missing sRGB color space")
    }
    try rgba.withUnsafeMutableBytes { buffer in
      guard
        let ctx = CGContext(
          data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
          space: color, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
      else {
        throw DecisionError("Image resize failed")
      }
      ctx.interpolationQuality = .high
      ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    var pixels = NDArray.zeros([1, 1536, 1, capacity])
    var pos = NDArray.zeros([1, channels, 1, capacity])
    var cos = NDArray.zeros([1, 1, rotaryQuarter * 2, capacity])
    var sin = cos
    var mask = NDArray.zeros([1, 1, 1, capacity])
    mask.write { p, s in for i in patches..<capacity { p[i * s[3]] = -30000 } }
    for i in 0..<patches {
      let group = i / 4
      let row = (group / (gw / 2)) * 2 + (i % 4) / 2
      let col = (group % (gw / 2)) * 2 + i % 2
      pixels.write { p, s in
        for c in 0..<3 {
          for y in 0..<16 {
            for x in 0..<16 {
              let channel = c * 512 + y * 16 + x
              let value = Float16(
                Float(rgba[((row * 16 + y) * w + col * 16 + x) * 4 + c]) / 127.5 - 1)
              p[channel * s[1] + i * s[3]] = value
              p[(channel + 256) * s[1] + i * s[3]] = value
            }
          }
        }
      }
      if let positions {
        let y = Float(row) * 47 / Float(gh - 1)
        let x = Float(col) * 47 / Float(gw - 1)
        let y0 = Int(y)
        let x0 = Int(x)
        let fy = y - Float(y0)
        let fx = x - Float(x0)
        let coords = [
          (y0, x0, (1 - fy) * (1 - fx)), (y0, min(x0 + 1, 47), (1 - fy) * fx),
          (min(y0 + 1, 47), x0, fy * (1 - fx)), (min(y0 + 1, 47), min(x0 + 1, 47), fy * fx),
        ]
        var vector = [Float](repeating: 0, count: channels)
        for (r, c, weight) in coords {
          let start = (r * 48 + c) * channels
          let v = positions.floats(start..<start + channels)
          for d in v.indices { vector[d] += v[d] * weight }
        }
        pos.write { p, s in for d in vector.indices { p[d * s[1] + i * s[3]] = Float16(vector[d]) }
        }
      }
      cos.write { pc, sc in
        sin.write { ps, ss in
          for d in 0..<(rotaryQuarter * 2) {
            if let lookup = backbone.rotary["vision"] {
              let index = d < rotaryQuarter ? row : col
              let offset = (index * rotaryQuarter + d % rotaryQuarter) * 2
              pc[d * sc[2] + i * sc[3]] = Float16(bitPattern: lookup[offset])
              ps[d * ss[2] + i * ss[3]] = Float16(bitPattern: lookup[offset + 1])
              continue
            }
            let angle =
              Float(d < rotaryQuarter ? row : col)
              * powf(10000, -Float(d % rotaryQuarter) / Float(rotaryQuarter))
            pc[d * sc[2] + i * sc[3]] = Float16(cosf(angle))
            ps[d * ss[2] + i * ss[3]] = Float16(sinf(angle))
          }
        }
      }
    }
    if let asset = backbone.manifest.visionPosition {
      let project = try await backbone.function("visionpos", in: asset.asset)
      guard let name = backbone.manifest.visionFractions else {
        throw DecisionError("Missing vision fraction table")
      }
      let data = try Data(contentsOf: backbone.root.bundleChild(name))
      guard data.count == 511 * 512, (2...512).contains(gh), (2...512).contains(gw) else {
        throw DecisionError("Invalid vision fraction table or grid")
      }
      let fractions = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
      func fraction(_ remainder: Int, _ denominator: Int) -> Float16 {
        let i = denominator * (denominator - 1) / 2 + remainder
        return Float16(bitPattern: UInt16(littleEndian: fractions[i]))
      }
      for start in stride(from: 0, to: patches, by: 16) {
        var inputs = [String: NDArray]()
        for key in ["row", "column", "fy", "fx"] {
          var arr = NDArray.zeros([1, 1, 1, 16])
          arr.write { a, s in
            for t in 0..<16 {
              let i = min(start + t, patches - 1)
              let group = i / 4
              let row = (group / (gw / 2)) * 2 + (i % 4) / 2
              let col = (group % (gw / 2)) * 2 + i % 2
              let value: Float16
              switch key {
              case "row": value = Float16(row * 47 / (gh - 1))
              case "column": value = Float16(col * 47 / (gw - 1))
              case "fy": value = fraction(row * 47 % (gh - 1), gh - 1)
              default: value = fraction(col * 47 % (gw - 1), gw - 1)
              }
              a[t * s[3]] = value
            }
          }
          inputs[key] = arr
        }
        var result = try await project.run(inputs: inputs)
        let lanes = try result.tensor("pos")
        guard lanes.shape == [1, channels, 1, 16] else {
          throw DecisionError("Vision position lane shape")
        }
        lanes.read { a, sa in
          pos.write { b, sb in
            for t in 0..<min(16, patches - start) {
              for c in 0..<channels {
                b[c * sb[1] + (start + t) * sb[3]] = a[c * sa[1] + t * sa[3]]
              }
            }
          }
        }
      }
    }
    let f = try await backbone.function("p1024", in: asset)
    var out = try await f.run(inputs: [
      "pixel_values": pixels, "cos": cos, "sin": sin, "attn_mask": mask, "pos_embeds": pos,
    ])
    var hidden = try out.tensor("features")
    if let scale = backbone.manifest.featureScale {
      let f = try await backbone.function("featurescale", in: scale.asset)
      var result = try await f.run(inputs: ["features": hidden])
      hidden = try result.tensor("scaled")
    }
    let hsize = backbone.manifest.hidden
    guard hidden.shape == [1, hsize, 1, capacity / 4] else {
      throw DecisionError("Invalid visual features")
    }
    return VisualFeatures(
      rows: hidden.read { p, s in
        (0..<patches / 4).map { i in
          (0..<hsize).map { c in Float(p[c * s[1] + i * s[3]]) }
        }
      }, columns: gw / 2)
  }
}
#endif
