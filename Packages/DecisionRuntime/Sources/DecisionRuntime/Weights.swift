import Accelerate
import Foundation

/// Each view retains the mapped file. Only the requested embedding rows are faulted in.
struct Weights {
  struct Entry: Decodable {
    enum CodingKeys: String, CodingKey {
      case dtype
      case shape
      case dataOffsets = "data_offsets"
    }
    let dtype: String, shape: [Int], dataOffsets: [Int]
  }
  let file: Data, entries: [String: Entry], start: Int
  init(_ url: URL) throws {
    file = try Data(contentsOf: url, options: .alwaysMapped)
    guard file.count >= 8 else { throw DecisionError("Truncated tensor header") }
    let size = file.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    guard size <= min(100_000_000, file.count - 8) else {
      throw DecisionError("Invalid tensor header length")
    }
    start = 8 + Int(size)
    guard var header = try JSONSerialization.jsonObject(with: file[8..<start]) as? [String: Any]
    else {
      throw DecisionError("Invalid tensor header")
    }
    header.removeValue(forKey: "__metadata__")
    entries = try JSONDecoder().decode(
      [String: Entry].self, from: JSONSerialization.data(withJSONObject: header))
  }
  func get(_ name: String) throws -> Tensor {
    guard let e = entries[name], ["F16", "BF16", "F32"].contains(e.dtype),
      e.dataOffsets.count == 2,
      e.dataOffsets[0] >= 0, e.dataOffsets[1] >= e.dataOffsets[0],
      e.dataOffsets[1] <= file.count - start
    else { throw DecisionError("Invalid tensor: \(name)") }
    let stride = e.dtype == "F32" ? 4 : 2
    var count = stride
    for dim in e.shape {
      guard dim > 0, count <= Int.max / dim else { throw DecisionError("Invalid tensor shape") }
      count *= dim
    }
    guard e.dataOffsets[1] - e.dataOffsets[0] == count else {
      throw DecisionError("Tensor size mismatch")
    }
    return Tensor(
      file: file, offset: start + e.dataOffsets[0], dtype: e.dtype, shape: e.shape,
      count: count / stride)
  }
}
struct Tensor {
  let file: Data, offset: Int, dtype: String, shape: [Int], count: Int
  func floats(_ range: Range<Int>) -> [Float] {
    file.withUnsafeBytes { buf in
      let p = buf.baseAddress!.advanced(by: offset)
      switch dtype {
      case "F32": return range.map { p.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
      case "F16":
        return range.map { Float(p.loadUnaligned(fromByteOffset: $0 * 2, as: Float16.self)) }
      default:
        return range.map {
          Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) << 16)
        }
      }
    }
  }
}

struct Linear {
  let matrix: [Float], bias: [Float], input: Int
  init(_ w: Weights, _ name: String, input: Int, output: Int, hasBias: Bool = true) throws {
    let tensor = try w.get(name + "weight")
    guard tensor.shape == [output, input] else {
      throw DecisionError("Invalid projection dimensions in " + name)
    }
    self.input = input
    matrix = tensor.floats(0..<tensor.count)
    if hasBias {
      let b = try w.get(name + "bias")
      guard b.shape == [output] else { throw DecisionError("Invalid bias in " + name) }
      bias = b.floats(0..<output)
    } else {
      bias = Array(repeating: 0, count: output)
    }
  }
  func callAsFunction(_ x: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: bias.count)
    vDSP_mmul(matrix, 1, x, 1, &y, 1, vDSP_Length(y.count), 1, vDSP_Length(input))
    vDSP_vadd(y, 1, bias, 1, &y, 1, vDSP_Length(y.count))
    return y
  }
}
struct Norm {
  let weight: [Float], bias: [Float]
  init(_ w: Weights, _ name: String, _ size: Int) throws {
    let a = try w.get(name + ".weight")
    let b = try w.get(name + ".bias")
    guard a.shape == [size], b.shape == [size] else {
      throw DecisionError("Invalid layer norm in " + name)
    }
    weight = a.floats(0..<size)
    bias = b.floats(0..<size)
  }
  func callAsFunction(_ x: [Float]) -> [Float] {
    let mean = x.reduce(0, +) / Float(x.count)
    let centered = x.map { $0 - mean }
    let inverse = 1 / sqrt(centered.reduce(0) { $0 + $1 * $1 } / Float(x.count) + 1e-5)
    return x.indices.map { centered[$0] * inverse * weight[$0] + bias[$0] }
  }
}
struct FeedForward {
  let up: Linear, down: Linear
  init(_ w: Weights, _ name: String, _ size: Int, _ inner: Int) throws {
    up = try Linear(w, name + ".0.", input: size, output: inner)
    down = try Linear(w, name + ".2.", input: inner, output: size)
  }
  func callAsFunction(_ x: [Float]) -> [Float] {
    down(up(x).map { $0 * 0.5 * (1 + erff($0 * 0.70710678)) })
  }
}
struct MixerLayer {
  let tokenNorm: Norm, channelNorm: Norm, tokenMLP: FeedForward, channelMLP: FeedForward
  init(_ w: Weights, _ index: Int) throws {
    let p = "head.blocks.\(index)."
    tokenNorm = try Norm(w, p + "token_norm", 256)
    channelNorm = try Norm(w, p + "channel_norm", 256)
    tokenMLP = try FeedForward(w, p + "token_mlp", 4, 16)
    channelMLP = try FeedForward(w, p + "channel_mlp", 256, 1024)
  }
  func callAsFunction(_ rows: [[Float]]) -> [[Float]] {
    let n = rows.map { tokenNorm($0) }
    var mixed = rows
    for j in n[0].indices {
      let v = tokenMLP(n.map { $0[j] })
      for i in rows.indices { mixed[i][j] += v[i] }
    }
    return mixed.map { zip($0, channelMLP(channelNorm($0))).map(+) }
  }
}
struct Mixer {
  let norm: Norm, projection: Linear, roles: [Float], blocks: [MixerLayer], outputNorm: Norm,
    linear: Linear
  init(_ w: Weights, _ hidden: Int) throws {
    norm = try Norm(w, "head.input_norm", hidden)
    projection = try Linear(w, "head.projection.", input: hidden, output: 256)
    let t = try w.get("head.roles")
    guard t.shape == [1, 4, 256] else { throw DecisionError("Invalid Mixer role embeddings") }
    roles = t.floats(0..<t.count)
    blocks = try (0..<2).map { try MixerLayer(w, $0) }
    outputNorm = try Norm(w, "head.output.0", 256)
    linear = try Linear(w, "head.output.1.", input: 256, output: 1)
  }
  func scores(_ h: [[Float]], _ branch: Branch) -> [Float] {
    let pooled = branch.roles.map { span -> [Float] in
      var sum = Array(repeating: Float(0), count: h[0].count)
      for i in span { vDSP_vadd(sum, 1, h[i], 1, &sum, 1, vDSP_Length(sum.count)) }
      return projection(norm(sum.map { $0 / Float(span.count) }))
    }
    return pooled[2..<(pooled.count - 1)].map { candidate in
      var rows = [pooled[0], pooled[1], candidate, pooled.last!]
      for i in rows.indices { for j in rows[i].indices { rows[i][j] += roles[i * 256 + j] } }
      for layer in blocks { rows = layer(rows) }
      return linear(outputNorm(rows[2]))[0]
    }
  }
}
