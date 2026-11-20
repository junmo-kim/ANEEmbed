import CoreML
import CoreVideo
import Darwin
import Foundation

/// Each stage bank stores contiguous FP16 inputs in lexicographic name order.
@available(macOS 26.0, *)
struct CoreMLValenWeights {
  let model: MLModel
  let inputs: [(String, [NSNumber], Int)]
  let filePrefix: String

  init(model: MLModel, mlp: Bool) throws {
    self.model = model
    filePrefix = mlp ? "valenMLP" : "valenDelta"
    var shapes: [String: [Int]]
    if mlp {
      shapes = ["norm_scale": [1, 2560, 1, 1]]
      for (i, width) in [4096, 4096, 1024].enumerated() {
        shapes["g\(i)"] = [1, 1, width, 2560]
        shapes["u\(i)"] = [1, 1, width, 2560]
        shapes["d\(i)"] = [1, 1, 2560, width]
      }
    } else {
      shapes = ["qkv": [1, 1, 8192, 2560], "z": [1, 1, 4096, 2560], "out": [1, 1, 2560, 4096],
                "rate": [1, 1, 32, 2560], "beta": [1, 1, 32, 2560], "norm": [1, 1, 1, 2560],
                "gatenorm": [1, 1, 1, 128], "bias": [1, 32, 1, 1], "decay": [1, 32, 1, 1]]
      for tap in 0..<4 { shapes["tap\(tap)"] = [1, 8192, 1, 1] }
    }
    let names = shapes.keys.sorted()
    let dynamic = mlp ? ["residual"] : ["hidden_in", "packed_state"]
    let outputs = mlp ? ["hidden_out"] : ["hidden_out", "packed_state_out"]
    let description = model.modelDescription
    let features = description.inputDescriptionsByName
    guard Set(features.keys) == Set(names + dynamic), Set(description.outputDescriptionsByName.keys) == Set(outputs) else {
      throw DecisionError("Unexpected shared Valen inputs or outputs")
    }
    for (name, feature) in Array(features) + Array(description.outputDescriptionsByName) {
      guard let array = feature.multiArrayConstraint, array.dataType == .float16, array.shape.count == 4,
        array.shape.allSatisfy({ $0.intValue > 0 }) else { throw DecisionError("Invalid shared Valen tensor " + name) }
      let expected = shapes[name] ?? (name.hasPrefix("packed_state") ? [1, 4288, 1, 128] : [1, 2560, 1, 128])
      guard expected == array.shape.map(\.intValue) else { throw DecisionError("Unexpected tensor shape for " + name) }
    }
    inputs = names.map { name in
      let shape = shapes[name]!
      return (name, shape.map { NSNumber(value: $0) }, shape.reduce(2, *))
    }
  }

  func read(stage: Int, root: URL, verifyOnly: Bool) throws -> [String: MLFeatureValue] {
    let url = try root.bundleChild("\(filePrefix)\(stage).fp16")
    let fd = open(url.path, O_RDONLY | O_NOFOLLOW)
    guard fd >= 0 else { throw DecisionError("Cannot open " + url.lastPathComponent) }
    defer { close(fd) }
    var info = stat()
    guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
      info.st_size == inputs.reduce(0, { $0 + $1.2 }) else { throw DecisionError("Invalid weight bank " + url.lastPathComponent) }
    if verifyOnly { return [:] }
    var result = [String: MLFeatureValue]()
    for (name, shape, size) in inputs {
      let value: MLMultiArray
      if size >= 1 << 20 {
        let width = shape.last!.intValue
        var surface: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, size / (2 * width), kCVPixelFormatType_OneComponent16Half,
                                  attrs, &surface) == kCVReturnSuccess, let surface else { throw DecisionError("Cannot allocate weight " + name) }
        value = MLMultiArray(pixelBuffer: surface, shape: shape)
      } else {
        value = try MLMultiArray(shape: shape, dataType: .float16)
      }
      try value.withUnsafeMutableBytes { buffer, _ in
        guard buffer.count == size, let address = buffer.baseAddress else { throw DecisionError("Invalid buffer for " + name) }
        var offset = 0
        while offset < size {
          let n = Darwin.read(fd, address.advanced(by: offset), size - offset)
          if n < 0 && errno == EINTR { continue }
          guard n > 0 else { throw DecisionError("Could not read " + url.lastPathComponent) }
          offset += n
        }
      }
      result[name] = MLFeatureValue(multiArray: value)
    }
    return result
  }
}
