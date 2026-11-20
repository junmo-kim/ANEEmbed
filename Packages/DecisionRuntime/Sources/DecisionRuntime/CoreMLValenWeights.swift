import CoreML
import CoreVideo
import Darwin
import Foundation

/// Each stage bank stores contiguous FP16 inputs in lexicographic name order.
@available(macOS 26.0, *)
struct CoreMLValenWeights: Sendable {
  let inputs: [(String, [Int], Int)]
  let filePrefix: String

  init(model: MLModel, mlp: Bool) throws {
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
      return (name, shape, shape.reduce(2, *))
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
    for (name, dimensions, size) in inputs {
      try Task.checkCancellation()
      let shape = dimensions.map { NSNumber(value: $0) }
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

// Each read allocates new feature buffers. Only the consuming stage receives them,
// and it passes them to Core ML as immutable inputs.
@available(macOS 26.0, *)
struct ValenStageInputs: @unchecked Sendable {
  let prefix: [String: MLFeatureValue]
  let tail: [String: MLFeatureValue]
}

@available(macOS 26.0, *)
final class ValenStageReads {
  private var pending: (index: Int, task: Task<ValenStageInputs, Error>)?

  deinit { pending?.task.cancel() }

  func discard() async {
    guard let prior = pending else { return }
    pending = nil
    prior.task.cancel()
    _ = await prior.task.result
  }

  func take(_ stage: Int, followedBy next: Int?,
            read: @escaping @Sendable (Int) throws -> ValenStageInputs) async throws -> ValenStageInputs {
    try Task.checkCancellation()
    let inputs: ValenStageInputs
    if let prior = pending, prior.index == stage {
      pending = nil
      inputs = try await withTaskCancellationHandler {
        try await prior.task.value
      } onCancel: { prior.task.cancel() }
    } else {
      await discard()
      inputs = try read(stage)
    }
    try Task.checkCancellation()
    if let next {
      pending = (next, Task.detached(priority: Task.currentPriority) {
        try Task.checkCancellation()
        return try read(next)
      })
    }
    return inputs
  }
}
