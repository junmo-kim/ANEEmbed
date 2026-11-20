import CoreML
import Darwin
import Foundation

@available(macOS 26.0, *)
struct CoreMLPlacementError: Error, CustomStringConvertible {
  let description: String
}

@available(macOS 26.0, *)
public enum CoreMLDecisionCapability {
  public static var hardwareModel: String {
    var size = 0
    guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 1 else { return "unknown" }
    var bytes = [CChar](repeating: 0, count: size)
    guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
    return String(cString: bytes)
  }
  public static var hasNeuralEngine: Bool {
    MLModel.availableComputeDevices.contains { if case .neuralEngine = $0 { return true }; return false }
  }
  public static func requireHardware() throws {
    try requireHardware(neuralEngine: hasNeuralEngine, model: hardwareModel,
                        trialModel: ProcessInfo.processInfo.environment["ANERUN_COREML_TRIAL_MODEL"])
  }

  static func requireHardware(neuralEngine: Bool, model: String, trialModel: String? = nil) throws {
    guard neuralEngine else { throw DecisionError("Core ML Decisions require an available Neural Engine") }
    guard model == "Mac16,10" || (model.hasPrefix("Mac") && model == trialModel) else {
      throw DecisionError("Unsupported Core ML Valen hardware: " + model + " (experimental target: M4 Mac mini)")
    }
  }

  static func configuration(_ strategy: MLOptimizationHints.SpecializationStrategy = .default)
    -> MLModelConfiguration
  {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    config.optimizationHints.specializationStrategy = strategy
    return config
  }

  /// The plan is a static device preference. Core ML does not expose per-prediction device attestation.
  public static func plannedOperations(at compiled: URL) async throws -> Int {
    try requireHardware()
    let asset = try MLModelAsset(url: compiled)
    return try await withPreferredPlacement { config in
      try await checkPlan(asset, url: compiled, config)
    }
  }

  // Different specialization strategies can produce different segmentations.
  // Only return a result from a configuration whose entire plan passed.
  static func withPreferredPlacement<T>(_ run: (MLModelConfiguration) async throws -> T) async throws -> T {
    do { return try await run(configuration()) }
    catch is CoreMLPlacementError {
      try Task.checkCancellation()
      return try await run(configuration(.fastPrediction))
    }
  }

  // Core ML can fail to construct segmentation analytics for a loaded scan.
  // Retry plan construction once. Every successful plan still goes through the
  // complete operation-by-operation placement check.
  static func loadPlan<T>(_ load: () async throws -> T) async throws -> T {
    do { return try await load() }
    catch let error as NSError where error.domain == MLModelError.errorDomain && error.code == 0 {
      try Task.checkCancellation()
      return try await load()
    }
  }

  static func checkPlan(_ asset: MLModelAsset, url: URL, _ config: MLModelConfiguration) async throws -> Int {
    let plan: MLComputePlan
    do {
      plan = try await loadPlan { try await MLComputePlan.load(asset: asset, configuration: config) }
    } catch let error as NSError where error.domain == MLModelError.errorDomain && error.code == 0 {
      throw CoreMLPlacementError(description: "Could not construct the execution plan for "
        + url.lastPathComponent + ": " + error.localizedDescription)
    }
    func inspect(_ block: MLModelStructure.Program.Block) throws -> Int {
      var count = 0
      for op in block.operations {
        for child in op.blocks { count += try inspect(child) }
        // Literal constants encode parameters; they do not execute a model operation.
        if op.operatorName != "const" {
          let rejected: String
          switch plan.deviceUsage(for: op)?.preferred {
          case .neuralEngine?: count += 1; continue
          case .cpu?: rejected = "CPU"
          case .gpu?: rejected = "GPU"
          case nil: rejected = "unspecified"
          @unknown default: rejected = "unknown"
          }
          throw CoreMLPlacementError(description: "Expected preferred Neural Engine placement for " + op.operatorName
            + " in " + url.lastPathComponent + " (reported " + rejected + ")")
        }
      }
      return count
    }
    func countOperations(_ structure: MLModelStructure) throws -> Int {
      switch structure {
      case .program(let program):
        guard program.functions.count == 1, let function = program.functions["main"] else {
          throw DecisionError("Expected a single main ML Program in " + url.lastPathComponent)
        }
        return try inspect(function.block)
      case .pipeline(let pipeline):
        guard !pipeline.subModels.isEmpty else {
          throw DecisionError("Empty Core ML pipeline in " + url.lastPathComponent)
        }
        return try pipeline.subModels.reduce(0) { try $0 + countOperations($1) }
      default:
        throw DecisionError("Unsupported model structure in " + url.lastPathComponent)
      }
    }
    let count = try countOperations(plan.modelStructure)
    guard count > 0 else { throw DecisionError("No planned model operations in " + url.lastPathComponent) }
    return count
  }
}

@available(macOS 26.0, *)
struct CoreMLBuffer: DecisionBuffer {
  // Keep prediction outputs in Core ML's native layout between consecutive
  // models. Only materialize a Swift-owned buffer when a caller writes to it.
  private var owned: MLShapedArray<Float16>?
  private var output: MLMultiArray?

  init(output: MLMultiArray) { self.output = output }
  private init(owned: MLShapedArray<Float16>) { self.owned = owned }
  var shape: [Int] { owned?.shape ?? output!.shape.map(\.intValue) }
  var multiArray: MLMultiArray { output ?? MLMultiArray(owned!) }

  static func zeros(_ shape: [Int]) -> Self { Self(owned: .init(repeating: 0, shape: shape)) }
  func read<T>(_ body: (UnsafePointer<Float16>, [Int]) -> T) -> T {
    if let output {
      return output.withUnsafeBytes { bytes in
        body(bytes.bindMemory(to: Float16.self).baseAddress!, output.strides.map(\.intValue))
      }
    }
    return owned!.withUnsafeShapedBufferPointer { buffer, _, strides in body(buffer.baseAddress!, strides) }
  }
  mutating func write(_ body: (UnsafeMutablePointer<Float16>, [Int]) -> Void) {
    if let output {
      owned = MLShapedArray<Float16>(output)
      self.output = nil
    }
    owned!.withUnsafeMutableShapedBufferPointer { buffer, _, strides in body(buffer.baseAddress!, strides) }
  }
}

@available(macOS 26.0, *)
struct CoreMLResults: DecisionOutputs {
  let provider: MLFeatureProvider
  func tensor(_ name: String) throws -> CoreMLBuffer {
    guard let array = provider.featureValue(for: name)?.multiArrayValue,
          array.dataType == .float16, array.shape.count == 4, array.count > 0 else {
      throw DecisionError("Invalid FP16 output: " + name)
    }
    return CoreMLBuffer(output: array)
  }
}

@available(macOS 26.0, *)
struct CoreMLFunction: DecisionFunction {
  let model: MLModel
  func inputShape(_ name: String) throws -> [Int] {
    guard let constraint = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint,
          constraint.dataType == .float16, constraint.shape.count == 4, constraint.shape.allSatisfy({ $0.intValue > 0 }) else {
      throw DecisionError("Invalid fixed FP16 input: " + name)
    }
    return constraint.shape.map(\.intValue)
  }
  func evaluate(_ inputs: [String: CoreMLBuffer]) async throws -> CoreMLResults {
    guard Set(inputs.keys) == Set(model.modelDescription.inputDescriptionsByName.keys) else {
      throw DecisionError("Core ML function input names differ from manifest ABI")
    }
    var features = [String: MLFeatureValue]()
    for (name, buffer) in inputs {
      guard buffer.shape == (try inputShape(name)) else { throw DecisionError("Input shape mismatch: " + name) }
      features[name] = MLFeatureValue(multiArray: buffer.multiArray)
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: features)
    return CoreMLResults(provider: try await model.prediction(from: provider))
  }
}

@available(macOS 26.0, *)
final class CoreMLGraphs: DecisionDevice {
  let root: URL, manifest: Manifest
  var functions = [String: CoreMLFunction]()
  init(root: URL, manifest: Manifest) { self.root = root; self.manifest = manifest }

  private func checkedFunction(_ entry: String) async throws -> CoreMLFunction {
    let logical = Manifest.logicalFunction(entry)
    guard let path = manifest.coreMLFunctions?[logical], path.hasSuffix(".mlmodelc") else {
      throw DecisionError("Missing installed Core ML function: " + logical)
    }
    try CoreMLDecisionCapability.requireHardware()
    let url = try root.bundleChild(path)
    let modelAsset = try MLModelAsset(url: url)
    return try await CoreMLDecisionCapability.withPreferredPlacement { config in
      let model = try await MLModel.load(asset: modelAsset, configuration: config)
      _ = try await CoreMLDecisionCapability.checkPlan(modelAsset, url: url, config)
      try validateFeatures(model.modelDescription, entry: logical)
      return CoreMLFunction(model: model)
    }
  }

  // Packages are independently compiled. Validate the interface before any revision can become active.
  private func validateFeatures(_ description: MLModelDescription, entry: String) throws {
    let inputs: Set<String>, outputs: Set<String>
    if entry.hasPrefix("lookup") {
      inputs = ["one_hot"]; outputs = ["hidden_out"]
    } else if entry.hasPrefix("scan"), let i = Int(entry.dropFirst(4)), manifest.stageKinds.indices.contains(i) {
      let attention = manifest.stageKinds[i] == "attention"
      inputs = attention ? ["hidden_in", "packed_state", "cos", "sin"] : ["hidden_in", "packed_state"]
      outputs = attention ? ["hidden_out", "packed_state_out", "gate", "query", "key", "value"] : ["hidden_out", "packed_state_out"]
    } else if entry.hasPrefix("attend") {
      inputs = ["hidden_in", "query", "gate", "key", "value", "mask"]; outputs = ["hidden_out"]
    } else {
      switch entry {
      case "valenScore": inputs = ["pooled", "mask", "choice_scale", "choice_bias", "mad_scale"]; outputs = ["probabilities", "statistics"]
      case "valenPool", "valenPool288": inputs = ["hidden", "valid", "length", "previous"]; outputs = ["pooled"]
      case "valenMixer": inputs = ["pooled"]; outputs = ["logit"]
      case "readout": inputs = ["logits", "mask"]; outputs = ["probabilities"]
      case "stats": inputs = ["p", "top_index", "choice_scale", "choice_bias", "mad_scale"]; outputs = ["statistics"]
      case "visionpos": inputs = ["row", "column", "fy", "fx"]; outputs = ["pos"]
      case "featurescale": inputs = ["features"]; outputs = ["scaled"]
      case "p256", "p1024": inputs = ["pixel_values", "cos", "sin", "pos_embeds", "attn_mask"]; outputs = ["features"]
      default: throw DecisionError("Unknown Core ML Valen function " + entry)
      }
    }
    let incoming = description.inputDescriptionsByName, outgoing = description.outputDescriptionsByName
    guard Set(incoming.keys) == inputs, Set(outgoing.keys) == outputs else {
      throw DecisionError("Unexpected input/output names in " + entry)
    }
    if entry == "valenPool288" || entry == "valenScore" {
      let expected: [String: [Int]]
      if entry == "valenPool288" {
        expected = ["hidden": [1,2560,1,128], "valid": [1,1,128,288], "length": [1,1,1,288],
                    "previous": [1,2560,1,288], "pooled": [1,2560,1,288]]
      } else {
        expected = ["pooled": [1,2560,4,256], "mask": [1,256,1,1], "choice_scale": [1,1,1,1],
                    "choice_bias": [1,1,1,1], "mad_scale": [1,1,1,1], "probabilities": [1,256,1,1],
                    "statistics": [1,3,1,1]]
      }
      for (name, shape) in expected {
        let feature = incoming[name] ?? outgoing[name]
        guard feature?.multiArrayConstraint?.shape.map(\.intValue) == shape else {
          throw DecisionError("Unexpected shape for " + name + " in " + entry)
        }
      }
    }
    for feature in Array(incoming.values) + Array(outgoing.values) {
      guard let array = feature.multiArrayConstraint, array.dataType == .float16,
            array.shape.count == 4, array.shape.allSatisfy({ $0.intValue > 0 }) else {
        throw DecisionError("Expected a rank-four FP16 feature " + feature.name + " in " + entry)
      }
    }
  }

  func function(_ entry: String, asset: String) async throws -> CoreMLFunction {
    if let f = functions[entry] { return f }
    let f = try await checkedFunction(entry)
    functions[entry] = f
    return f
  }
  func prepare(_ entry: String, asset: String) async throws {
    // Audit the whole model without keeping every tier's weights resident.
    _ = try await checkedFunction(entry)
  }

  // A Core ML package holds one function. Load each on demand at its first evaluation.
  func needsPrefetch(_ asset: String) -> Bool { false }
  func loadAssets(_ needed: [(Manifest.Asset, [String])]) async throws -> [Never] { [] }
  func cache(_ assets: [Never]) {}
  func verifyPlacement() throws { try CoreMLDecisionCapability.requireHardware() }
}
