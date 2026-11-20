import CoreML
import Darwin
import Foundation

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
    try requireHardware(neuralEngine: hasNeuralEngine, model: hardwareModel)
  }

  static func requireHardware(neuralEngine: Bool, model: String) throws {
    guard neuralEngine else { throw DecisionError("Core ML Decisions require an available Neural Engine") }
    guard model == "Mac16,10" else {
      throw DecisionError("Unsupported Core ML Valen hardware: " + model + " (experimental target: M4 Mac mini)")
    }
  }

  static func configuration() -> MLModelConfiguration {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    return config
  }

  /// The plan is a static device preference. Core ML does not expose per-prediction device attestation.
  public static func plannedOperations(at compiled: URL) async throws -> Int {
    try requireHardware()
    return try await checkPlan(compiled, configuration())
  }

  static func checkPlan(_ url: URL, _ config: MLModelConfiguration) async throws -> Int {
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
    guard case .program(let program) = plan.modelStructure, program.functions.count == 1,
          let function = program.functions["main"] else {
      throw DecisionError("Expected a single main ML Program in " + url.lastPathComponent)
    }
    func inspect(_ block: MLModelStructure.Program.Block) throws -> Int {
      var count = 0
      for op in block.operations {
        // Literal constants encode parameters; they do not execute a model operation.
        if op.operatorName != "const" {
          guard let usage = plan.deviceUsage(for: op), case .neuralEngine = usage.preferred else {
            throw DecisionError("Missing preferred Neural Engine placement for " + op.operatorName + " in " + url.lastPathComponent)
          }
          count += 1
        }
        for child in op.blocks { count += try inspect(child) }
      }
      return count
    }
    let count = try inspect(function.block)
    guard count > 0 else { throw DecisionError("No planned model operations in " + url.lastPathComponent) }
    return count
  }
}

@available(macOS 26.0, *)
struct CoreMLBuffer: DecisionBuffer {
  var value: MLShapedArray<Float16>
  var shape: [Int] { value.shape }
  static func zeros(_ shape: [Int]) -> Self { Self(value: .init(repeating: 0, shape: shape)) }
  func read<T>(_ body: (UnsafePointer<Float16>, [Int]) -> T) -> T {
    value.withUnsafeShapedBufferPointer { buffer, _, strides in body(buffer.baseAddress!, strides) }
  }
  mutating func write(_ body: (UnsafeMutablePointer<Float16>, [Int]) -> Void) {
    value.withUnsafeMutableShapedBufferPointer { buffer, _, strides in body(buffer.baseAddress!, strides) }
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
    return CoreMLBuffer(value: MLShapedArray<Float16>(array))
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
      features[name] = MLFeatureValue(multiArray: MLMultiArray(buffer.value))
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
    let config = CoreMLDecisionCapability.configuration()
    _ = try await CoreMLDecisionCapability.checkPlan(url, config)
    let model = try await MLModel.load(contentsOf: url, configuration: config)
    try validateFeatures(model.modelDescription, entry: logical)
    return CoreMLFunction(model: model)
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
      case "valenPool": inputs = ["hidden", "valid", "length", "previous"]; outputs = ["pooled"]
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
