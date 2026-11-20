#if canImport(CoreAI)
import CoreAI
import Foundation

@available(macOS 27.0, *)
extension NDArray: DecisionBuffer {
  /// All bundled functions use rank-four FP16 tensors. Core AI may pad individual axes.
  mutating func write(_ f: (UnsafeMutablePointer<Float16>, [Int]) -> Void) {
    let s = strides
    mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in f(p, s) }
  }
  func read<T>(_ f: (UnsafePointer<Float16>, [Int]) -> T) -> T {
    let s = strides
    return rawView().withUnsafeBytes { raw, _, _ in f(raw.assumingMemoryBound(to: Float16.self), s)
    }
  }
  static func zeros(_ shape: [Int]) -> NDArray {
    var a = NDArray(shape: shape, scalarType: .float16)
    let span = zip(shape, a.strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
    _ = a.mutableRawView().withUnsafeMutableBytes { raw, _, _ in
      raw.initializeMemory(as: Float16.self, repeating: 0, count: span)
    }
    return a
  }
}
@available(macOS 27.0, *)
final class CoreAIResults: DecisionOutputs {
  var values: InferenceFunction.Outputs
  init(_ values: consuming InferenceFunction.Outputs) { self.values = values }
  func tensor(_ name: String) throws -> NDArray {
    guard let t = values.remove(name)?.ndArray, t.scalarType == .float16, t.shape.count == 4 else {
      throw DecisionError("Invalid output " + name)
    }
    return t
  }
}

@available(macOS 27.0, *)
extension InferenceFunction: DecisionFunction {
  func evaluate(_ inputs: [String: NDArray]) async throws -> CoreAIResults { try await CoreAIResults(run(inputs: inputs)) }
  func inputShape(_ name: String) throws -> [Int] {
    guard case .ndArray(let desc)? = descriptor.inputDescriptor(of: name), desc.scalarType == .float16 else {
      throw DecisionError("Invalid input " + name)
    }
    return desc.shape
  }
}

@available(macOS 27.0, *)
final class CoreAIGraphs: DecisionDevice {
  let root: URL, manifest: Manifest
  init(root: URL, manifest: Manifest) { self.root = root; self.manifest = manifest }
  private var models = [String: AIModel](), functions = [String: InferenceFunction]()
  func needsPrefetch(_ asset: String) -> Bool { models[asset] == nil }
  func function(_ physicalName: String, asset: String) async throws -> InferenceFunction {
    if let f = functions[physicalName] { return f }
    let model: AIModel
    if let cached = models[asset] {
      model = cached
    } else {
      model = try await Self.specialize(root.bundleChild(asset))
      models[asset] = model
    }
    let f = try Self.load(physicalName, from: model, asset: asset)
    functions[physicalName] = f
    return f
  }
  private static func specialize(_ url: URL) async throws -> AIModel {
    var options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    options.expectFrequentReshapes = false
    return try await AIModel.specialize(contentsOf: url, options: options, cachePolicy: .persistent)
  }

  private static func load(_ entry: String, from model: AIModel, asset: String) throws -> InferenceFunction {
    do {
      guard let f = try model.loadFunction(named: entry) else { throw DecisionError("Missing function " + entry) }
      return f
    } catch where String(describing: error) == "nilError" {
      throw CompiledFunctionLoadFailure(asset: asset, function: entry)
    }
  }

  typealias LoadedAsset = (String, AIModel, [String: InferenceFunction])

  func cache(_ assets: [LoadedAsset]) {
    for (name, model, entries) in assets {
      models[name] = model
      functions.merge(entries) { existing, _ in existing }
    }
  }

  func loadAssets(_ needed: [(Manifest.Asset, [String])])
    async throws -> [LoadedAsset]
  {
    try await withThrowingTaskGroup(of: LoadedAsset.self) { group in
      var completed = [LoadedAsset]()
      for (index, item) in needed.enumerated() {
        if index >= 4, let next = try await group.next() { completed.append(next) }
        group.addTask {
          try Task.checkCancellation()
          let (asset, names) = item
          let model = try await Self.specialize(self.root.bundleChild(asset.asset))
          var entries = [String: InferenceFunction]()
          for name in names { entries[name] = try Self.load(name, from: model, asset: asset.asset) }
          return (asset.asset, model, entries)
        }
      }
      for try await result in group { completed.append(result) }
      return completed
    }
  }

}

/// Inspect the compiled graph, since a preferred compute unit permits fallback.
@available(macOS 27.0, *)
extension CoreAIGraphs {
  func verifyPlacement() throws {
    let assets =
      (manifest.assets + manifest.deviceExtras).map { ($0.asset, $0.sourceHash) }
      + (manifest.visionAsset.map { [($0, manifest.visionSourceHash ?? "")] } ?? [])
    struct Metadata: Decodable { let sourceHash: String }
    for (asset, digest) in assets {
      let url = try root.bundleChild(asset)
      let meta = try url.decode("metadata.json", as: Metadata.self)
      guard meta.sourceHash.lowercased() == digest.lowercased(),
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil),
        let path = files.compactMap({ $0 as? URL }).first(where: {
          $0.lastPathComponent == "manifest.plist"
        })
      else { throw DecisionError("Missing compiled identity for " + asset) }
      let raw = try PropertyListSerialization.propertyList(
        from: Data(contentsOf: path), format: nil)
      guard let plist = raw as? [String: Any],
        let versions = plist["Package Version"] as? [String: [String: Any]],
        !versions.isEmpty
      else { throw DecisionError("Invalid placement manifest for " + asset) }
      var placedNames = Set<String>()
      for v in versions.values {
        guard let modules = v["Optimized Modules"] as? [String: [String: Any]], !modules.isEmpty
        else {
          throw DecisionError("Missing compiled modules in " + asset)
        }
        for module in modules.values {
          guard let entries = module["Entry Function Attributes"] as? [String: [Any]],
            !entries.isEmpty,
            entries.values.allSatisfy({ attrs in
              attrs.contains(where: { ($0 as? String) == "mps.fullyPlacedOnANE" })
            })
          else { throw DecisionError("Incomplete ANE placement in " + asset) }
          placedNames.formUnion(entries.keys)
        }
      }
      if manifest.embeddingLookup != nil {
        let expected =
          (manifest.assets + manifest.deviceExtras).first { $0.asset == asset }?.functions
          ?? manifest.visionFunctions ?? []
        guard !expected.isEmpty, placedNames.count == expected.count,
          expected.allSatisfy({ name in
            placedNames.contains { $0 == name || $0.hasPrefix(name + "_") }
          })
        else { throw DecisionError("Unexpected compiled functions in " + asset) }
      }
    }
  }
}
#endif
