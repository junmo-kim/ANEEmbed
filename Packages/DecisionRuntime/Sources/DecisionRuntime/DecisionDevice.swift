import Foundation

protocol DecisionBuffer {
  var shape: [Int] { get }
  static func zeros(_ shape: [Int]) -> Self
  mutating func write(_ body: (UnsafeMutablePointer<Float16>, [Int]) -> Void)
  func read<T>(_ body: (UnsafePointer<Float16>, [Int]) -> T) -> T
}

protocol DecisionOutputs {
  associatedtype Buffer: DecisionBuffer
  mutating func tensor(_ name: String) throws -> Buffer
}

protocol DecisionFunction {
  associatedtype ResultTensors: DecisionOutputs
  func inputShape(_ name: String) throws -> [Int]
  func evaluate(_ inputs: [String: ResultTensors.Buffer]) async throws -> ResultTensors
}

protocol DecisionDevice: AnyObject {
  associatedtype Function: DecisionFunction
  associatedtype LoadedAsset
  init(root: URL, manifest: Manifest)
  func function(_ entry: String, asset: String) async throws -> Function
  func prepare(_ entry: String, asset: String) async throws
  func needsPrefetch(_ asset: String) -> Bool
  func loadAssets(_ needed: [(Manifest.Asset, [String])]) async throws -> [LoadedAsset]
  func cache(_ assets: [LoadedAsset])
  func verifyPlacement() throws
}

extension DecisionDevice {
  func prepare(_ entry: String, asset: String) async throws {
    _ = try await function(entry, asset: asset)
  }
}
