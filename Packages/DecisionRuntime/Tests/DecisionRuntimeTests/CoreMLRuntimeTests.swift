import CoreML
import Foundation
import XCTest
@testable import DecisionRuntime

@available(macOS 26.0, *)
final class CoreMLRuntimeTests: XCTestCase {
  func testWeightReadAheadUsesOnlyTheRequestedStageInTheCurrentPass() async throws {
    let reads = ValenStageReads()
    let original: @Sendable (Int) throws -> ValenStageInputs = { stage in
      ValenStageInputs(prefix: ["stage": MLFeatureValue(int64: Int64(stage))], tail: [:])
    }
    let changed: @Sendable (Int) throws -> ValenStageInputs = { stage in
      ValenStageInputs(prefix: ["stage": MLFeatureValue(int64: -Int64(stage))], tail: [:])
    }
    _ = try await reads.take(0, followedBy: 1, read: original)
    let hit = try await reads.take(1, followedBy: 2, read: changed)
    XCTAssertEqual(hit.prefix["stage"]?.int64Value, 1)
    let skipped = try await reads.take(4, followedBy: 5, read: original)
    XCTAssertEqual(skipped.prefix["stage"]?.int64Value, 4)
    await reads.discard()
    let restarted = try await reads.take(5, followedBy: nil, read: changed)
    XCTAssertEqual(restarted.prefix["stage"]?.int64Value, -5)
  }

  func testPrefetchedWeightReadFailureReachesItsStage() async throws {
    let reads = ValenStageReads()
    let empty = ValenStageInputs(prefix: [:], tail: [:])
    _ = try await reads.take(0, followedBy: 1) { index in
      if index == 1 { throw CocoaError(.fileReadCorruptFile) }
      return empty
    }
    do {
      _ = try await reads.take(1, followedBy: nil) { _ in empty }
      XCTFail("Expected the prefetched bank error")
    } catch let error as CocoaError {
      XCTAssertEqual(error.code, .fileReadCorruptFile)
    }
    _ = try await reads.take(1, followedBy: nil) { _ in empty }
  }

  func testPlanConstructionRetriesOnlyAnInternalCoreMLErrorOnce() async throws {
    let internalError = NSError(domain: MLModelError.errorDomain, code: 0)
    var calls = 0
    let value = try await CoreMLDecisionCapability.loadPlan {
      calls += 1
      if calls == 1 { throw internalError }
      return 17
    }
    XCTAssertEqual(value, 17)
    XCTAssertEqual(calls, 2)
    for (error, expected) in [(internalError, 2), (NSError(domain: NSCocoaErrorDomain, code: 0), 1)] {
      calls = 0
      do {
        let _: Int = try await CoreMLDecisionCapability.loadPlan { calls += 1; throw error }
        XCTFail("Plan generation should fail")
      } catch let reported as NSError {
        XCTAssertEqual(reported.domain, error.domain)
        XCTAssertEqual(reported.code, error.code)
      }
      XCTAssertEqual(calls, expected)
    }
  }

  func testPlacementMayTryOneOtherSpecializationAndRequiresItsOwnPlan() async throws {
    var strategies = [MLOptimizationHints.SpecializationStrategy]()
    let count: Int = try await CoreMLDecisionCapability.withPreferredPlacement { config in
      XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
      strategies.append(config.optimizationHints.specializationStrategy)
      if strategies.count == 1 { throw CoreMLPlacementError(description: "CPU") }
      return 381
    }
    XCTAssertEqual(count, 381)
    XCTAssertEqual(strategies, [.default, .fastPrediction])

    for (error, expectedCalls): (Error, Int) in [
      (CoreMLPlacementError(description: "CPU"), 2), (DecisionError("Invalid manifest ABI"), 1),
    ] {
      var calls = 0
      do {
        let _: Int = try await CoreMLDecisionCapability.withPreferredPlacement { _ in
          calls += 1
          throw error
        }
        XCTFail("Neither rejected placement nor an invalid interface may enable this function")
      } catch {
        XCTAssertEqual(calls, expectedCalls)
      }
    }
  }

  func testUnresolvedPlacementRechecksTheSameFilesBeforeRememberingAnAlias() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mlmodelc")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    var paths: CoreMLAssetPaths? = CoreMLAssetPaths()
    var calls = [URL]()
    let resolved = try await paths!.withResolvedPlan(at: root) { candidate -> URL in
      calls.append(candidate)
      if candidate == root { throw CoreMLPlacementError(description: "unspecified", missingDevice: true) }
      XCTAssertEqual(candidate.resolvingSymlinksInPath().path, root.resolvingSymlinksInPath().path)
      return candidate
    }
    XCTAssertEqual(calls, [root, resolved])
    let cached = try await paths!.withResolvedPlan(at: root) { candidate in
      calls.append(candidate)
      return candidate
    }
    XCTAssertEqual(cached, resolved)
    XCTAssertEqual(calls, [root, resolved, resolved])
    paths = nil
    XCTAssertFalse(FileManager.default.fileExists(atPath: resolved.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
  }

  func testReportedCPUAndUnresolvedSecondPathFailClosed() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for unresolved in [false, true] {
      let paths = CoreMLAssetPaths()
      var attempts = 0
      do {
        let _: Int = try await paths.withResolvedPlan(at: root) { _ in
          attempts += 1
          throw CoreMLPlacementError(description: "rejected", missingDevice: unresolved)
        }
        XCTFail("Rejected placement must stop the load")
      } catch is CoreMLPlacementError {
        XCTAssertEqual(attempts, unresolved ? 2 : 1)
      }
    }
  }

  func testEligibilityRequiresBothTargetMacAndExposedANE() throws {
    XCTAssertThrowsError(try CoreMLDecisionCapability.requireHardware(neuralEngine: false, model: "Mac16,10"))
    XCTAssertThrowsError(try CoreMLDecisionCapability.requireHardware(neuralEngine: true, model: "unknown"))
    try CoreMLDecisionCapability.requireHardware(neuralEngine: true, model: "Mac16,10")
    XCTAssertEqual(CoreMLDecisionCapability.configuration().computeUnits, .cpuAndNeuralEngine)
  }

  func testTrialRequiresTheSelectedMacAndAnAvailableNeuralEngine() throws {
    let laptop = "MacBookPro18,3"
    XCTAssertThrowsError(try CoreMLDecisionCapability.requireHardware(neuralEngine: true, model: laptop))
    XCTAssertThrowsError(try CoreMLDecisionCapability.requireHardware(neuralEngine: true, model: laptop, trialModel: "Mac16,10"))
    XCTAssertThrowsError(try CoreMLDecisionCapability.requireHardware(neuralEngine: false, model: laptop, trialModel: laptop))
    try CoreMLDecisionCapability.requireHardware(neuralEngine: true, model: laptop, trialModel: laptop)
  }

  func testRotaryAndCacheCopiesKeepIndependentValues() {
    var first = CoreMLBuffer.zeros([1, 2, 1, 3])
    first.write { pointer, strides in pointer[strides[1] + 2 * strides[3]] = 7 }
    var second = first
    second.write { pointer, strides in pointer[strides[1] + 2 * strides[3]] = 9 }
    XCTAssertEqual(first.read { $0[$1[1] + 2 * $1[3]] }, 7)
    XCTAssertEqual(second.read { $0[$1[1] + 2 * $1[3]] }, 9)
  }

  func testPredictionBuffersForwardTheCoreMLArrayAndDetachOnWrite() {
    let tensor = MLMultiArray(shape: [1, 2, 1, 3], dataType: .float16, strides: [10, 5, 3, 1])
    tensor[[0, 1, 0, 2] as [NSNumber]] = 7
    let input = CoreMLBuffer(output: tensor)
    XCTAssertTrue(input.multiArray === tensor)
    XCTAssertEqual(input.read { $0[$1[1] + 2 * $1[3]] }, 7)
    var copy = input
    copy.write { $0[$1[1] + 2 * $1[3]] = 9 }
    XCTAssertEqual(copy.read { $0[$1[1] + 2 * $1[3]] }, 9)
    XCTAssertEqual(tensor[[0, 1, 0, 2] as [NSNumber]].floatValue, 7)
  }

  func testPoolMaskKeepsOverlappingRolesAcrossChunks() {
    let spans = [0..<1, 126..<130, 128..<133, 132..<133]
    let first: CoreMLBuffer = ValenRoleBatch.mask(spans: spans, chunk: 0, width: 128)
    let second: CoreMLBuffer = ValenRoleBatch.mask(spans: spans, chunk: 1, width: 128)
    for (chunk, mask) in [first, second].enumerated() {
      XCTAssertEqual(mask.shape, [1, 1, 128, 288])
      mask.read { values, strides in
        for token in 0..<128 {
          for role in 0...spans.count {
            let active = role < spans.count && spans[role].contains(128 * chunk + token)
            XCTAssertEqual(values[token * strides[2] + role * strides[3]], active ? 1 : 0)
          }
        }
      }
    }
  }

  func testScoreLanesPreserveRoleAndCandidateOrderAndPadding() {
    var pooled = CoreMLBuffer.zeros([1, 2, 1, 288])
    pooled.write { values, strides in
      for channel in 0..<2 {
        for role in 0..<288 {
          values[channel * strides[1] + role * strides[3]] = Float16(512 * channel + role)
        }
      }
    }
    for count in [2, 3, 256] {
      let packed = ValenRoleBatch.candidateRoles(pooled, count: count)
      XCTAssertEqual(packed.shape, [1, 2, 4, 256])
      packed.read { values, strides in
        for channel in 0..<2 {
          for slot in 0..<256 {
            for (role, column) in [0, 1, 2 + slot, 2 + count].enumerated() {
              let expected = slot < count ? Float16(512 * channel + column) : 0
              XCTAssertEqual(values[channel * strides[1] + role * strides[2] + slot * strides[3]], expected)
            }
          }
        }
      }
    }
  }

  func testCoreMLValenCanShipOnlyItsExportedAttentionTiers() throws {
    try Manifest.validateCoreMLTiers([256], context: 256)
    try Manifest.validateCoreMLTiers([256, 1024], context: 1024)
    try Manifest.validateCoreMLTiers([256, 512, 1024, 2048, 4096], context: 4096)
    XCTAssertEqual(Manifest.valenRotaryPositions, 4096)
    for tiers in [[], [0, 256], [512], [256, 256], [256, 1024, 512], [256, 768], [256, 8192]] {
      XCTAssertThrowsError(try Manifest.validateCoreMLTiers(tiers, context: tiers.last ?? 0))
    }
    XCTAssertThrowsError(try Manifest.validateCoreMLTiers([256], context: 4096))
  }

  func testSharedRecurrentStagesRequireBothPrefixesAndTheSharedMLP() throws {
    let stages = ["linear", "linear", "attention"]
    let functions = [["scan0"], ["scan1"], ["scan2", "attend2c256"]]
    let physical = ["valenDeltaFirst", "valenDelta", "valenMLP", "scan2", "attend2c256"]
    func check(_ names: [String]) throws {
      let bindings = Dictionary(uniqueKeysWithValues: names.map { ($0, $0 + ".mlmodelc") })
      let model: [String: Any] = [
        "source": ["repo": "Valen-Team/Valen-Decision-1.5-4B", "revision": "test"],
        "architecture": "h16g", "backend": "coreml", "coreml_functions": bindings,
        "coreml_recurrent_layout": "shared-v1", "stage_kinds": stages,
        "assets": functions.enumerated().map { i, entries in
          ["asset": "stage\(i).aimodel", "source_hash": "test", "functions": entries] as [String: Any]
        },
        "execution": [["stages": [0, 1, 2], "passes": 1]], "width": 128, "hidden": 2560,
        "context": 256, "tiers": [256], "heads": 16, "kv_heads": 4, "head_dim": 256,
        "rotary_dim": 64, "block_mask": true
      ]
      let data = try JSONSerialization.data(withJSONObject: model)
      try JSONDecoder().decode(Manifest.self, from: data).validateCoreMLBindings()
    }
    try check(physical)
    for missing in physical { XCTAssertThrowsError(try check(physical.filter { $0 != missing })) }
    XCTAssertThrowsError(try check(physical + ["scan1"]))
  }

  func testFunctionBindingsRequireMatchingLogicalNamesAndAllAttentionTiers() throws {
    func validate(_ entries: [String], _ bindings: [String: String]) throws {
      let manifest: [String: Any] = [
        "source": ["repo": "Valen-Team/Valen-Decision-1.5-4B", "revision": "test"],
        "architecture": "h16g", "backend": "coreml", "coreml_functions": bindings,
        "assets": [["asset": "stage0.aimodel", "source_hash": "test", "functions": entries]],
        "execution": [["stages": [0], "passes": 1]], "stage_kinds": ["attention"],
        "width": 128, "hidden": 2560, "context": 512, "tiers": [256, 512],
        "heads": 16, "kv_heads": 4, "head_dim": 256, "rotary_dim": 64, "block_mask": true
      ]
      let data = try JSONSerialization.data(withJSONObject: manifest)
      try JSONDecoder().decode(Manifest.self, from: data).validateCoreMLBindings()
    }
    let logical = ["scan0", "attend0c256", "attend0c512"]
    let entries = ["scan0_" + String(repeating: "a", count: 64)] + logical.dropFirst()
    let bindings = Dictionary(uniqueKeysWithValues: logical.map { ($0, $0 + ".mlmodelc") })
    try validate(entries, bindings)
    let missingTier = bindings.filter { $0.key != "attend0c512" }
    XCTAssertThrowsError(try validate(Array(entries.dropLast()), missingTier))
    XCTAssertThrowsError(try validate(entries, missingTier))
    var escape = bindings
    escape["scan0"] = "../scan0.mlmodelc"
    XCTAssertThrowsError(try validate(entries, escape))
    XCTAssertThrowsError(try validate(entries + ["scan0"], bindings))
  }
}
