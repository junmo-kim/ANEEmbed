import CoreML
import Foundation
import XCTest

final class CoreMLFixtureTests: XCTestCase {
  func testCompilesWithExactFloat16ABI() throws {
    let source = try XCTUnwrap(Bundle.module.url(
      forResource: "ValenCoreMLFixture", withExtension: "mlpackage", subdirectory: "Fixtures"))
    let compiled = try MLModel.compileModel(at: source)
    defer { try? FileManager.default.removeItem(at: compiled) }

    // A synthetic reference for the hosted runner. Product model eligibility is tested separately.
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let model = try MLModel(contentsOf: compiled, configuration: config)
    let shape: [NSNumber] = [1, 2, 1, 2]
    let inputs = model.modelDescription.inputDescriptionsByName
    let outputs = model.modelDescription.outputDescriptionsByName
    XCTAssertEqual(Set(inputs.keys), ["hidden_in"])
    XCTAssertEqual(Set(outputs.keys), ["hidden_out"])
    for feature in [inputs["hidden_in"], outputs["hidden_out"]] {
      let abi = try XCTUnwrap(feature?.multiArrayConstraint)
      XCTAssertEqual(abi.shape, shape)
      XCTAssertEqual(abi.dataType, .float16)
    }
    let input = try MLMultiArray(shape: shape, dataType: .float16)
    for (index, value) in [1, 2, 3, 4].enumerated() { input[index] = NSNumber(value: value) }
    let features = try MLDictionaryFeatureProvider(dictionary: ["hidden_in": MLFeatureValue(multiArray: input)])
    let prediction = try model.prediction(from: features)
    let result = try XCTUnwrap(prediction.featureValue(for: "hidden_out")?.multiArrayValue)
    XCTAssertEqual(result.shape, shape)
    XCTAssertEqual(result.dataType, .float16)
    XCTAssertEqual((0..<result.count).map { result[$0].floatValue }, [0, 1, 8, 12])
  }
}
