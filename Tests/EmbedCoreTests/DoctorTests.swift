import Foundation
import Testing

@testable import EmbedCore

@Suite("Doctor")
struct DoctorTests {
    @Test func reportsAMissingModel() throws {
        let config = try Config(flags: ["model-dir": "/nonexistent/aneembed"], environment: [:])
        let report = Doctor.report(config)
        #expect(report["model"]?["present"] == false)
        #expect(report["ready"] == false)
        #expect(report["compile_cache"]?.stringValue?.hasPrefix("cold") == true)
        #expect(report["neural_engine"] != nil && report["macos_build"] != nil)
    }

    @Test func fakeRuntimeNeedsNoModel() throws {
        let config = try Config(flags: ["model-dir": "/nonexistent/aneembed", "runtime": "fake"], environment: [:])
        #expect(Doctor.report(config)["problems"] == [])
    }
}
