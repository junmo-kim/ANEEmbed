import Foundation
import Testing

/// Runs the jpeg-fuzz tool (Sources/jpeg-fuzz) as a child process over the fixture JPEGs: a decoder trap kills
/// only the child, and its exit status fails the test. ANEEMBED_FUZZ_ITERATIONS overrides the iteration count.
@Suite("JPEG decoder fuzzing")
struct JPEGFuzzTests {
    static var tool: URL? {
        // The tool is built next to the test bundle (EmbedCoreTests depends on the jpeg-fuzz target).
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            let candidate = bundle.bundleURL.deletingLastPathComponent().appendingPathComponent("jpeg-fuzz")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        let arguments = CommandLine.arguments.compactMap { $0.hasSuffix(".xctest") || $0.contains(".xctest/") ? $0 : nil }
        for path in arguments {
            var url = URL(fileURLWithPath: path)
            while url.pathComponents.count > 1, !url.lastPathComponent.hasSuffix(".xctest") { url.deleteLastPathComponent() }
            let candidate = url.deletingLastPathComponent().appendingPathComponent("jpeg-fuzz")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    func run(_ arguments: [String]) throws -> (Int32, Process.TerminationReason, String) {
        let tool = try #require(Self.tool, "jpeg-fuzz was not built next to the test bundle")
        let process = Process()
        process.executableURL = tool
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, process.terminationReason, String(decoding: data, as: UTF8.self))
    }

    @Test func mutatedFixturesNeverCrashTheDecoder() throws {
        let iterations = ProcessInfo.processInfo.environment["ANEEMBED_FUZZ_ITERATIONS"].flatMap(Int.init) ?? 3000
        let crash = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-crash-\(getpid()).jpg")
        defer { try? FileManager.default.removeItem(at: crash) }
        // This checks crashes only. The per-input time limit (10 s by default) is calibrated for the release build that
        // CI fuzzes separately; a debug build on a shared runner can take longer than that on a legitimate input.
        let (status, reason, output) = try run(["--iterations", "\(iterations)", "--seed", "1", "--max-seconds", "600",
                                                "--crash-file", crash.path, Fixtures.url("images").path])
        #expect(reason == .exit && status == 0, "jpeg-fuzz failed (\(reason == .uncaughtSignal ? "signal" : "exit") \(status)): \(output)")
        #expect(output.contains("0 failures"))
    }

    /// The harness itself must notice a crash: a JPEG whose DC symbol is 255 traps an unfixed decoder, so here it
    /// only has to be rejected, and a missing corpus is a usage error rather than a silent pass.
    @Test func replaysAFileAndRefusesAnEmptyCorpus() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-replay-\(getpid()).jpg")
        try CraftedJPEG.make(dcSymbol: 255).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let (status, _, output) = try run(["--replay", file.path])
        #expect(status == 0 && output.contains("rejected"), "\(output)")
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-empty-\(getpid())")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(try run([empty.path]).0 == 64)
    }
}
