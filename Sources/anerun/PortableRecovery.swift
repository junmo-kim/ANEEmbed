import DecisionRuntime
import EmbedCore
import Foundation
#if canImport(CoreAI)
import CoreAI
#endif

/// One targeted source load and one ordinary AOT prepare, in separate processes under the
/// same executable identity. The parent owns the install lock, receipt audit and activation.
struct PortableRecovery {
    var cacheRoot: () throws -> URL
    var warmSource: (URL, String) async throws -> Void
    var retryCompiled: (URL) async throws -> Void

    static var live: PortableRecovery { PortableRecovery(cacheRoot: PortableDecision.nativeCacheRoot,
        warmSource: { try await child(["_portable-load-source", $0.path, $1]) },
        retryCompiled: { try await child(["_portable-prepare-compiled", $0.path]) }) }

    private static func child(_ arguments: [String]) async throws {
        try Task.checkCancellation()
        guard let executable = Bundle.main.executableURL else { throw PortableDecision.invalid("missing installer executable") }
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = FileHandle.standardError
        try process.run()
        do {
            while process.isRunning { try await Task.sleep(for: .milliseconds(100)) }
        } catch {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
            throw error
        }
        guard process.terminationStatus == 0 else {
            throw ServiceError(.unavailable, "prepare_failed", "portable recovery child exited with status \(process.terminationStatus)")
        }
    }

    static func runChild(_ args: [String]) async throws {
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            if args.count == 3, args[0] == "_portable-load-source" {
                let url = URL(fileURLWithPath: args[1])
                guard url.pathExtension == "aimodel" else { throw PortableDecision.invalid("expected a source asset") }
                var options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
                options.expectFrequentReshapes = false
                let model = try await AIModel.specialize(contentsOf: url, options: options, cachePolicy: .persistent)
                guard try model.loadFunction(named: args[2]) != nil else { throw PortableDecision.invalid("missing source function") }
                return
            }
            if args.count == 2, args[0] == "_portable-prepare-compiled" {
                try await DecisionCommands.prepare(URL(fileURLWithPath: args[1]))
                return
            }
        }
        #endif
        throw PortableDecision.invalid("unsupported portable recovery operation")
    }

    func recover(_ load: CompiledFunctionLoadFailure, directory: URL, scratch: URL, package: DecisionPackage,
                 fetch: (String, URL, Int?) async throws(ServiceError) -> Data) async throws {
        try Task.checkCancellation()
        let data = try SmallFile.read(directory.appendingPathComponent(PortableDecision.receiptName), limit: DownloadLimit.manifestBytes)
        let receipt = try JSONDecoder().decode(PortableDecision.Receipt.self, from: data)
        let suffix = "." + package.architecture + ".aimodelc"
        guard receipt.inventory == package.manifest.sha256, receipt.architecture == package.architecture,
              load.asset.hasSuffix(suffix), (load.asset as NSString).lastPathComponent == load.asset,
              let row = receipt.assets.first(where: { $0.asset == load.asset }), row.functions.contains(load.function) else {
            throw PortableDecision.invalid("failed function is not in the verified AOT receipt")
        }
        try PortableDecision.auditAsset(row, in: directory, cache: cacheRoot(), elephant: receipt.elephant)
        let name = String(load.asset.dropLast(suffix.count)) + ".aimodel"
        let source = scratch.appendingPathComponent(name)
        let files = package.manifest.files.filter { $0.path.hasPrefix(name + "/") }
        guard files.contains(where: { $0.path == name + "/main.mlirb" && $0.sha256 == row.sourceHash }),
              files.contains(where: { $0.path == name + "/main.hash" }) else { throw PortableDecision.invalid("missing recovery source") }
        // Staged sources are released progressively during AOT. Materialize only the
        // affected asset from --from or the immutable remote commit on this exceptional path.
        for path in package.directories where path.hasPrefix(name + "/") {
            try FileManager.default.createDirectory(at: scratch.appendingPathComponent(path), withIntermediateDirectories: true)
        }
        for file in files {
            try Task.checkCancellation()
            _ = try await fetch(file.path, scratch, file.size)
        }
        guard files.allSatisfy({ Manifest.fileProblem($0, in: scratch, hash: true) == nil }),
              try PortableDecision.hexKey(source) == row.sourceHash else { throw PortableDecision.invalid("recovery source changed") }
        Log.info("Loading verified source \(name) entry \(load.function) after compiled loadFunction threw nilError")
        try await warmSource(source, load.function)
        try Task.checkCancellation()
        Log.info("Retrying ordinary compiled prepare at \(directory.path)")
        try await retryCompiled(directory)
    }
}
