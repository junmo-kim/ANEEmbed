import CoreML
import DecisionRuntime
import EmbedCore
import Foundation

@available(macOS 26.0, *)
enum CoreMLDecisionInstall {
    private struct Receipt: Codable {
        let source: String
        let installed: String
    }
    private static let receiptName = ".decision-coreml.json"

    static func install(_ source: URL, store: ModelStore,
                        prepare: (URL) async throws -> Void) async throws -> URL {
        try CoreMLDecisionCapability.requireHardware()
        let manager = FileManager.default
        let lock = try ModelsCommand.lock(store)
        defer { lock.release() }
        let fingerprint = try DecisionInstall.fingerprint(source)
        let preferred = try store.directory(for: "local-" + fingerprint.prefix(32))
        ModelInstaller.removeStaleStaging(for: preferred)
        return try await DecisionInstall.publish(preferred, store: store, matches: { directory in
            guard let data = try? SmallFile.read(directory.appendingPathComponent(receiptName), limit: 1024),
                  let receipt = try? JSONDecoder().decode(Receipt.self, from: data),
                  receipt.source == fingerprint else { return false }
            return (try? DecisionInstall.fingerprint(directory, excludingRootFile: receiptName)) == receipt.installed
        }, stage: { stage in
            try manager.copyItem(at: source, to: stage)
            guard try DecisionInstall.fingerprint(stage) == fingerprint else {
                throw ServiceError(.invalidInput, "model_files_changed", "Core ML source changed during staging")
            }
            let descriptor = stage.appendingPathComponent("model.json")
            guard var model = try JSONSerialization.jsonObject(with: SmallFile.read(descriptor, limit: DownloadLimit.manifestBytes)) as? [String: Any],
                  let bindings = model["coreml_functions"] as? [String: String], !bindings.isEmpty,
                  !manager.fileExists(atPath: stage.appendingPathComponent(receiptName).path),
                  bindings.allSatisfy({ key, path in !key.isEmpty && path == key + ".mlpackage" && (path as NSString).lastPathComponent == path })
            else { throw ServiceError(.invalidInput, "invalid_bundle", "expected flat per-function Core ML source packages") }
            model["coreml_functions"] = bindings.mapValues { ($0 as NSString).deletingPathExtension + ".mlmodelc" }
            try JSONSerialization.data(withJSONObject: model, options: .sortedKeys).write(to: descriptor, options: .atomic)
            try DecisionEngine.validateManifest(at: stage)
            // A rejected recurrent scan cannot serve any request. Check it before compiling the other tiers.
            let first = model["coreml_recurrent_layout"] == nil ? "scan0" : "valenDeltaFirst"
            let names = [first] + bindings.keys.filter { $0 != first }.sorted()
            for key in names {
                guard let name = bindings[key] else { throw ServiceError(.invalidInput, "invalid_bundle", "missing " + key) }
                try Task.checkCancellation()
                let package = stage.appendingPathComponent(name)
                let compiled = try await MLModel.compileModel(at: package)
                defer { try? manager.removeItem(at: compiled) }
                _ = try await CoreMLDecisionCapability.plannedOperations(at: compiled)
                try manager.moveItem(at: compiled, to: stage.appendingPathComponent(key + ".mlmodelc"))
                try manager.removeItem(at: package)
            }
            let receipt = try Receipt(source: fingerprint, installed: DecisionInstall.fingerprint(stage))
            try JSONEncoder().encode(receipt).write(to: stage.appendingPathComponent(receiptName), options: .withoutOverwriting)
        }, prepare: prepare)
    }
}
