// `aneembed doctor`: what this Mac and model directory can do, before anything fails. Fast by default
// (presence and sizes); --verify also hashes every file against the manifest.
import Foundation

public enum Doctor {
    public static func report(_ config: Config, verify: Bool = false) -> JSONValue {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let engine = NeuralEngine.device
        let directory = config.modelDirectory
        let exists = FileManager.default.fileExists(atPath: directory.path)
        var problems: [JSONValue] = []

        var model: JSONObject = ["dir": .string(directory.path), "present": .bool(exists)]
        if exists {
            let files: [(String, Bool)] = [
                ("text_model", ModelLayout.model(ModelLayout.textModel, in: directory) != nil),
                ("tokenizer", FileManager.default.fileExists(atPath: directory.appendingPathComponent(ModelLayout.tokenizer).path)),
                ("token_table", FileManager.default.fileExists(atPath: directory.appendingPathComponent(ModelLayout.tokenTable).path)),
                ("vision_model", ModelLayout.model(ModelLayout.visionModel, in: directory) != nil),
            ]
            model["files"] = .object(JSONObject(files.map { (key: $0.0, value: .string($0.1 ? "present" : "missing")) }))
            for (name, present) in files.prefix(3) where !present { problems.append(.string("model file missing: \(name)")) }
            do {
                let manifest = try Manifest.load(from: directory)
                let issues = manifest.verify(in: directory, hash: verify)
                model["revision"] = .string(manifest.revision)
                model["manifest"] = [
                    "files": .int(manifest.files.count), "checked": .string(verify ? "sha256" : "size"),
                    "problems": .array(issues.map { .string("\($0.path): \($0.issue)") }),
                ]
                if !issues.isEmpty { problems.append(.string("\(issues.count) model files fail the manifest check")) }
            } catch {
                model["manifest"] = .string("missing")
            }
        } else if !config.useFakeRuntime {
            problems.append(.string("no model: run `aneembed models install` or pass --model-dir"))
        }

        let cache: String
        if let warm = WarmState.load(from: directory) {
            if warm.osBuild != operatingSystemBuild {
                cache = "stale: warmed on macOS build \(warm.osBuild), now \(operatingSystemBuild)"
            } else if warm.maxTokens < config.maxTokens {
                cache = "partial: warmed up to \(warm.maxTokens) tokens, configured \(config.maxTokens)"
            } else {
                cache = "warm"
            }
        } else {
            cache = "cold: never warmed (the first start compiles for the Neural Engine)"
        }
        if cache != "warm" && exists { problems.append(.string("run `aneembed models warm` (compile cache \(cache.prefix { $0 != ":" }))")) }
        if !engine.available && !config.useFakeRuntime { problems.append(.string("no Neural Engine")) }

        return [
            "aneembed": .string(aneembedVersion),
            "macos": .string("\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"),
            "macos_build": .string(operatingSystemBuild),
            "neural_engine": engine.available ? .string("available (\(engine.cores) cores)") : "unavailable",
            "runtime": .string(config.string("runtime")),
            "max_tokens": .int(config.maxTokens),
            "model": .object(model),
            "compile_cache": .string(cache),
            "ready": .bool(problems.isEmpty),
            "problems": .array(problems),
        ]
    }
}
