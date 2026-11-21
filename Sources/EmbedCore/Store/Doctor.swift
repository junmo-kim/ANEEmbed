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
            var loadedManifest: Manifest?
            do {
                let manifest = try Manifest.load(from: directory)
                loadedManifest = manifest
                let (issues, compiled) = manifest.verifyAll(in: directory, hash: verify)
                model["name"] = .string(manifest.model)
                model["manifest_sha256"] = .string(manifest.sha256)
                model["manifest"] = [
                    "files": .int(manifest.sourceFiles.count), "checked": .string(verify ? "sha256" : "size"),
                    "problems": .array(issues.map { .string("\($0.path): \($0.issue)") }),
                ]
                if verify {
                    // Informational: another Core ML compiler stamps its version into model.mil.
                    model["compiled_differs_from_packager"] = .array(compiled.map { .string("\($0.path): \($0.issue)") })
                }
                do { try manifest.checkPinned(config.pinnedManifestSHA256) } catch {
                    problems.append(.string(error.message + (error.hint.map { "; \($0)" } ?? "")))
                }
                if !issues.isEmpty { problems.append(.string("\(issues.count) model files fail the manifest check")) }
                for item in manifest.compiledStatus(in: directory) where item.status.hasPrefix("incomplete") {
                    problems.append(.string("\(item.package).mlmodelc is \(item.status)"))
                }
                do { try manifest.checkContract() } catch { problems.append(.string(error.message)) }
            } catch {
                model["manifest"] = .string(error.reason == "manifest_missing" ? "missing" : "invalid")
                problems.append(.string(error.message))
            }
            if config.skipSelfTest {
                model["selftest_data"] = "not used (--selftest skip: every function is served unchecked)"
            } else if !config.useFakeRuntime, let loadedManifest {
                do {
                    let suite = try SelfTestSuite.require(from: directory, manifest: loadedManifest)
                    model["selftest_data"] = .string("\(suite.cases.count) tests")
                } catch {
                    model["selftest_data"] = .string(error.reason)
                    problems.append(.string(error.message))
                }
            } else if !config.useFakeRuntime {
                model["selftest_data"] = "not checked: no valid manifest.json to check selftest.bin against"
            }
        } else if !config.useFakeRuntime {
            problems.append(.string("no model: run `aneembed models install` or pass --model-dir"))
        }

        let cache: String
        if let warm = WarmState.load(from: directory) {
            if warm.osBuild != operatingSystemBuild {
                cache = "stale: warmed on macOS build \(warm.osBuild), now \(operatingSystemBuild)"
            } else if let manifest = try? Manifest.load(from: directory), warm.manifestSHA256 != manifest.sha256 {
                cache = "stale: warmed for another model package (manifest sha256 \(warm.manifestSHA256.prefix(12))); the next start compiles again"
            } else if warm.binary != Executable.identity {
                cache = "stale: warmed by another aneembed binary (\(warm.binaryVersion)); the first start compiles again"
            } else if warm.maxTokens < config.maxTokens {
                cache = "partial: warmed up to \(warm.maxTokens) tokens, configured \(config.maxTokens)"
            } else {
                cache = "warm"
            }
        } else {
            cache = "cold: never warmed (the first start compiles for the Neural Engine)"
        }
        if cache != "warm" && exists { problems.append(.string("run `aneembed models warm` (compile cache \(cache.prefix { $0 != ":" }))")) }

        // The self-test of the last load (serve, mcp, models warm/install): functions that failed are not served.
        var selfTest: JSONValue = "not run yet"
        if let report = SelfTestReport.load(from: directory) {
            var object: JSONObject = ["summary": .string(report.summary), "os_build": .string(report.osBuild),
                                      "ran_at": .string(ISO8601DateFormatter().string(from: report.ranAt))]
            if !report.failed.isEmpty {
                object["failed"] = .array(report.failed.map(\.json))
                problems.append(.string("self-test failed for \(report.failed.map(\.function).joined(separator: ", ")); "
                    + "those functions are not served (please report the macOS build)"))
            }
            if report.osBuild != operatingSystemBuild { object["note"] = "ran on another macOS build; the next load runs it again" }
            selfTest = .object(object)
        }
        if !engine.available && !config.useFakeRuntime { problems.append(.string("no Neural Engine")) }

        return [
            "aneembed": .string(aneembedVersion),
            "macos": .string("\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"),
            "macos_build": .string(operatingSystemBuild),
            "neural_engine": engine.available ? .string("available (\(engine.cores) cores)") : "unavailable",
            "runtime": .string(config.string("runtime")),
            "max_tokens": .int(config.maxTokens),
            // What a load with this --max-tokens compiles and self-tests (long functions only above 2048).
            "text_functions": .array((CoreMLBucketRunner.lengths(forMaxTokens: config.maxTokens).map(BucketContract.functionName)
                + WindowContract.longLengths(forMaxTokens: config.maxTokens).map(WindowContract.longName)).map(JSONValue.string)),
            "model": .object(model),
            "compile_cache": .string(cache),
            "selftest": selfTest,
            "ready": .bool(problems.isEmpty),
            "problems": .array(problems),
        ]
    }
}
