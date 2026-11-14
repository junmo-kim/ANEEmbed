// Model files on disk.
//
//   <data-dir>/models/<revision>/          one directory per installed revision, never moved: the OS keys its
//                                          ANE compile cache on the model path, so a moved model recompiles
//   <data-dir>/models/current              text file naming the active revision directory; switched
//                                          atomically after a new revision is downloaded, verified and warmed
//   <model-dir>/manifest.json              files with sizes and SHA-256 (written by the packaging tools)
//   <model-dir>/.aneembed-warm.json        what was last warmed, on which OS build (doctor, serve)
import Foundation

/// Where the published model package comes from.
/// TODO(model): fill in once the package is published; until then `models install` reports
/// `model_not_published` and `--model-dir` points at a local package.
public struct ModelSource: Sendable, Equatable {
    public var repository: String
    public var revision: String

    public static let pinned = ModelSource(repository: "junmo-kim/embeddinggemma-2-ane", revision: "UNPUBLISHED")

    public var isPublished: Bool { revision != "UNPUBLISHED" && !revision.isEmpty }

    /// Hugging Face "resolve" URL of a file at the pinned revision.
    public func url(for path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(path)")!
    }
}

/// File names inside a model directory.
/// TODO(model): final names come with the package; keep every lookup going through here.
public enum ModelLayout {
    public static let manifest = "manifest.json"
    public static let tokenizer = "tokenizer.json"
    public static let tokenTable = "embeddings.bf16"
    public static let textModel = "EmbeddingGemma2Text"
    public static let visionModel = "EmbeddingGemma2Vision"
    public static let positionTable = "position_embeddings.f16"
    public static let warmState = ".aneembed-warm.json"

    /// Compiled model when present, else the package (compiled at install/warm time).
    public static func model(_ name: String, in directory: URL) -> URL? {
        for suffix in [".mlmodelc", ".mlpackage"] {
            let url = directory.appendingPathComponent(name + suffix)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}

public struct ManifestEntry: Sendable, Equatable {
    public var path: String
    public var size: Int
    public var sha256: String
}

public struct Manifest: Sendable, Equatable {
    public var model: String
    public var revision: String
    public var files: [ManifestEntry]

    public static func parse(_ data: Data) throws(ServiceError) -> Manifest {
        let root: JSONValue
        do { root = try JSONValue.parse(data) } catch {
            throw ServiceError(.unavailable, "invalid_manifest", "manifest.json is not valid JSON: \(error.description)")
        }
        guard let files = root["files"]?.arrayValue else {
            throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has no files list")
        }
        var entries: [ManifestEntry] = []
        for file in files {
            guard let path = file["path"]?.stringValue, let size = file["size"]?.intValue,
                let sha = file["sha256"]?.stringValue, sha.count == 64,
                !path.hasPrefix("/"), !path.split(separator: "/").contains("..")
            else { throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has an invalid file entry") }
            entries.append(ManifestEntry(path: path, size: size, sha256: sha.lowercased()))
        }
        return Manifest(model: root["model"]?.stringValue ?? "", revision: root["revision"]?.stringValue ?? "", files: entries)
    }

    public static func load(from directory: URL) throws(ServiceError) -> Manifest {
        let url = directory.appendingPathComponent(ModelLayout.manifest)
        guard let data = try? Data(contentsOf: url) else {
            throw ServiceError(.unavailable, "manifest_missing", "no manifest.json in \(directory.path)",
                               hint: "install with `aneembed models install` or point --model-dir at a model package")
        }
        return try parse(data)
    }

    public struct Problem: Sendable, Equatable {
        public var path: String
        public var issue: String  // missing | size_mismatch | sha256_mismatch
    }

    /// Checks every file; `hash: false` only checks presence and size (fast, for doctor).
    public func verify(in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil) -> [Problem] {
        var problems: [Problem] = []
        for file in files {
            let url = directory.appendingPathComponent(file.path)
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = (attributes[.size] as? NSNumber)?.intValue
            else {
                problems.append(Problem(path: file.path, issue: "missing"))
                continue
            }
            guard size == file.size else {
                problems.append(Problem(path: file.path, issue: "size_mismatch"))
                continue
            }
            if hash {
                progress?(file.path)
                if (try? SHA256Hasher.hex(fileAt: url)) != file.sha256 {
                    problems.append(Problem(path: file.path, issue: "sha256_mismatch"))
                }
            }
        }
        return problems
    }
}

/// What `models warm` compiled, so a later start knows whether the OS cache is likely still warm.
public struct WarmState: Sendable, Equatable {
    public var osBuild: String
    public var binaryVersion: String
    public var maxTokens: Int
    public var functions: [String]
    public var warmedAt: Date
    public var seconds: Double

    public init(osBuild: String, binaryVersion: String, maxTokens: Int, functions: [String], warmedAt: Date, seconds: Double) {
        self.osBuild = osBuild
        self.binaryVersion = binaryVersion
        self.maxTokens = maxTokens
        self.functions = functions
        self.warmedAt = warmedAt
        self.seconds = seconds
    }

    public var json: JSONValue {
        [
            "os_build": .string(osBuild), "binary_version": .string(binaryVersion), "max_tokens": .int(maxTokens),
            "functions": .array(functions.map(JSONValue.string)),
            "warmed_at": .string(ISO8601DateFormatter().string(from: warmedAt)), "seconds": .number(seconds),
        ]
    }

    public static func load(from directory: URL) -> WarmState? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(ModelLayout.warmState)),
            let root = try? JSONValue.parse(data), let build = root["os_build"]?.stringValue,
            let maxTokens = root["max_tokens"]?.intValue
        else { return nil }
        return WarmState(
            osBuild: build, binaryVersion: root["binary_version"]?.stringValue ?? "", maxTokens: maxTokens,
            functions: root["functions"]?.arrayValue?.compactMap(\.stringValue) ?? [],
            warmedAt: root["warmed_at"]?.stringValue.flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast,
            seconds: root["seconds"]?.doubleValue ?? 0)
    }

    public func save(to directory: URL) throws {
        try (json.render(as: .json) + "\n").write(to: directory.appendingPathComponent(ModelLayout.warmState),
                                                  atomically: true, encoding: .utf8)
    }

    /// Fresh when warmed on this OS build for at least `maxTokens`.
    public func covers(osBuild: String, maxTokens: Int) -> Bool {
        self.osBuild == osBuild && self.maxTokens >= maxTokens
    }
}

public struct ModelStore: Sendable {
    public let dataDirectory: URL

    public init(dataDirectory: URL) {
        self.dataDirectory = dataDirectory
    }

    public var modelsDirectory: URL { dataDirectory.appendingPathComponent("models", isDirectory: true) }
    var currentPointer: URL { modelsDirectory.appendingPathComponent("current") }

    /// The active revision directory (may not exist yet).
    public var currentDirectory: URL {
        if let name = try? String(contentsOf: currentPointer, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty, !name.contains("/")
        {
            return modelsDirectory.appendingPathComponent(name, isDirectory: true)
        }
        return modelsDirectory.appendingPathComponent(ModelSource.pinned.revision, isDirectory: true)
    }

    public func directory(for revision: String) -> URL {
        modelsDirectory.appendingPathComponent(revision, isDirectory: true)
    }

    /// Points `current` at `revision` (write to a temporary file, then rename: atomic).
    public func activate(revision: String) throws {
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        let temporary = modelsDirectory.appendingPathComponent(".current.\(ProcessInfo.processInfo.processIdentifier)")
        try (revision + "\n").write(to: temporary, atomically: false, encoding: .utf8)
        if rename(temporary.path, currentPointer.path) != 0 {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: currentPointer.path])
        }
    }
}

/// The running macOS build, e.g. "26A434".
public var operatingSystemBuild: String {
    var size = 0
    sysctlbyname("kern.osversion", nil, &size, nil, 0)
    var buffer = [UInt8](repeating: 0, count: max(size, 1))
    sysctlbyname("kern.osversion", &buffer, &size, nil, 0)
    return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
}
