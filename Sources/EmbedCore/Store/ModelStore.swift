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
public struct ModelSource: Sendable, Equatable {
    public var repository: String
    /// A commit hash, so the files behind it cannot change.
    public var revision: String
    /// SHA-256 of manifest.json at that revision. With it, everything downloaded is checked against a
    /// hash compiled into the binary rather than one fetched alongside the files.
    public var manifestSHA256: String?

    public init(repository: String, revision: String, manifestSHA256: String? = nil) {
        self.repository = repository
        self.revision = revision
        self.manifestSHA256 = manifestSHA256
    }

    /// The package this build installs: junmo-k/embeddinggemma-2-ane at the commit of its 0.1.0 upload, checked
    /// against the manifest hash below.
    public static let pinned = ModelSource(
        repository: "junmo-k/embeddinggemma-2-ane", revision: "3c9b6f2feeb640c5a359846659740cbf4d2debc7",
        manifestSHA256: "4773e00cd3282ada9ba3d5a723fd806ce8607167b1b8db553f1b5f1caabca52b")

    public var isPublished: Bool { revision != "UNPUBLISHED" && !revision.isEmpty }

    /// Whether `revision` can name a directory under models/: one plain path component of letters, digits, `.`, `_`
    /// and `-`, starting with a letter or digit (so never `.`, `..` or a hidden name), at most 64 characters.
    public static func isValidRevision(_ revision: String) -> Bool {
        isName(revision, maxLength: 64)
    }

    /// Whether `repository` is a Hugging Face repository id, `owner/name`, each part a plain name.
    public static func isValidRepository(_ repository: String) -> Bool {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { isName(String($0), maxLength: 96) && !$0.contains("..") }
    }

    static func isName(_ name: String, maxLength: Int) -> Bool {
        guard let first = name.unicodeScalars.first, name.unicodeScalars.count <= maxLength,
            first.isASCII, first.properties.isAlphabetic || ("0"..."9").contains(first)
        else { return false }
        return name.unicodeScalars.allSatisfy {
            $0.isASCII && ($0.properties.isAlphabetic || ("0"..."9").contains($0) || ".-_".unicodeScalars.contains($0))
        }
    }

    /// Fails for a repository or revision that `--repo` / `--revision` must not take.
    public func validate() throws(ServiceError) {
        guard Self.isValidRepository(repository) else {
            throw .badRequest("invalid_repo", "--repo \(repository): expected a Hugging Face repository id, owner/name", param: "repo")
        }
        guard Self.isValidRevision(revision) else {
            throw .badRequest("invalid_revision",
                              "--revision \(revision): expected a commit hash or a name of letters, digits, '.', '_' and '-'",
                              param: "revision")
        }
    }

    /// Hugging Face "resolve" URL of a file (a manifest path) at the revision; every part is percent-encoded.
    public func url(for path: String) throws(ServiceError) -> URL {
        try validate()
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repository)/resolve/\(revision)/\(path)"
        guard Manifest.isValidPath(path), let url = components.url else {
            throw ServiceError(.invalidInput, "invalid_manifest", "cannot build a download URL for \(path)")
        }
        return url
    }
}

/// File names inside a model directory (the layout of the published package, see its manifest.json).
public enum ModelLayout {
    public static let manifest = "manifest.json"
    public static let tokenizer = "tokenizer.json"
    public static let tokenTable = "embeddings.bf16"
    public static let textModel = "EmbeddingGemma2Text"
    public static let visionModel = "EmbeddingGemma2Vision"
    public static let positionTable = "position_embeddings.f16"
    public static let selfTest = "selftest.json"
    public static let selfTestData = "selftest.bin"
    public static let warmState = ".aneembed-warm.json"
    public static let selfTestState = ".aneembed-selftest.json"
    public static let packages = [textModel, visionModel]

    /// Compiled model when present, else the package (compiled at install/warm time).
    public static func model(_ name: String, in directory: URL) -> URL? {
        for suffix in [".mlmodelc", ".mlpackage"] {
            let url = directory.appendingPathComponent(name + suffix)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// Whether `path` (relative to the model directory) lies inside a compiled model (`*.mlmodelc/...`).
    /// Those files are produced on this Mac from the `.mlpackage`, not downloaded.
    public static func isCompiled(_ path: String) -> Bool {
        path.split(separator: "/").first?.hasSuffix(".mlmodelc") ?? false
    }
}

public struct ManifestEntry: Sendable, Equatable {
    public var path: String
    public var size: Int
    public var sha256: String
}

public struct Manifest: Sendable, Equatable {
    /// Package name (`name`; older drafts used `model`).
    public var model: String
    /// Revision recorded in the manifest, if any (the published one is pinned by commit, not by this field).
    public var revision: String
    public var created: String
    public var files: [ManifestEntry]
    /// SHA-256 of the manifest bytes: identifies a package exactly.
    public var sha256: String
    /// The bucket functions the package declares (`buckets`): length -> pool slots.
    public var buckets: [Int: Int]
    /// The long-text path (`long_path`): chunk, halo and sequence buckets, when the package has one.
    public var longPath: LongPathContract?

    public struct LongPathContract: Sendable, Equatable {
        public var chunk: Int
        public var halo: Int
        public var buckets: [Int]
    }

    /// Files to download or copy; compiled models are built on this Mac instead.
    public var sourceFiles: [ManifestEntry] { files.filter { !ModelLayout.isCompiled($0.path) } }

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
            guard let path = file["path"]?.stringValue, let size = file["size"]?.intValue, size >= 0,
                let sha = file["sha256"]?.stringValue, sha.count == 64, sha.allSatisfy(\.isHexDigit), isValidPath(path)
            else { throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has an invalid file entry") }
            entries.append(ManifestEntry(path: path, size: size, sha256: sha.lowercased()))
        }
        var buckets: [Int: Int] = [:]
        for bucket in root["buckets"]?.arrayValue ?? [] {
            if let tokens = bucket["tokens"]?.intValue, let slots = bucket["slots"]?.intValue { buckets[tokens] = slots }
        }
        var longPath: LongPathContract?
        if let long = root["long_path"], let chunk = long["chunk"]?.intValue, let halo = long["halo"]?.intValue {
            longPath = LongPathContract(chunk: chunk, halo: halo,
                                        buckets: long["buckets"]?.arrayValue?.compactMap(\.intValue) ?? [])
        }
        return Manifest(model: root["name"]?.stringValue ?? root["model"]?.stringValue ?? "",
                        revision: root["revision"]?.stringValue ?? "", created: root["created"]?.stringValue ?? "",
                        files: entries, sha256: SHA256Hasher.hex(data), buckets: buckets, longPath: longPath)
    }

    /// A relative path inside the package: non-empty components, none `.` or `..`, no control characters.
    public static func isValidPath(_ path: String) -> Bool {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.hasPrefix("/") && components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
            && !path.unicodeScalars.contains { $0.properties.generalCategory == .control }
    }

    public static func load(from directory: URL) throws(ServiceError) -> Manifest {
        let url = directory.appendingPathComponent(ModelLayout.manifest)
        guard let data = try? Data(contentsOf: url) else {
            throw ServiceError(.unavailable, "manifest_missing", "no manifest.json in \(directory.path)",
                               hint: "install with `aneembed models install` or point --model-dir at a model package")
        }
        return try parse(data)
    }

    /// Fails when the package declares a different contract than this build implements.
    public func checkContract() throws(ServiceError) {
        if !buckets.isEmpty {
            for (length, slots) in buckets where BucketContract.lengths.contains(length) == false
                || BucketContract.slots(forLength: length) != slots
            {
                throw ServiceError(.unavailable, "unsupported_model",
                                   "the package declares bucket \(length) with \(slots) slots; this build expects \(BucketContract.lengths) with max(8, S/32) slots",
                                   hint: "update aneembed")
            }
        }
        if let longPath, longPath.chunk != LongContract.chunk || longPath.halo != LongContract.halo
            || !longPath.buckets.allSatisfy(LongContract.lengths.contains)
        {
            throw ServiceError(.unavailable, "unsupported_model",
                               "the package's long path (chunk \(longPath.chunk), halo \(longPath.halo), \(longPath.buckets)) differs from this build's",
                               hint: "update aneembed")
        }
    }

    public struct Problem: Sendable, Equatable {
        public var path: String
        public var issue: String  // missing | not_regular_file | size_mismatch | sha256_mismatch
    }

    /// A regular file itself, not a symlink, FIFO, socket or device (those could hang the hash or point elsewhere).
    public static func isRegularFile(_ url: URL) -> Bool {
        var info = stat()
        return lstat(url.path, &info) == 0 && (info.st_mode & S_IFMT) == S_IFREG
    }

    /// Checks every file; `hash: false` only checks presence and size (fast, for doctor).
    ///
    /// Files of a compiled model (`X.mlmodelc/...`) are checked only when that compiled model exists:
    /// they are built on this Mac from `X.mlpackage`, and the Core ML compiler of another macOS release
    /// may produce different bytes (the load-time self-test is what decides whether they compute correctly).
    /// Their problems come back separately in `compiled`.
    public func verify(in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil) -> [Problem] {
        verifyAll(in: directory, hash: hash, progress: progress).source
    }

    public func verifyAll(in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil)
        -> (source: [Problem], compiled: [Problem])
    {
        var source: [Problem] = []
        var compiled: [Problem] = []
        for file in files {
            let isCompiled = ModelLayout.isCompiled(file.path)
            if isCompiled {
                let root = String(file.path.split(separator: "/")[0])
                guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(root).path) else { continue }
            }
            let url = directory.appendingPathComponent(file.path)
            var problem: Problem?
            if let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = (attributes[.size] as? NSNumber)?.intValue
            {
                if !Self.isRegularFile(url) {
                    problem = Problem(path: file.path, issue: "not_regular_file")
                } else if size != file.size {
                    problem = Problem(path: file.path, issue: "size_mismatch")
                } else if hash {
                    progress?(file.path)
                    if (try? SHA256Hasher.hex(fileAt: url)) != file.sha256 {
                        problem = Problem(path: file.path, issue: "sha256_mismatch")
                    }
                }
            } else {
                problem = Problem(path: file.path, issue: "missing")
            }
            if let problem {
                if isCompiled { compiled.append(problem) } else { source.append(problem) }
            }
        }
        return (source, compiled)
    }
}

/// What `models warm` compiled, so a later start knows whether the OS cache is likely still warm.
///
/// The OS keeps the Neural Engine compile per model path, macOS build and executable: the same binary
/// copied to another path compiles again (measured), so an upgraded aneembed recompiles once too.
public struct WarmState: Sendable, Equatable {
    public var osBuild: String
    public var binaryVersion: String
    /// SHA-256 of the executable that warmed (empty in states written before it was recorded).
    public var binary: String
    public var maxTokens: Int
    public var functions: [String]
    public var warmedAt: Date
    public var seconds: Double

    public init(osBuild: String, binaryVersion: String, binary: String = Executable.identity, maxTokens: Int,
                functions: [String], warmedAt: Date, seconds: Double)
    {
        self.osBuild = osBuild
        self.binaryVersion = binaryVersion
        self.binary = binary
        self.maxTokens = maxTokens
        self.functions = functions
        self.warmedAt = warmedAt
        self.seconds = seconds
    }

    public var json: JSONValue {
        [
            "os_build": .string(osBuild), "binary_version": .string(binaryVersion), "binary_sha256": .string(binary),
            "max_tokens": .int(maxTokens), "functions": .array(functions.map(JSONValue.string)),
            "warmed_at": .string(ISO8601DateFormatter().string(from: warmedAt)), "seconds": .number(seconds),
        ]
    }

    public static func load(from directory: URL) -> WarmState? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(ModelLayout.warmState)),
            let root = try? JSONValue.parse(data), let build = root["os_build"]?.stringValue,
            let maxTokens = root["max_tokens"]?.intValue
        else { return nil }
        return WarmState(
            osBuild: build, binaryVersion: root["binary_version"]?.stringValue ?? "",
            binary: root["binary_sha256"]?.stringValue ?? "", maxTokens: maxTokens,
            functions: root["functions"]?.arrayValue?.compactMap(\.stringValue) ?? [],
            warmedAt: root["warmed_at"]?.stringValue.flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast,
            seconds: root["seconds"]?.doubleValue ?? 0)
    }

    public func save(to directory: URL) throws {
        try (json.render(as: .json) + "\n").write(to: directory.appendingPathComponent(ModelLayout.warmState),
                                                  atomically: true, encoding: .utf8)
    }

    /// Fresh when this executable warmed on this OS build for at least `maxTokens`.
    public func covers(osBuild: String, maxTokens: Int, binary: String = Executable.identity) -> Bool {
        self.osBuild == osBuild && self.maxTokens >= maxTokens && self.binary == binary
    }
}

/// The running executable, identified by content.
public enum Executable {
    public static let identity: String = {
        guard let path = Bundle.main.executablePath, let hash = try? SHA256Hasher.hex(fileAt: URL(fileURLWithPath: path))
        else { return "" }
        return hash
    }()
}

public struct ModelStore: Sendable {
    public let dataDirectory: URL

    public init(dataDirectory: URL) {
        self.dataDirectory = dataDirectory
    }

    public var modelsDirectory: URL { dataDirectory.appendingPathComponent("models", isDirectory: true) }
    var currentPointer: URL { modelsDirectory.appendingPathComponent("current") }

    /// The active revision directory (may not exist yet). A `current` that is not a valid revision name is ignored.
    public var currentDirectory: URL {
        if let name = try? String(contentsOf: currentPointer, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), ModelSource.isValidRevision(name)
        {
            return modelsDirectory.appendingPathComponent(name, isDirectory: true)
        }
        return modelsDirectory.appendingPathComponent(ModelSource.pinned.revision, isDirectory: true)
    }

    /// The directory of `revision`, which must be a valid revision name (it is removed and replaced on install).
    public func directory(for revision: String) throws(ServiceError) -> URL {
        guard ModelSource.isValidRevision(revision) else {
            throw .badRequest("invalid_revision", "\(revision) is not a valid revision name", param: "revision")
        }
        let url = modelsDirectory.appendingPathComponent(revision, isDirectory: true)
        try checkInside(url)
        return url
    }

    /// Fails unless `url` is a direct child of the models directory once symlinks and `..` are resolved: the
    /// installer removes and moves such paths.
    public func checkInside(_ url: URL) throws(ServiceError) {
        let parent = url.deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath().path
        let models = modelsDirectory.standardizedFileURL.resolvingSymlinksInPath().path
        guard parent == models, ModelSource.isName(String(url.lastPathComponent.drop { $0 == "." }), maxLength: 160) else {
            throw ServiceError(.internal, "outside_models_directory", "refusing to replace \(url.path): not inside \(models)")
        }
    }

    /// Points `current` at `revision` (write to a temporary file, then rename: atomic).
    public func activate(revision: String) throws {
        guard ModelSource.isValidRevision(revision) else {
            throw ServiceError.badRequest("invalid_revision", "\(revision) is not a valid revision name", param: "revision")
        }
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
