import Foundation
import Testing

@testable import EmbedCore

@Suite("Config")
struct ConfigTests {
    @Test func defaultsEnvironmentAndFlags() throws {
        let defaults = try Config(environment: [:])
        #expect(defaults.port == 18180 && defaults.host == "127.0.0.1" && defaults.maxTokens == 2048)
        #expect(defaults.sources["port"] == .default)

        let env = try Config(environment: ["ANEEMBED_PORT": "9000", "ANEEMBED_MAX_TOKENS": "8192"])
        #expect(env.port == 9000 && env.maxTokens == 8192 && env.sources["port"] == .environment)

        let flag = try Config(flags: ["port": "9001"], environment: ["ANEEMBED_PORT": "9000"])
        #expect(flag.port == 9001 && flag.sources["port"] == .flag)
    }

    @Test func everyOptionHasTheSameNameAsFlagAndEnvironmentVariable() {
        for option in Config.options {
            #expect(option.environmentName == "ANEEMBED_" + option.name.uppercased().replacingOccurrences(of: "-", with: "_"))
        }
    }

    @Test func rejectsInvalidValues() {
        #expect(throws: ServiceError.self) { try Config(flags: ["max-tokens": "9000"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(flags: ["port": "x"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(flags: ["nope": "1"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(environment: ["ANEEMBED_RUNTIME": "gpu"]) }
    }

    @Test func modelDirectoryFollowsTheDataDirectory() throws {
        let config = try Config(flags: ["data-dir": "/tmp/x"], environment: [:])
        #expect(config.modelDirectory.path.hasPrefix("/tmp/x/models/"))
        let explicit = try Config(flags: ["model-dir": "/opt/m"], environment: [:])
        #expect(explicit.modelDirectory.path == "/opt/m")
        #expect(config.printable["port"]?["source"] == "default")
        #expect(config.printable["data-dir"]?["env"] == "ANEEMBED_DATA_DIR")
    }
}

@Suite("Model store")
struct ModelStoreTests {
    func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func verifiesFilesAgainstTheManifest() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("a.bin"))
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("m.mlmodelc"), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: directory.appendingPathComponent("m.mlmodelc/model.mil"))
        let manifest: JSONValue = [
            "model": "embeddinggemma-2", "revision": "r1",
            "files": [
                ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
                ["path": "m.mlmodelc/model.mil", "size": 1, "sha256": .string(String(repeating: "0", count: 64))],
                ["path": "missing.bin", "size": 1, "sha256": .string(String(repeating: "0", count: 64))],
            ],
        ]
        try manifest.jsonData().write(to: directory.appendingPathComponent("manifest.json"))
        let loaded = try Manifest.load(from: directory)
        #expect(loaded.revision == "r1" && loaded.files.count == 3)
        #expect(loaded.sourceFiles.map(\.path) == ["a.bin", "missing.bin"])
        #expect(loaded.verify(in: directory, hash: false) == [.init(path: "missing.bin", issue: "missing")])
        // Compiled files are built on this Mac: reported apart, and only when the compiled model exists.
        let checked = loaded.verifyAll(in: directory, hash: true)
        #expect(checked.source == [.init(path: "missing.bin", issue: "missing")])
        #expect(checked.compiled == [.init(path: "m.mlmodelc/model.mil", issue: "sha256_mismatch")])
        try FileManager.default.removeItem(at: directory.appendingPathComponent("m.mlmodelc"))
        #expect(loaded.verifyAll(in: directory, hash: true).compiled.isEmpty)
    }

    /// "Already installed" means this exact package: the manifest must hash to the expected value (pinned in the
    /// binary, or fetched for the requested source), not just match the files next to it.
    @Test func anInstalledPackageMustMatchTheExpectedManifest() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("a.bin"))
        let manifest: JSONValue = ["files": [["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))]]]
        let data = manifest.jsonData()
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        #expect(ModelInstaller.isInstalled(at: directory, manifestSHA256: SHA256Hasher.hex(data)))
        // A self-consistent package that is not the expected one is not "installed".
        #expect(!ModelInstaller.isInstalled(at: directory, manifestSHA256: ModelSource.pinned.manifestSHA256!))
        try Data("Weights".utf8).write(to: directory.appendingPathComponent("a.bin"))
        #expect(!ModelInstaller.isInstalled(at: directory, manifestSHA256: SHA256Hasher.hex(data)))
        #expect(!ModelInstaller.isInstalled(at: directory.appendingPathComponent("none"), manifestSHA256: SHA256Hasher.hex(data)))
    }

    /// Only regular files are hashed or copied: a symlink (to /dev/zero, say) or a FIFO would hang the hash.
    @Test func packageFilesMustBeRegularFiles() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("pkg")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("real.bin"))
        try FileManager.default.createSymbolicLink(atPath: source.appendingPathComponent("a.bin").path,
                                                   withDestinationPath: directory.appendingPathComponent("real.bin").path)
        #expect(mkfifo(source.appendingPathComponent("f.bin").path, 0o600) == 0)
        let manifest: JSONValue = ["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
            ["path": "f.bin", "size": 0, "sha256": .string(SHA256Hasher.hex(Data()))],
        ]]
        try manifest.jsonData().write(to: source.appendingPathComponent("manifest.json"))
        let loaded = try Manifest.load(from: source)
        #expect(loaded.verify(in: source, hash: true) == [.init(path: "a.bin", issue: "not_regular_file"),
                                                         .init(path: "f.bin", issue: "not_regular_file")])
        let store = ModelStore(dataDirectory: directory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        #expect(throws: ServiceError.self) {
            try ModelInstaller.copy(from: source, into: store.directory(for: "local-x"), store: store)
        }
        #expect(!FileManager.default.fileExists(atPath: store.modelsDirectory.appendingPathComponent("local-x").path))
    }

    /// At load: every source file present with its manifest size, and tokenizer.json and the self-test files
    /// (which the self-test cannot check itself) hashed. The full hash pass stays in `models verify`.
    @Test func loadChecksSizesAndTheFilesTheSelfTestCannotSee() throws {
        let directory = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try Manifest.load(from: directory)
        #expect(manifest.checkForLoad(in: directory).isEmpty)
        try Data("[]".utf8).write(to: directory.appendingPathComponent(ModelLayout.tokenizer))  // same size, other bytes
        try Data(count: 4).write(to: directory.appendingPathComponent(ModelLayout.tokenTable))
        #expect(manifest.checkForLoad(in: directory) == [.init(path: ModelLayout.tokenTable, issue: "size_mismatch"),
                                                         .init(path: ModelLayout.tokenizer, issue: "sha256_mismatch")])
        var unlisted = manifest
        unlisted.files.removeAll { $0.path == ModelLayout.tokenizer }
        #expect(unlisted.checkForLoad(in: directory).contains(.init(path: ModelLayout.tokenizer, issue: "not_in_manifest")))
    }

    /// Two installers never share a staging directory or race on the revision directory: an exclusive lock on
    /// models/.install.lock, a staging directory per process, and an atomic swap into place.
    @Test func installsAreSerializedAndSwappedAtomically() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        let first = try InstallLock.acquire(in: store)
        #expect(throws: ServiceError.self) { try InstallLock.acquire(in: store, wait: false) }
        first.release()
        let second = try InstallLock.acquire(in: store, wait: false)
        second.release()

        let target = try store.directory(for: "r1")
        let staging = ModelInstaller.stagingDirectory(for: target)
        #expect(staging != ModelInstaller.stagingDirectory(for: target))
        #expect(throws: Never.self) { try store.checkInside(staging) }
        // A target in use is replaced by one rename (swap), and the old contents go away.
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: target.appendingPathComponent("a.bin"))
        let payload = Data("new".utf8)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try payload.write(to: staging.appendingPathComponent("a.bin"))
        let manifest = try Manifest.parse(JSONValue.object(["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
        ]]).jsonData())
        try ModelInstaller.finish(staging: staging, into: target, store: store, manifest: manifest, log: { _ in })
        #expect(try Data(contentsOf: target.appendingPathComponent("a.bin")) == payload)
        #expect(!FileManager.default.fileExists(atPath: staging.path))
        // Staging left by an interrupted install is removed by the next one.
        let stale = ModelInstaller.stagingDirectory(for: target)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        ModelInstaller.removeStaleStaging(for: target)
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(FileManager.default.fileExists(atPath: target.path))
    }

    /// Whatever fails once the staging directory exists (the manifest download, the manifest's pinned hash, a file it
    /// lists), the staging directory is removed and nothing is installed.
    @Test func aFailedDownloadLeavesNoStagingDirectory() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        let target = try store.directory(for: "r1")
        let payload = Data("weights".utf8)
        let manifest = JSONValue.object(["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
        ]]).jsonData()
        let pinned = ModelSource(repository: "owner/name", revision: "r1", manifestSHA256: SHA256Hasher.hex(manifest))
        let otherPin = ModelSource(repository: "owner/name", revision: "r1", manifestSHA256: String(repeating: "0", count: 64))

        /// Downloads through a stand-in for the network that fails on `failing`; returns the error.
        func download(_ source: ModelSource, failing: String?) async -> ServiceError? {
            do {
                try await ModelInstaller.download(source, into: target, store: store) { path, staging, _ throws(ServiceError) in
                    guard path != failing else { throw ServiceError(.unavailable, "download_failed", "\(path): offline") }
                    let data = path == ModelLayout.manifest ? manifest : payload
                    do { try data.write(to: staging.appendingPathComponent(path)) } catch {
                        throw ServiceError(.internal, "write_failed", "\(error)")
                    }
                    return path == ModelLayout.manifest ? data : Data()
                }
                return nil
            } catch {
                return error
            }
        }
        func staged() throws -> [String] {
            try FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path).filter { $0.contains(".partial") }
        }

        #expect(await download(pinned, failing: ModelLayout.manifest)?.reason == "download_failed")
        #expect(try staged().isEmpty)
        #expect(await download(otherPin, failing: nil)?.reason == "checksum_mismatch")
        #expect(try staged().isEmpty)
        #expect(await download(pinned, failing: "a.bin")?.reason == "download_failed")
        #expect(try staged().isEmpty && !FileManager.default.fileExists(atPath: target.path))
        #expect(await download(pinned, failing: nil) == nil)
        #expect(try staged().isEmpty && Data(contentsOf: target.appendingPathComponent("a.bin")) == payload)
    }

    /// A download stops as soon as it is larger than the manifest says, or announces another length.
    @Test func downloadsAreCappedAtTheDeclaredSize() {
        #expect(DownloadLimit.problem(expectedLength: -1, received: 10, declaredSize: 100, finished: false) == nil)
        #expect(DownloadLimit.problem(expectedLength: 100, received: 100, declaredSize: 100, finished: true) == nil)
        #expect(DownloadLimit.problem(expectedLength: 200, received: 0, declaredSize: 100, finished: false) != nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: 101, declaredSize: 100, finished: false) != nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: 99, declaredSize: 100, finished: true) != nil)
        // manifest.json has no declared size: capped at DownloadLimit.manifestBytes.
        #expect(DownloadLimit.problem(expectedLength: -1, received: 30_000, declaredSize: nil, finished: true) == nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: Int64(DownloadLimit.manifestBytes) + 1, declaredSize: nil,
                                      finished: false) != nil)
    }

    @Test func readsThePackageManifest() throws {
        let manifest: JSONValue = [
            "name": "embeddinggemma-2-ane", "manifest_version": 2, "created": "2026-11-15",
            "buckets": .array(BucketContract.fullAttentionLengths.map {
                ["function": .string("embed_\($0)"), "tokens": .int($0), "slots": .int(BucketContract.slots(forLength: $0))]
            }),
            "window": ["block": 512, "halo": 512, "band_columns": 1536,
                       "packing": [["function": "winp1536_b512", "tokens": 1536, "slots": 48],
                                   ["function": "winp2048_b512", "tokens": 2048, "slots": 64]],
                       "long": .array([3072, 4096, 6144, 8192].map { ["function": .string("winl\($0)_b512"), "tokens": .int($0)] })],
            "files": [["path": "tokenizer.json", "size": 1, "sha256": .string(String(repeating: "a", count: 64))]],
        ]
        let data = manifest.jsonData()
        let parsed = try Manifest.parse(data)
        #expect(parsed.model == "embeddinggemma-2-ane" && parsed.created == "2026-11-15")
        #expect(parsed.sha256 == SHA256Hasher.hex(data))
        #expect(parsed.buckets[1280] == 40 && parsed.window?.packing[2048] == 64 && parsed.window?.long == [3072, 4096, 6144, 8192])
        #expect(!parsed.legacyLongPath)
        #expect(throws: Never.self) { try parsed.checkContract() }
        var other = parsed
        other.window?.halo = 256
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.window?.long.append(16384)
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.buckets[512] = 8
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.buckets[2048] = 64  // the first layout's largest bucket is a winp length now
        #expect(throws: ServiceError.self) { try other.checkContract() }
        // The first layout (per-layer long path) is rejected with a clear message.
        var legacy = manifest.objectValue!
        legacy["long_path"] = ["chunk": 1024, "halo": 512, "buckets": [4096, 8192]]
        let old = try Manifest.parse(JSONValue.object(legacy).jsonData())
        #expect(old.legacyLongPath)
        #expect(throws: ServiceError.self) { try old.checkContract() }
        #expect(ModelInstaller.localRevision(parsed) == "local-" + parsed.sha256.prefix(12))
    }

    /// The runtime loads every function of the contract, so the manifest must declare all of them.
    @Test func contractRequiresEveryDeclaredFunction() throws {
        let full = DoctorTests.contract(files: [])
        #expect(throws: Never.self) { try Manifest.parse(JSONValue.object(full).jsonData()).checkContract() }
        for key in ["buckets", "window"] {
            var missing = full
            missing[key] = nil
            #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(missing).jsonData()).checkContract() }
        }
        var partial = full
        partial["buckets"] = [["tokens": 64, "slots": 8]]
        #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(partial).jsonData()).checkContract() }
        var noLong = full
        noLong["window"] = ["block": 512, "halo": 512, "band_columns": 1536,
                            "packing": [["tokens": 1536, "slots": 48], ["tokens": 2048, "slots": 64]]]
        #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(noLong).jsonData()).checkContract() }
    }

    @Test func rejectsPathsOutsideTheModelDirectory() {
        let bad: JSONValue = ["files": [["path": "../x", "size": 1, "sha256": .string(String(repeating: "a", count: 64))]]]
        #expect(throws: ServiceError.self) { try Manifest.parse(bad.jsonData()) }
    }

    @Test func activatesRevisionsAtomically() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        try store.activate(revision: "abc123")
        #expect(store.currentDirectory.lastPathComponent == "abc123")
        try store.activate(revision: "def456")
        #expect(try store.currentDirectory == store.directory(for: "def456"))
    }

    @Test func warmStateRoundTrips() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let state = WarmState(osBuild: "26A434", binaryVersion: "0.1.0", binary: "abc", maxTokens: 2048, functions: ["embed_64"],
                              warmedAt: Date(timeIntervalSince1970: 1_800_000_000), seconds: 31.5)
        try state.save(to: directory)
        #expect(WarmState.load(from: directory) == state)
        #expect(state.covers(osBuild: "26A434", maxTokens: 512, binary: "abc"))
        #expect(!state.covers(osBuild: "26B1", maxTokens: 512, binary: "abc"))
        #expect(!state.covers(osBuild: "26A434", maxTokens: 4096, binary: "abc"))
        #expect(!state.covers(osBuild: "26A434", maxTokens: 512, binary: "def"))  // another executable compiles again
        #expect(Executable.identity.count == 64)
        #expect(!operatingSystemBuild.isEmpty)
    }

    /// The data directory's pinned revision loads only with the manifest pinned in this build; another installed
    /// revision and --model-dir are checked against their own manifest.
    @Test func thePinnedRevisionLoadsOnlyWithThePinnedManifest() throws {
        let (dataDirectory, revision) = try DoctorTests.replacedPinnedRevision()
        defer { try? FileManager.default.removeItem(at: dataDirectory) }
        let config = try Config(flags: ["data-dir": dataDirectory.path], environment: [:])
        #expect(config.pinnedManifestSHA256 == ModelSource.pinned.manifestSHA256)
        let error = #expect(throws: ServiceError.self) { try ModelLoader.checkPackage(config) }
        #expect(error?.reason == "pinned_manifest_mismatch" && error?.hint?.contains("aneembed models install") == true)

        let explicit = try Config(flags: ["model-dir": revision.path], environment: [:])
        #expect(explicit.pinnedManifestSHA256 == nil)
        #expect(throws: Never.self) { try ModelLoader.checkPackage(explicit) }
        let store = ModelStore(dataDirectory: dataDirectory)
        try FileManager.default.moveItem(at: revision, to: store.modelsDirectory.appendingPathComponent("local-0123456789ab"))
        try store.activate(revision: "local-0123456789ab")
        #expect(config.pinnedManifestSHA256 == nil)
        #expect(throws: Never.self) { try ModelLoader.checkPackage(config) }
    }

    @Test func pinnedSourceIsAPublishedCommit() throws {
        #expect(ModelSource.pinned.isPublished)
        #expect(ModelSource.pinned.revision.count == 40 && ModelSource.pinned.revision.allSatisfy(\.isHexDigit))
        #expect(!ModelSource(repository: "a/b", revision: "UNPUBLISHED").isPublished)
        #expect(ModelSource.pinned.manifestSHA256?.count == 64)
        #expect(try ModelSource(repository: "a/b", revision: "0123abc").url(for: "manifest.json").absoluteString
                == "https://huggingface.co/a/b/resolve/0123abc/manifest.json")
    }

    /// A revision becomes a directory name under models/, so only one plain path component is accepted.
    @Test func revisionsAreOnePlainPathComponent() throws {
        for good in [ModelSource.pinned.revision, "local-0123456789ab", "v1.0", "main", "a_b-c.d"] {
            #expect(ModelSource.isValidRevision(good), "\(good)")
        }
        for bad in ["", ".", "..", "../x", "a/b", "/", "-x", ".hidden", " a", "a b", "a\u{0}", String(repeating: "a", count: 65)] {
            #expect(!ModelSource.isValidRevision(bad), "\(bad)")
        }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        #expect(throws: ServiceError.self) { try store.directory(for: "..") }
        #expect(throws: ServiceError.self) { try store.directory(for: "../../etc") }
        #expect(throws: ServiceError.self) { try store.activate(revision: "..") }
        #expect(try store.directory(for: "abc").deletingLastPathComponent().standardizedFileURL
                == store.modelsDirectory.standardizedFileURL)
        // A hand-edited `current` pointer is held to the same rule (else the pinned revision is used).
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        try "..\n".write(to: store.modelsDirectory.appendingPathComponent("current"), atomically: true, encoding: .utf8)
        #expect(store.currentDirectory.lastPathComponent == ModelSource.pinned.revision)
        try store.activate(revision: "abc")
        #expect(store.currentDirectory.lastPathComponent == "abc")
        // The installer replaces only direct children of models/, after resolving `..` and symlinks.
        let manifest = try Manifest.parse(JSONValue.object(["files": []]).jsonData())
        let staging = store.modelsDirectory.appendingPathComponent(".abc.partial")
        for target in [store.modelsDirectory.appendingPathComponent(".."), directory,
                       store.modelsDirectory.appendingPathComponent("link/abc")]
        {
            try? FileManager.default.createSymbolicLink(at: store.modelsDirectory.appendingPathComponent("link"),
                                                        withDestinationURL: FileManager.default.temporaryDirectory)
            #expect(throws: ServiceError.self, "\(target.path)") {
                try ModelInstaller.finish(staging: staging, into: target, store: store, manifest: manifest, log: { _ in })
            }
        }
        #expect(FileManager.default.fileExists(atPath: store.currentDirectory.deletingLastPathComponent().path))
    }

    /// Repository, revision and file paths end up in a URL: malformed ones are errors, not a crash.
    @Test func downloadURLsAreBuiltSafely() throws {
        #expect(try ModelSource(repository: "a/b", revision: "r1").url(for: "dir/tok enizer.json").absoluteString
                == "https://huggingface.co/a/b/resolve/r1/dir/tok%20enizer.json")
        for repository in ["a b/c", "a", "a/b/c", "../b", "a/..", "a/b?x", ""] {
            #expect(throws: ServiceError.self, "\(repository)") { try ModelSource(repository: repository, revision: "r1").url(for: "x") }
        }
        #expect(throws: ServiceError.self) { try ModelSource(repository: "a/b", revision: "r 1").url(for: "x") }
        #expect(throws: ServiceError.self) { try ModelSource(repository: "a/b", revision: "..").url(for: "x") }
        for path in ["a//b", "./a", "a/./b", "a\u{1}b"] {
            let bad: JSONValue = ["files": [["path": .string(path), "size": 1, "sha256": .string(String(repeating: "a", count: 64))]]]
            #expect(throws: ServiceError.self, "\(path)") { try Manifest.parse(bad.jsonData()) }
        }
    }
}
