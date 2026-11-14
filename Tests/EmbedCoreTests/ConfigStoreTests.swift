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
        #expect(loaded.verify(in: directory, hash: false) == [.init(path: "missing.bin", issue: "missing")])
        #expect(loaded.verify(in: directory, hash: true) == [
            .init(path: "m.mlmodelc/model.mil", issue: "sha256_mismatch"), .init(path: "missing.bin", issue: "missing"),
        ])
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
        #expect(store.currentDirectory == store.directory(for: "def456"))
    }

    @Test func warmStateRoundTrips() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let state = WarmState(osBuild: "26A434", binaryVersion: "0.1.0", maxTokens: 2048, functions: ["embed_64"],
                              warmedAt: Date(timeIntervalSince1970: 1_800_000_000), seconds: 31.5)
        try state.save(to: directory)
        #expect(WarmState.load(from: directory) == state)
        #expect(state.covers(osBuild: "26A434", maxTokens: 512))
        #expect(!state.covers(osBuild: "26B1", maxTokens: 512))
        #expect(!state.covers(osBuild: "26A434", maxTokens: 4096))
        #expect(!operatingSystemBuild.isEmpty)
    }

    @Test func pinnedSourceIsAPlaceholderUntilPublished() {
        #expect(!ModelSource.pinned.isPublished)
        #expect(ModelSource(repository: "a/b", revision: "0123abc").url(for: "manifest.json").absoluteString
                == "https://huggingface.co/a/b/resolve/0123abc/manifest.json")
    }
}
