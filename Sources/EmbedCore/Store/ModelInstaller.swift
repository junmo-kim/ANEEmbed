// Installing a model package: copy (from a local directory) or download (ModelDownloader) into a staging
// directory, check every file against the manifest's SHA-256, move it into place, then compile the
// .mlpackage files on this Mac. Warming and the self-test follow in `models install`.
@preconcurrency import CoreML
import Foundation

/// How far a download may go: what the manifest declares, and at most `manifestBytes` for manifest.json itself.
public enum DownloadLimit {
    public static let manifestBytes = 4 << 20

    /// Why the download must stop, or nil. `expectedLength` is the response's Content-Length (-1: unknown);
    /// `finished` checks the final size too.
    public static func problem(expectedLength: Int64, received: Int64, declaredSize: Int?, finished: Bool) -> String? {
        let cap = Int64(declaredSize ?? manifestBytes)
        if let declaredSize, expectedLength >= 0, expectedLength != Int64(declaredSize) {
            return "the server sends \(expectedLength) bytes; the manifest says \(declaredSize)"
        }
        if expectedLength > cap { return "the server sends \(expectedLength) bytes, more than \(cap)" }
        if received > cap { return "more than \(cap) bytes" }
        if finished, let declaredSize, received != Int64(declaredSize) {
            return "\(received) bytes; the manifest says \(declaredSize)"
        }
        return nil
    }
}

/// Serializes installs: an exclusive flock on `models/.install.lock`, held until `release()` (or deinit).
public final class InstallLock: @unchecked Sendable {
    private var descriptor: Int32

    private init(descriptor: Int32) { self.descriptor = descriptor }

    /// Takes the lock; when another install holds it, waits (`wait`, after calling `onWait`) or fails.
    public static func acquire(in store: ModelStore, wait: Bool = true, onWait: () -> Void = {}) throws(ServiceError)
        -> InstallLock
    {
        let path = store.modelsDirectory.appendingPathComponent(".install.lock").path
        do { try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true) } catch {
            throw ServiceError(.internal, "install_failed", "cannot create \(store.modelsDirectory.path): \(error.localizedDescription)")
        }
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else {
            throw ServiceError(.internal, "install_failed", "cannot open \(path): \(String(cString: strerror(errno)))")
        }
        if flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            guard errno == EWOULDBLOCK, wait else {
                close(descriptor)
                throw ServiceError(.unavailable, "install_in_progress", "another `aneembed models install` is running",
                                   hint: "wait for it to finish, then run the command again")
            }
            onWait()
            while flock(descriptor, LOCK_EX) != 0 {
                guard errno == EINTR else {
                    close(descriptor)
                    throw ServiceError(.internal, "install_failed", "cannot lock \(path): \(String(cString: strerror(errno)))")
                }
            }
        }
        return InstallLock(descriptor: descriptor)
    }

    public func release() {
        guard descriptor >= 0 else { return }
        flock(descriptor, LOCK_UN)
        close(descriptor)
        descriptor = -1
    }

    deinit { release() }
}

/// Puts a model package in place: copy or download, check, compile.
public enum ModelInstaller {
    /// A staging directory next to `target`, unique to this call (process id and a random suffix).
    public static func stagingDirectory(for target: URL) -> URL {
        let suffix = UUID().uuidString.prefix(8).lowercased()
        return target.deletingLastPathComponent()
            .appendingPathComponent(".\(target.lastPathComponent).partial-\(ProcessInfo.processInfo.processIdentifier)-\(suffix)")
    }

    /// Removes the staging directories interrupted installs of `target` left behind. Call with the install lock held.
    public static func removeStaleStaging(for target: URL) {
        let parent = target.deletingLastPathComponent()
        let prefix = ".\(target.lastPathComponent).partial"
        for name in (try? FileManager.default.contentsOfDirectory(atPath: parent.path)) ?? [] where name.hasPrefix(prefix) {
            try? FileManager.default.removeItem(at: parent.appendingPathComponent(name))
        }
    }

    /// The revision directory name for a package installed from a local directory.
    public static func localRevision(_ manifest: Manifest) -> String { "local-" + manifest.sha256.prefix(12) }

    /// Whether `target` holds exactly the package whose manifest.json hashes to `manifestSHA256`, every source file
    /// matching it. A package that is only consistent with its own manifest is not enough.
    public static func isInstalled(at target: URL, manifestSHA256: String) -> Bool {
        guard let manifest = try? Manifest.load(from: target), manifest.sha256 == manifestSHA256.lowercased() else { return false }
        return manifest.verify(in: target, hash: false).isEmpty && manifest.verify(in: target, hash: true).isEmpty
    }

    /// Copies every source file the manifest of `source` lists into `target` through a staging directory
    /// and checks each against its SHA-256. `target` is replaced only when everything matches.
    public static func copy(from source: URL, into target: URL, store: ModelStore, log: (String) -> Void = { _ in })
        throws(ServiceError) -> Manifest
    {
        let manifest = try Manifest.load(from: source)
        let fileManager = FileManager.default
        let staging = stagingDirectory(for: target)
        try store.checkInside(target)
        try store.checkInside(staging)
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
            for (index, file) in manifest.sourceFiles.enumerated() {
                log("copying \(index + 1)/\(manifest.sourceFiles.count) \(file.path)")
                let origin = source.appendingPathComponent(file.path)
                guard Manifest.isRegularFile(origin) else {
                    throw ServiceError(.invalidInput, "checksum_mismatch", "\(file.path) is not a regular file")
                }
                let destination = staging.appendingPathComponent(file.path)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fileManager.copyItem(at: origin, to: destination)
            }
            try fileManager.copyItem(at: source.appendingPathComponent(ModelLayout.manifest),
                                     to: staging.appendingPathComponent(ModelLayout.manifest))
        } catch let error as ServiceError {
            try? fileManager.removeItem(at: staging)
            throw error
        } catch {
            try? fileManager.removeItem(at: staging)
            throw ServiceError(.unavailable, "copy_failed", "cannot copy the package: \(error.localizedDescription)")
        }
        try finish(staging: staging, into: target, store: store, manifest: manifest, log: log)
        return manifest
    }

    /// Downloads `source` into `target` through a staging directory: manifest.json first (checked against the
    /// source's pinned hash), then every source file it lists; `finish` checks each against its SHA-256.
    /// `fetch(path, directory, size)` stores one file of the revision in `directory` and returns the bytes of
    /// manifest.json (`size` nil); ModelDownloader passes the network. Every failure once the staging directory
    /// exists removes it, so a failed install leaves no partial download behind.
    public static func download(_ source: ModelSource, into target: URL, store: ModelStore, log: (String) -> Void = { _ in },
                                fetch: (_ path: String, _ directory: URL, _ size: Int?) async throws(ServiceError) -> Data)
        async throws(ServiceError)
    {
        try source.validate()
        let fileManager = FileManager.default
        // Skip only when the installed package is the pinned one (by the hash compiled into this build).
        if let pinned = source.manifestSHA256, isInstalled(at: target, manifestSHA256: pinned) {
            log("\(target.lastPathComponent) is already installed and matches the pinned manifest")
            return
        }
        let staging = stagingDirectory(for: target)
        try store.checkInside(staging)
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        } catch {
            throw ServiceError(.internal, "staging_failed", "cannot create \(staging.path): \(error.localizedDescription)")
        }
        do throws(ServiceError) {
            let manifestData = try await fetch(ModelLayout.manifest, staging, nil)
            if let pinned = source.manifestSHA256, SHA256Hasher.hex(manifestData) != pinned {
                throw ServiceError(.invalidInput, "checksum_mismatch", "manifest.json does not match the hash pinned in this build")
            }
            // An explicit --revision has no pinned hash: skip only when the installed package is the one this source
            // serves now (another --repo with the same revision name must not reuse it).
            if source.manifestSHA256 == nil, isInstalled(at: target, manifestSHA256: SHA256Hasher.hex(manifestData)) {
                try? fileManager.removeItem(at: staging)
                log("\(target.lastPathComponent) is already installed and matches \(source.repository)'s manifest")
                return
            }
            let manifest = try Manifest.parse(manifestData)
            for (index, file) in manifest.sourceFiles.enumerated() {
                log("downloading \(index + 1)/\(manifest.sourceFiles.count) \(file.path) (\(file.size / (1 << 20)) MiB)")
                _ = try await fetch(file.path, staging, file.size)
            }
            try finish(staging: staging, into: target, store: store, manifest: manifest, log: log)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    /// Verifies a staged package against `manifest` and moves it to `target`. Both must be inside the store's
    /// models directory: `target` is replaced.
    public static func finish(staging: URL, into target: URL, store: ModelStore, manifest: Manifest, log: (String) -> Void)
        throws(ServiceError)
    {
        try store.checkInside(target)
        try store.checkInside(staging)
        let problems = withoutActuallyEscaping(log) { log in manifest.verify(in: staging, hash: true) { log("verifying \($0)") } }
        guard problems.isEmpty else {
            try? FileManager.default.removeItem(at: staging)
            throw ServiceError(.invalidInput, "checksum_mismatch",
                               "the package does not match its manifest: \(problems.map { "\($0.path) \($0.issue)" }.joined(separator: ", "))")
        }
        // One rename: a reader sees the old package or the new one, never no directory. An existing target is
        // swapped with the staging directory (APFS/HFS+ RENAME_SWAP), which then holds the old package.
        let swapped = FileManager.default.fileExists(atPath: target.path)
        let result = swapped ? renamex_np(staging.path, target.path, UInt32(RENAME_SWAP)) : rename(staging.path, target.path)
        guard result == 0 else {
            let reason = String(cString: strerror(errno))
            try? FileManager.default.removeItem(at: staging)
            throw ServiceError(.internal, "install_failed", "cannot move the package into place: \(reason)")
        }
        if swapped { try? FileManager.default.removeItem(at: staging) }
    }

    /// Compiles every `.mlpackage` in `directory` that has no `.mlmodelc` yet. Returns the compiled files that
    /// differ from the manifest's reference compile (expected only with another Core ML compiler; the
    /// self-test decides whether they compute correctly).
    public static func compile(in directory: URL, log: (String) -> Void = { _ in }) async throws(ServiceError)
        -> [Manifest.Problem]
    {
        for name in ModelLayout.packages {
            let package = directory.appendingPathComponent(name + ".mlpackage")
            guard FileManager.default.fileExists(atPath: package.path) else { continue }
            log("compiling \(name).mlpackage")
            do { _ = try await compiledModelURL(package) } catch {
                throw ServiceError(.unavailable, "compile_failed", "cannot compile \(name): \(error)")
            }
        }
        guard let manifest = try? Manifest.load(from: directory) else { return [] }
        return manifest.verifyAll(in: directory, hash: true).compiled
    }
}
