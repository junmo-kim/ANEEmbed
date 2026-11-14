// The only outbound network use: downloading a pinned model revision from Hugging Face.
// manifest.json first, then every file it lists into a staging directory, each checked against its
// SHA-256; the staging directory becomes the revision directory only when everything matches.
import EmbedCore
import Foundation

struct ModelDownloader {
    let source: ModelSource

    func install(into target: URL) async throws(ServiceError) {
        let fileManager = FileManager.default
        if let manifest = try? Manifest.load(from: target), manifest.verify(in: target, hash: false).isEmpty {
            Log.info("\(target.lastPathComponent) is already installed; verifying")
            if manifest.verify(in: target, hash: true).isEmpty { return }
        }
        let staging = target.deletingLastPathComponent().appendingPathComponent(".\(target.lastPathComponent).partial")
        do {
            try? fileManager.removeItem(at: staging)
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        } catch {
            throw ServiceError(.internal, "staging_failed", "cannot create \(staging.path): \(error.localizedDescription)")
        }
        let manifestData = try await fetch(ModelLayout.manifest, to: staging)
        let manifest = try Manifest.parse(manifestData)
        for (index, file) in manifest.files.enumerated() {
            Log.info("downloading \(index + 1)/\(manifest.files.count) \(file.path) (\(file.size / (1 << 20)) MiB)")
            _ = try await fetch(file.path, to: staging, keepData: false)
        }
        let problems = manifest.verify(in: staging, hash: true) { Log.info("verifying \($0)") }
        guard problems.isEmpty else {
            throw ServiceError(.invalidInput, "checksum_mismatch",
                               "downloaded files do not match the manifest: \(problems.map { "\($0.path) \($0.issue)" }.joined(separator: ", "))")
        }
        do {
            try? fileManager.removeItem(at: target)
            try fileManager.moveItem(at: staging, to: target)
        } catch {
            throw ServiceError(.internal, "install_failed", "cannot move the download into place: \(error.localizedDescription)")
        }
    }

    /// Downloads `path` into `directory`; returns its bytes when `keepData`.
    @discardableResult
    private func fetch(_ path: String, to directory: URL, keepData: Bool = true) async throws(ServiceError) -> Data {
        let url = source.url(for: path)
        let destination = directory.appendingPathComponent(path)
        do {
            let (temporary, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                throw ServiceError(.unavailable, "download_failed", "GET \(url.absoluteString) returned HTTP \(status)")
            }
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: temporary, to: destination)
            return keepData ? try Data(contentsOf: destination) : Data()
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "download_failed", "\(url.absoluteString): \(error.localizedDescription)")
        }
    }
}
