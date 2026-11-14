// The only outbound network use: downloading a pinned model revision from Hugging Face. ModelInstaller.download
// decides what is fetched: manifest.json first (checked against the hash pinned in this build), then every source
// file it lists (compiled models are built here, not downloaded) into a staging directory, each checked against
// its SHA-256; the staging directory becomes the revision directory only when everything matches.
import EmbedCore
import Foundation

struct ModelDownloader {
    let source: ModelSource

    func install(into target: URL, store: ModelStore) async throws(ServiceError) {
        try await ModelInstaller.download(source, into: target, store: store, log: { Log.info($0) }) {
            path, directory, size throws(ServiceError) in
            try await fetch(path, to: directory, size: size)
        }
    }

    /// Downloads `path` into `directory`, stopping as soon as it exceeds `size` (the manifest's) or, for
    /// manifest.json (`size` nil), DownloadLimit.manifestBytes. Returns the bytes of manifest.json.
    @discardableResult
    private func fetch(_ path: String, to directory: URL, size: Int? = nil) async throws(ServiceError) -> Data {
        let url = try source.url(for: path)
        let destination = directory.appendingPathComponent(path)
        let download = CappedDownload(destination: destination, declaredSize: size)
        let session = URLSession(configuration: .ephemeral, delegate: download, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        do {
            try await download.run(session.downloadTask(with: url))
            return size == nil ? try Data(contentsOf: destination) : Data()
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "download_failed", "\(url.absoluteString): \(error.localizedDescription)")
        }
    }
}

/// One download task that is cancelled as soon as it goes past the declared size, and whose file is moved to
/// `destination` only when it is complete and of that size.
private final class CappedDownload: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let destination: URL
    let declaredSize: Int?
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var failure: Error?

    init(destination: URL, declaredSize: Int?) {
        self.destination = destination
        self.declaredSize = declaredSize
    }

    func run(_ task: URLSessionDownloadTask) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            lock.withLock { self.continuation = continuation }
            task.resume()
        }
    }

    private func fail(_ message: String, _ task: URLSessionTask) {
        let url = task.originalRequest?.url?.absoluteString ?? "download"
        lock.withLock { if failure == nil { failure = ServiceError(.unavailable, "download_failed", "\(url): \(message)") } }
        task.cancel()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64)
    {
        let expected = totalBytesExpectedToWrite == NSURLSessionTransferSizeUnknown ? -1 : totalBytesExpectedToWrite
        if let problem = DownloadLimit.problem(expectedLength: expected, received: totalBytesWritten, declaredSize: declaredSize,
                                               finished: false)
        {
            fail(problem, downloadTask)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let http = downloadTask.response as? HTTPURLResponse, http.statusCode == 200 else {
            fail("HTTP \((downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0)", downloadTask)
            return
        }
        // The file at `location` exists only during this call: check it and move it now.
        let received = ((try? FileManager.default.attributesOfItem(atPath: location.path))?[.size] as? NSNumber)?.int64Value ?? -1
        if let problem = DownloadLimit.problem(expectedLength: -1, received: received, declaredSize: declaredSize, finished: true) {
            fail(problem, downloadTask)
            return
        }
        do {
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            fail(error.localizedDescription, downloadTask)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let (continuation, failure) = lock.withLock { () -> (CheckedContinuation<Void, Error>?, Error?) in
            defer { self.continuation = nil }
            return (self.continuation, self.failure)
        }
        if let failure = failure ?? error {
            continuation?.resume(throwing: failure)
        } else if FileManager.default.fileExists(atPath: destination.path) {
            continuation?.resume()
        } else {
            continuation?.resume(throwing: ServiceError(.unavailable, "download_failed", "\(destination.lastPathComponent): no data"))
        }
    }
}
