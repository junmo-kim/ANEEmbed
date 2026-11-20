import CryptoKit
import Darwin
import Foundation

struct Slice: Decodable {
    let file: String
    let offset: UInt64
    let size: Int
}

struct Tensor: Decodable {
    let source: Slice
    let sha256: String
    let scale: Int
}

struct Recipe: Decodable {
    let emptyDirectories: [String]
    let original: String
    let base, overlay: [String: String]
    let sharedFiles: [String: String]?
    let stages: [String: [String: Tensor]]
}

struct Invalid: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

func child(_ root: URL, _ relative: String) throws -> URL {
    let parts = relative.split(separator: "/", omittingEmptySubsequences: false)
    guard parts.allSatisfy({ !$0.isEmpty && $0 != ".." && $0 != "." }) else { throw Invalid(relative) }
    let file = root.appendingPathComponent(relative)
    guard file.resolvingSymlinksInPath().standardizedFileURL.path == file.standardizedFileURL.path else {
        throw Invalid("Unexpected symbolic link: " + relative)
    }
    return file
}

func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }

func hash(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var h = SHA256()
    while try autoreleasepool(invoking: {
        guard let d = try handle.read(upToCount: 4 << 20), !d.isEmpty else { return false }
        h.update(data: d)
        return true
    }) {}
    return hex(h.finalize())
}

func append(_ tensor: Tensor, from original: URL, to output: FileHandle) throws {
    let input = try FileHandle(forReadingFrom: child(original, tensor.source.file))
    defer { try? input.close() }
    try input.seek(toOffset: tensor.source.offset)
    var remaining = tensor.source.size
    guard remaining > 0, remaining.isMultiple(of: 2), [1, 64].contains(tensor.scale) else { throw Invalid("Invalid weight slice") }
    var hasher = SHA256()
    while remaining > 0 {
        try autoreleasepool {
            let expected = min(remaining, 4 << 20)
            var data = Data()
            while data.count < expected {
                guard let bytes = try input.read(upToCount: expected - data.count), !bytes.isEmpty else {
                    throw Invalid("Truncated FP16 weights")
                }
                data.append(bytes)
            }
            hasher.update(data: data)
            if tensor.scale == 64 {
                data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    let elements = raw.bindMemory(to: Float16.self)
                    for i in elements.indices { elements[i] *= 64 }
                }
            }
            try output.write(contentsOf: data)
            remaining -= expected
        }
    }
    guard hex(hasher.finalize()) == tensor.sha256 else { throw Invalid("Weight digest mismatch in " + tensor.source.file) }
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else { throw Invalid("Usage: repack EXISTING_MODEL KIT NEW_MODEL") }
    let fm = FileManager.default
    let urls = args.dropFirst().map { URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath() }
    let (original, kit, dest) = (urls[0], urls[1], urls[2])
    guard !fm.fileExists(atPath: dest.path) else { throw Invalid("Destination already exists") }
    let recipe = try JSONDecoder().decode(Recipe.self, from: Data(contentsOf: kit.appendingPathComponent("recipe.json")))
    guard try hash(child(original, "model.json")) == recipe.original else { throw Invalid("Unexpected input bundle revision") }
    let temp = dest.deletingLastPathComponent().appendingPathComponent(".valen-repack-" + UUID().uuidString)
    try fm.createDirectory(at: temp, withIntermediateDirectories: false)
    defer { try? fm.removeItem(at: temp) }
    for (root, files) in [(original, recipe.base), (kit, recipe.overlay)] {
        for (relative, digest) in files {
            let target = try child(temp, relative)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: child(root, relative), to: target)
            guard try hash(target) == digest else { throw Invalid("File digest mismatch: " + relative) }
        }
    }
    // Two fixed-shape programs may reference identical immutable weights.
    for (relative, existing) in recipe.sharedFiles ?? [:] {
        guard recipe.base[existing] != nil || recipe.overlay[existing] != nil else {
            throw Invalid("Shared file must refer to a verified payload: " + existing)
        }
        let source = try child(temp, existing), target = try child(temp, relative)
        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        if link(source.path, target.path) != 0 {
            let code = errno
            guard code == ENOTSUP || code == EPERM || code == EXDEV || code == EMLINK else {
                throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
            }
            try fm.copyItem(at: source, to: target)
        }
    }
    for relative in recipe.emptyDirectories {
        try fm.createDirectory(at: child(temp, relative), withIntermediateDirectories: true)
    }
    let mlpNames = Set(["norm_scale","g0","g1","g2","u0","u1","u2","d0","d1","d2"])
    for (stage, tensors) in recipe.stages {
        guard let index = Int(stage), index >= 0, String(index) == stage else { throw Invalid("Invalid stage") }
        for mlp in [false, true] {
            let file = temp.appendingPathComponent((mlp ? "valenMLP" : "valenDelta") + stage + ".fp16")
            guard !fm.fileExists(atPath: file.path), fm.createFile(atPath: file.path, contents: nil) else {
                throw Invalid("Cannot create weight bank " + file.lastPathComponent)
            }
            let writer = try FileHandle(forWritingTo: file)
            defer { try? writer.close() }
            for name in tensors.keys.sorted() where mlpNames.contains(name) == mlp {
                try append(tensors[name]!, from: original, to: writer)
            }
        }
    }
    try fm.moveItem(at: temp, to: dest)
}
do { try run() } catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
