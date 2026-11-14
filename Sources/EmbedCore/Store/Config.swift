// Configuration: every option is a flag and an environment variable with the same name
// (--max-tokens <-> ANEEMBED_MAX_TOKENS). Precedence: flag > environment > default.
// `--print-config` shows each resolved value and where it came from.
import Foundation

public struct Config: Sendable, Equatable {
    public enum Source: String, Sendable {
        case flag
        case environment = "env"
        case `default`
    }

    public struct Option: Sendable {
        public let name: String
        public let help: String
        public let defaultValue: @Sendable () -> String
        public let validate: @Sendable (String) -> String?  // error message or nil

        public var flag: String { "--\(name)" }
        public var environmentName: String { "ANEEMBED_" + name.uppercased().replacingOccurrences(of: "-", with: "_") }
    }

    public static let options: [Option] = [
        Option(name: "model-dir", help: "model package directory (default: <data-dir>/models/current)",
               defaultValue: { "" }, validate: { _ in nil }),
        Option(name: "data-dir", help: "where models and state live",
               defaultValue: { Config.defaultDataDirectory.path }, validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "host", help: "listen address (loopback by default; anything else exposes an unauthenticated API)",
               defaultValue: { "127.0.0.1" }, validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "port", help: "listen port", defaultValue: { "18180" }, validate: integer(1...65535)),
        Option(name: "max-tokens", help: "longest text accepted, tokens incl. <bos>/<eos>; only functions up to it are loaded",
               defaultValue: { "2048" }, validate: integer(2...8192)),
        Option(name: "model-name", help: "model id reported by the API", defaultValue: { "embeddinggemma-2" },
               validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "max-body", help: "request body limit in MiB", defaultValue: { "72" }, validate: integer(1...1024)),
        Option(name: "max-inputs", help: "texts per request", defaultValue: { "2048" }, validate: integer(1...100_000)),
        Option(name: "max-images", help: "images per request", defaultValue: { "8" }, validate: integer(1...64)),
        Option(name: "query-concurrency", help: "concurrent query-lane batches (texts <= 512 tokens)",
               defaultValue: { "8" }, validate: integer(1...64)),
        Option(name: "image-concurrency", help: "concurrent image decodes and encodes", defaultValue: { "2" },
               validate: integer(1...64)),
        Option(name: "long-concurrency", help: "concurrent texts over 512 tokens", defaultValue: { "2" },
               validate: integer(1...64)),
        Option(name: "runtime", help: "coreml (Neural Engine) or fake (no model, for testing integrations)",
               defaultValue: { "coreml" }, validate: { ["coreml", "fake"].contains($0) ? nil : "must be coreml or fake" }),
        Option(name: "selftest",
               help: "require: serve only functions that pass the package's self-test; skip: serve every function unchecked (unsafe)",
               defaultValue: { "require" }, validate: { ["require", "skip"].contains($0) ? nil : "must be require or skip" }),
    ]

    public static func option(named name: String) -> Option? { options.first { $0.name == name } }

    private static func integer(_ range: ClosedRange<Int>) -> @Sendable (String) -> String? {
        { value in
            guard let int = Int(value) else { return "must be an integer" }
            return range.contains(int) ? nil : "must be in \(range.lowerBound)...\(range.upperBound)"
        }
    }

    public static var defaultDataDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/ANEEmbed", isDirectory: true)
    }

    public private(set) var values: [String: String] = [:]
    public private(set) var sources: [String: Source] = [:]

    /// Resolves every option from `flags` (name without dashes -> value) and `environment`.
    public init(flags: [String: String] = [:], environment: [String: String] = ProcessInfo.processInfo.environment)
        throws(ServiceError)
    {
        for name in flags.keys where Self.option(named: name) == nil {
            throw .badRequest("unknown_option", "unknown option --\(name)")
        }
        for option in Self.options {
            let value: String
            let source: Source
            if let flag = flags[option.name] {
                (value, source) = (flag, .flag)
            } else if let env = environment[option.environmentName], !env.isEmpty {
                (value, source) = (env, .environment)
            } else {
                (value, source) = (option.defaultValue(), .default)
            }
            if source != .default || !value.isEmpty, let problem = option.validate(value) {
                let origin = source == .flag ? option.flag : option.environmentName
                throw .badRequest("invalid_option", "\(origin)=\(value): \(problem)", param: option.name)
            }
            values[option.name] = value
            sources[option.name] = source
        }
    }

    public func string(_ name: String) -> String { values[name] ?? "" }
    public func int(_ name: String) -> Int { Int(values[name] ?? "") ?? 0 }

    public var dataDirectory: URL { URL(fileURLWithPath: (string("data-dir") as NSString).expandingTildeInPath, isDirectory: true) }
    public var host: String { string("host") }
    public var port: Int { int("port") }
    public var maxTokens: Int { int("max-tokens") }
    public var modelName: String { string("model-name") }
    public var maxBodyBytes: Int { int("max-body") << 20 }
    public var useFakeRuntime: Bool { string("runtime") == "fake" }
    /// `--selftest skip`: load without the package's self-test and serve every function unchecked (unsafe).
    public var skipSelfTest: Bool { string("selftest") == "skip" }

    /// The configured model directory, or the data directory's current revision.
    public var modelDirectory: URL {
        let explicit = string("model-dir")
        if !explicit.isEmpty { return URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath, isDirectory: true) }
        return ModelStore(dataDirectory: dataDirectory).currentDirectory
    }

    public var limits: ServiceLimits {
        var limits = ServiceLimits()
        limits.maxTokens = maxTokens
        limits.maxInputs = int("max-inputs")
        limits.maxImages = int("max-images")
        limits.queryConcurrency = int("query-concurrency")
        limits.imageConcurrency = int("image-concurrency")
        limits.longConcurrency = int("long-concurrency")
        return limits
    }

    /// `--print-config`: value and source of every option.
    public var printable: JSONValue {
        .object(JSONObject(Self.options.map { option in
            var value = string(option.name)
            if option.name == "model-dir" && value.isEmpty { value = modelDirectory.path }
            return (key: option.name, value: ["value": .string(value), "source": .string(sources[option.name]!.rawValue),
                                              "env": .string(option.environmentName)])
        }))
    }

    public static func == (lhs: Config, rhs: Config) -> Bool { lhs.values == rhs.values && lhs.sources == rhs.sources }
}
