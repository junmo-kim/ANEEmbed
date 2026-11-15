// swift-tools-version: 6.0
import PackageDescription

// ANEEmbed: an EmbeddingGemma 2 embedding server and CLI that runs only on the Apple Neural Engine.
// 100% Swift, single binary, no third-party dependencies (Foundation, Network, CoreML, ImageIO,
// Accelerate and libSystem only). Deployment target macOS 15: the first release with multifunction
// Core ML models.
//
// Release builds use the default -O. Unlike a short-lived CLI, this is a long-running server whose
// own code (tokenizer, gathers, bias building, JSON encoding) sits on the request path, so speed is
// the default; -Osize is to be measured against it before changing.

let package = Package(
    name: "ANEEmbed",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "aneembed", targets: ["aneembed"]),
        .library(name: "EmbedCore", targets: ["EmbedCore"]),
    ],
    targets: [
        // Tokenizer, token table, image preprocessing, bias/pool builders, packing planner,
        // runtime protocols, API and MCP logic. Everything that does not open a socket.
        .target(name: "EmbedCore"),
        // CLI, HTTP listener (Network.framework) and stdio MCP loop.
        .executableTarget(name: "aneembed", dependencies: ["EmbedCore"]),
        // Development tool, not a product: deterministic mutation fuzzing of the JPEG decoder. Tests run it as a
        // child process, because a trap cannot be caught in process.
        .executableTarget(name: "jpeg-fuzz", dependencies: ["EmbedCore"]),
        // Pure logic: runs anywhere, including CI runners without a Neural Engine.
        .testTarget(name: "EmbedCoreTests", dependencies: ["EmbedCore", "jpeg-fuzz"]),
        // HTTP/OpenAI/MCP contracts end to end through the fake runtime (no model needed).
        .testTarget(name: "ServiceTests", dependencies: ["EmbedCore"]),
        // Real model on the Neural Engine. Skipped when CI is set or no model directory exists.
        .testTarget(name: "ModelTests", dependencies: ["EmbedCore"]),
    ]
)
