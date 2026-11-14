// aneembed: EmbeddingGemma 2 embeddings on the Apple Neural Engine. See `aneembed --help`.
import EmbedCore
import Foundation

let code = await CLI.run(Array(CommandLine.arguments.dropFirst()))
exit(code)
