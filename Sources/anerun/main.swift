// anerun: model installation, serving and inference on the Apple Neural Engine.
import EmbedCore
import Foundation

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.first == "_portable-load-source" || arguments.first == "_portable-prepare-compiled" {
    do { try await PortableRecovery.runChild(arguments) }
    catch {
        Output.write("portable recovery failed: \(error)\n", to: .standardError)
        exit(1)
    }
} else {
    exit(await CLI.run(arguments))
}
