// anerun: model installation, serving and inference on the Apple Neural Engine.
import DecisionRuntime
import EmbedCore
import Foundation

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.first == "_portable-load-source" || arguments.first == "_portable-prepare-compiled" {
    do { try await PortableRecovery.runChild(arguments) }
    catch let failure as CompiledFunctionLoadFailure where arguments.first == "_portable-prepare-compiled" && arguments.count == 3 {
        if let payload = try? JSONEncoder().encode(failure), payload.count <= 4096,
           (try? payload.write(to: URL(fileURLWithPath: arguments[2]), options: .withoutOverwriting)) != nil {
            exit(PortableRecovery.entryLoadExit)
        }
        exit(1)
    } catch {
        Output.write("portable recovery failed: \(error)\n", to: .standardError)
        exit(1)
    }
} else {
    exit(await CLI.run(arguments))
}
