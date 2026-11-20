import CoreML
import DecisionRuntime
import EmbedCore
import Foundation

struct DecisionCapabilityCommand: Command {
    let spec = ArgumentSpec(values: ["format", "plan"], flags: [])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        guard arguments.positionals.isEmpty else {
            throw .badRequest("invalid_arguments", "try anerun decision-capability [--plan FUNCTION.mlpackage]")
        }
        guard #available(macOS 26.0, *) else {
            throw ServiceError(.unavailable, "unsupported_os", "experimental Core ML decisions require macOS 26 or later")
        }
        guard CoreMLDecisionCapability.hasNeuralEngine else {
            throw ServiceError(.unavailable, "neural_engine_unavailable", "Core ML exposes no Neural Engine on this host")
        }
        do { try CoreMLDecisionCapability.requireHardware() }
        catch { throw ServiceError(.unavailable, "unsupported_hardware", String(describing: error)) }
        var result: JSONObject = ["backend": "coreml", "experimental": true,
                                          "hardware_model": .string(CoreMLDecisionCapability.hardwareModel),
                                          "neural_engine": true, "runtime_attestation": false]
        if let path = arguments.value("plan") {
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            guard ["mlpackage", "mlmodelc"].contains(url.pathExtension) else {
                throw .badRequest("invalid_model_source", "--plan expects one .mlpackage or compiled .mlmodelc")
            }
            do {
                let source = url.pathExtension == "mlpackage"
                let compiled = source ? try await MLModel.compileModel(at: url) : url
                defer { if source { try? FileManager.default.removeItem(at: compiled) } }
                result["ane_preferred_operations"] = .int(try await CoreMLDecisionCapability.plannedOperations(at: compiled))
            } catch { throw ServiceError(.unavailable, "coreml_plan_rejected", String(describing: error)) }
        }
        Output.data(.object(result), format: format)
        return .success
    }
}
