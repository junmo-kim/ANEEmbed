// HTTP routes of `aneembed serve`, independent of the socket layer:
//
//   POST  /v1/embeddings (/embeddings)   OpenAI-compatible embeddings
//   GET   /v1/models (/models)           the model id and its limits
//   GET   /health                        200 ready ("ok", or "degraded" when loaded functions are not served
//                                        or the self-test was skipped), 503 loading or failed
//   POST  /mcp                           MCP JSON-RPC (one message per request)
//   QUERY /{tool}?<options>              one-shot tool call (RFC 10008 QUERY, as in macvis): the body is the
//                                        raw input (UTF-8 text for embed_text, image bytes for embed_image)
//                                        or, with Content-Type application/json, the tool arguments
import Foundation

public struct Router: Sendable {
    public let state: ServerState
    public let modelName: String
    /// Server log: the details of internal errors, which responses do not carry.
    public let log: @Sendable (String) -> Void

    public init(state: ServerState, modelName: String, log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.state = state
        self.modelName = modelName
        self.log = log
    }

    /// The error as a client sees it (generic for 500/503), with an internal one's detail logged.
    func published(_ error: ServiceError, _ request: HTTPRequest) -> ServiceError {
        if error.kind == .internal { log("\(request.method) \(request.path): \(error.description)") }
        return error.publicFacing
    }

    public func handle(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/health"), ("HEAD", "/health"):
            return health()
        case ("GET", "/v1/models"), ("GET", "/models"):
            return models()
        case ("POST", "/v1/embeddings"), ("POST", "/embeddings"):
            return await embeddings(request)
        case ("POST", "/mcp"):
            let handler = MCPHandler(state: state, allowPaths: false, waitForModel: false, publicErrors: true, log: log)
            guard let response = await handler.handle(raw: request.body) else { return HTTPResponse(status: 202) }
            return .json(200, response)
        case ("QUERY", let path):
            return await query(request, tool: String(path.dropFirst()))
        case (_, "/health"), (_, "/v1/models"), (_, "/models"), (_, "/v1/embeddings"), (_, "/embeddings"), (_, "/mcp"):
            return .openAIError(ServiceError(.badRequest, "method_not_allowed", "\(request.method) is not allowed on \(request.path)"))
                .with(status: 405)
        default:
            return .openAIError(ServiceError(.notFound, "not_found", "no route for \(request.method) \(request.path)"))
        }
    }

    func health() -> HTTPResponse {
        switch state.status {
        case .ready(let service):
            let info = service.runtime.info
            guard info.degraded else { return .json(200, ["status": "ok"]) }
            var body: JSONObject = ["status": "degraded", "max_context_length": .int(service.maxTokens),
                                    "max_image_tokens": .int(service.maxImageTokens)]
            if !info.notServed.isEmpty { body["not_served"] = .array(info.notServed.map(JSONValue.string)) }
            if info.selfTestSkipped { body["selftest"] = "skipped" }
            return .json(200, .object(body))
        case .loading: return .json(503, ["status": "loading"])
        case .failed(let error):
            return .json(503, ["status": "error", "reason": .string(error.reason), "detail": .string(error.publicFacing.message)])
        }
    }

    func models() -> HTTPResponse {
        var model: JSONObject = [
            "id": .string(modelName), "object": "model", "owned_by": "local",
            "dimensions": .array(Vectors.dimensions.map { .int($0) }), "image_preprocessing": "hf-gemma4-280",
        ]
        if case .ready(let service) = state.status {
            model["max_context_length"] = .int(service.maxTokens)
            model["max_image_tokens"] = .int(service.maxImageTokens)
            model["max_inputs"] = .int(service.limits.maxInputs)
            model["max_images"] = .int(service.limits.maxImages)
        }
        return .json(200, ["object": "list", "data": [.object(model)]])
    }

    func embeddings(_ request: HTTPRequest) async -> HTTPResponse {
        do throws(ServiceError) {
            // The service first: its limits bound the parse (no image is decoded beyond --max-images), and a server
            // that is still loading answers 503 without parsing.
            let service = try state.service()
            let parsed = try EmbeddingsRequest.parse(request.body, limits: service.limits)
            let result: EmbeddingResult
            switch parsed.input {
            case .texts(let texts): result = try await service.embedTexts(texts, inputType: parsed.inputType)
            case .images(let images): result = try await service.embedImages(images)
            }
            return .json(200, raw: EmbeddingsResponse.body(result, request: parsed, model: modelName))
        } catch {
            return .openAIError(published(error, request))
        }
    }

    func query(_ request: HTTPRequest, tool: String) async -> HTTPResponse {
        let format = request.queryValue("format").flatMap(OutputFormat.init(rawValue:)) ?? .yaml
        func reply(_ status: Int, _ value: JSONValue) -> HTTPResponse {
            HTTPResponse(status: status,
                         headers: [("Content-Type", format == .json ? "application/json" : "application/yaml; charset=utf-8")],
                         body: Data((value.render(as: format) + "\n").utf8))
        }
        guard Tools.names.contains(tool) else {
            let error = ServiceError(.notFound, "unknown_tool", "unknown tool \(tool); tools: \(Tools.names.joined(separator: ", "))")
            return reply(404, error.structured)
        }
        guard let contentType = request.header("Content-Type")?.lowercased() else {
            return reply(400, ServiceError.badRequest("missing_content_type", "QUERY needs a Content-Type header").structured)
        }
        do throws(ServiceError) {
            var arguments = JSONObject()
            if contentType.hasPrefix("application/json") {
                let parsed: JSONValue
                do {
                    parsed = try JSONValue.parse(request.body)
                } catch {
                    throw ServiceError.badRequest("invalid_json", error.description)
                }
                guard let object = parsed.objectValue else {
                    throw ServiceError.badRequest("invalid_json", "the JSON body must be an object of tool arguments")
                }
                arguments = object
            } else {
                switch tool {
                case "embed_text":
                    guard let text = String(validating: request.body, as: UTF8.self) else {
                        throw ServiceError.badRequest("invalid_utf8", "the embed_text body must be UTF-8 text")
                    }
                    arguments["text"] = .string(text)
                case "embed_image": arguments["data"] = .string(request.body.base64EncodedString())
                default:
                    throw ServiceError.badRequest("json_required", "\(tool) takes its arguments as a JSON body")
                }
            }
            for (name, value) in request.query where name != "format" { arguments[name] = .string(value) }
            let service = try state.service()
            let output = try await Tools.call(tool, arguments: arguments, context: ToolContext(service: service, allowPaths: false))
            return reply(200, output)
        } catch {
            return reply(error.httpStatus, published(error, request).structured)
        }
    }
}

extension HTTPResponse {
    func with(status: Int) -> HTTPResponse {
        var copy = self
        copy.status = status
        return copy
    }
}
