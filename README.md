# ANEEmbed (`aneembed`)

[![CI](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml/badge.svg)](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform: macOS 15+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-lightgrey)

EmbeddingGemma 2 embeddings that run **only on the Apple Neural Engine**, as a single **100% Swift
binary with no third-party dependencies** (Foundation, Network, CoreML, ImageIO, Accelerate).

- **OpenAI drop-in**: `POST /v1/embeddings` with text and images, `encoding_format` float/base64,
  Matryoshka `dimensions` 768/512/256/128.
- **Leaves the GPU alone**: nothing runs on the GPU, so a local LLM keeps it to itself.
- **Accurate**: measured against the original fp32 Hugging Face model (see [Accuracy](#accuracy)).
- **Small**: resident footprint around 130 MB (the ANE side is not counted in it, see [Pitfalls](#known-pitfalls)).
- **CLI, HTTP and MCP** from one binary: `aneembed embed`, `aneembed serve`, `aneembed mcp`.

> **Status: pre-release.** The converted model package (uniform bucket functions up to 2048 tokens,
> long-text functions up to 8192) is being produced and is not published yet. Everything that does not
> depend on it is implemented and tested; numbers marked _TBD_ will be filled in from measurements.

## Install

Apple Silicon, macOS 15 or later.

**Homebrew**

```bash
brew install junmo-kim/tap/aneembed
```

**mise**

```bash
mise use -g github:junmo-kim/ANEEmbed
```

**From source** (Xcode 16 or later)

```bash
swift build -c release && cp .build/release/aneembed /usr/local/bin/
```

Then install the model. This downloads a pinned revision, checks every file's SHA-256 and compiles the
functions for the Neural Engine once (the OS caches the result, so later starts take about half a second
per function instead of minutes):

```bash
aneembed models install
aneembed doctor        # Neural Engine, model files, compile cache: what is ready and what to fix
```

The release binary is ad-hoc signed. A tarball downloaded with a browser carries a quarantine attribute;
clear it with `xattr -d com.apple.quarantine aneembed` (Homebrew, mise and curl do not set it).

## Usage

```bash
aneembed serve                                    # http://127.0.0.1:18180
curl -s localhost:18180/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "task: search result | query: how do I reset my router?"}'

aneembed embed "first text" "second text" --format json
aneembed embed --prompt query "how do I reset my router?"    # adds the model's query prefix
aneembed embed --image photo.jpg --dimensions 256
cat docs.jsonl | aneembed embed --stdin --prompt document   # JSONL in, JSONL out, batched
aneembed similarity "a cat on a mat" --image-b cat.jpg
aneembed tokenize "hello world" --show-tokens
```

One-off commands load the model each time (seconds even with a warm cache). For regular use keep
`serve` or `mcp` running.

### MCP

```json
{ "mcpServers": { "aneembed": { "command": "aneembed", "args": ["mcp"] } } }
```

Tools: `embed_text`, `embed_image`, `similarity`. The same tools are reachable over HTTP as
`POST /mcp` and as one-shot `QUERY /{tool}?<options>` with the raw text or image bytes as the body.

## API

| Endpoint | |
|---|---|
| `POST /v1/embeddings` | OpenAI-compatible. `input`: string, list of strings, or list of llama-server image items `{"content": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]}` (texts and images are not mixed in one request) |
| `GET /v1/models` | the model id, `max_context_length`, `max_image_tokens`, `dimensions`, `image_preprocessing` |
| `GET /health` | `200` ready, `503` while loading (the first start compiles for the Neural Engine) or after a load failure |
| `POST /mcp` | MCP JSON-RPC, one message per request |
| `QUERY /{tool}` | `embed_text` (UTF-8 body), `embed_image` (image body), `similarity` (JSON body); YAML by default, `?format=json` |

Request fields of `/v1/embeddings`:

- `encoding_format`: `float` (default) or `base64` (little-endian float32).
- `dimensions`: 768 (default), 512, 256 or 128: the first N values, re-normalized.
- `input_type` (extension): `query` or `document` adds the model's task prefix
  (`task: search result | query: ` / `title: none | text: `). Without it texts are embedded as given,
  like llama-server, and callers add the prefix themselves.
- `model`, `user`: accepted and ignored; the response names the configured model.

Texts over the configured limit are rejected with `400 context_length_exceeded`, never truncated.
`usage.prompt_tokens` counts real tokens including `<bos>`/`<eos>`; an image counts its soft tokens + 4.
Errors use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

## Configuration

Every option is a flag and an environment variable of the same name (flag > environment > default).
`aneembed serve --print-config` shows each value and where it came from.

| Flag | Environment | Default | |
|---|---|---|---|
| `--model-dir` | `ANEEMBED_MODEL_DIR` | `<data-dir>/models/current` | model package directory |
| `--data-dir` | `ANEEMBED_DATA_DIR` | `~/Library/Application Support/ANEEmbed` | installed models and state |
| `--host` | `ANEEMBED_HOST` | `127.0.0.1` | anything else exposes an unauthenticated API |
| `--port` | `ANEEMBED_PORT` | `18180` | |
| `--max-tokens` | `ANEEMBED_MAX_TOKENS` | `2048` | up to 8192; only functions up to it are loaded and compiled |
| `--model-name` | `ANEEMBED_MODEL_NAME` | `embeddinggemma-2` | |
| `--max-body` | `ANEEMBED_MAX_BODY` | `72` (MiB) | 8 images of 8 MiB plus JSON |
| `--max-inputs` / `--max-images` | `ANEEMBED_MAX_INPUTS` / `ANEEMBED_MAX_IMAGES` | `2048` / `8` | per request |
| `--query-concurrency` / `--image-concurrency` / `--long-concurrency` | `ANEEMBED_*_CONCURRENCY` | `8` / `2` / `2` | separate lanes so long documents and images cannot hold short queries back |
| `--runtime` | `ANEEMBED_RUNTIME` | `coreml` | `fake` runs without a model, for testing integrations |

## Trade-offs

Measured on _TBD_ (M4, M5 Ultra). The default favours latency and compile time; the model's limit is
8192 tokens and the cost of raising `--max-tokens` is shown here rather than hidden.

| `--max-tokens` | p50 latency at the limit | first compile | resident memory |
|---|---|---|---|
| 512 | _TBD_ | _TBD_ | _TBD_ |
| 2048 (default) | _TBD_ (about 180 ms on M5 Ultra in the prototype) | _TBD_ (about 30 s) | _TBD_ |
| 8192 | _TBD_ | _TBD_ | _TBD_ |

| Workload | ANEEmbed | _TBD_ (llama.cpp Q8 on GPU) |
|---|---|---|
| one short query | _TBD_ | _TBD_ |
| 64 short queries, one request | _TBD_ | _TBD_ |
| one image (280 tokens) | _TBD_ (about 0.45–0.5 s on M4 in the prototype) | n/a |

## Accuracy

The reference is the original model in fp32 through Hugging Face (`sentence-transformers` /
`transformers`), not another quantized port. Method: embed the same inputs with both, report the cosine
similarity per input (min / mean), text by token-length band over public multilingual documents, images
over a public image set at the official 280-token canvas. The prototype measured 0.99997 for text and
0.9997 for images; the table below will be re-measured on public data with this binary.

| Input | n | min cos | mean cos |
|---|---|---|---|
| text ≤ 512 tokens | _TBD_ | _TBD_ | _TBD_ |
| text 513–2048 | _TBD_ | _TBD_ | _TBD_ |
| text 2049–8192 | _TBD_ | _TBD_ | _TBD_ |
| images | _TBD_ | _TBD_ | _TBD_ |

The tokenizer is checked token for token against Hugging Face `tokenizers` on 12,420 generated texts
(every id must match), and image preprocessing bit for bit against `Gemma4ImageProcessor`.

## Known pitfalls

- **Graph-constant masks miscompile on the ANE.** The sliding-window band of Gemma's local layers is an
  input (`local_bias`), not a constant in the graph.
- **Variable shapes fall back to the CPU.** Every function has fixed shapes (buckets of 64…2048 tokens);
  several short texts are packed into one call with block-diagonal attention.
- **The first compile takes minutes**, and the OS caches it per model path and OS build. `models install`
  does it once; a macOS update invalidates it (`doctor` reports `stale`, `serve` re-warms before it reports
  healthy). Do not move the model directory.
- **Memory reporting**: the ANE side does not show in the process footprint, and freed malloc pages only
  become reclaimable about two minutes later, so short-lived measurements overstate memory.
- **llama.cpp GGUF**: text vectors are compatible; image preprocessing in llama.cpp differs from the
  official recipe, so image vectors are not interchangeable.

## Running as a service

```xml
<!-- ~/Library/LaunchAgents/com.github.junmo-kim.aneembed.plist -->
<plist version="1.0"><dict>
  <key>Label</key><string>com.github.junmo-kim.aneembed</string>
  <key>ProgramArguments</key><array><string>/opt/homebrew/bin/aneembed</string><string>serve</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/aneembed.log</string>
</dict></plist>
```

There is no authentication or TLS. To serve other machines, keep `--host 127.0.0.1` and put a reverse
proxy in front.

## Build and test

```bash
swift build -c release
swift test                                   # unit, golden and fake-runtime contract tests
Tests/Integration/release-binary-launch-smoke.sh .build/release/aneembed
ANEEMBED_TEST_MODEL_DIR=/path/to/model swift test --filter ModelTests   # real model, needs the ANE
```

## License

MIT for this code. The EmbeddingGemma 2 model and tokenizer are Google's, under Apache-2.0; see
[NOTICE](NOTICE).
