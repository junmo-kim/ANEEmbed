---
name: aneembed
description: >
  Local text and image embeddings with EmbeddingGemma 2 on the Apple Neural Engine (ANEEmbed, macOS).
  Embed texts or images into one 768-d space (Matryoshka 512/256/128), compare two inputs by cosine
  similarity, count tokens, or run an OpenAI-compatible /v1/embeddings server, without a cloud API and
  without using the GPU. Reach for this when an agent needs semantic search, deduplication, clustering or
  retrieval over local data, needs to embed a query and documents consistently, or needs to check how
  similar two texts or a text and an image are. Triggers: embed text, embedding vector, semantic
  similarity, cosine similarity, vector search, RAG embeddings, OpenAI embeddings endpoint locally,
  image embedding, text-to-image search, token count for EmbeddingGemma. Apple Silicon + macOS 15+.
user_invocable: true
---

# aneembed

`aneembed` embeds texts and images on the Neural Engine. Put the binary on your `PATH` (README → Install)
and install the model once with `aneembed models install`. Output is YAML by default; add
`--format json` to parse. Errors go to stderr as `error` / `reason` / `detail` / `hint` with an exit code.

## Which command

| You need | Command |
| --- | --- |
| Is it set up? What to fix? | `aneembed doctor` |
| Embed a search query | `aneembed embed --prompt query "<query>"` |
| Embed documents for an index | `aneembed embed --prompt document "<doc>" ...` or JSONL: `aneembed embed --stdin --prompt document < docs.jsonl` |
| Embed an image | `aneembed embed --image <path>` |
| Similarity of two inputs | `aneembed similarity "<a>" "<b>"` (`--image-a` / `--image-b` for images) |
| Count tokens before sending | `aneembed tokenize "<text>"` |
| Many calls / a long session | keep `aneembed serve` (HTTP) or `aneembed mcp` (stdio) running |

## Rules that matter

- **Prefixes**: for retrieval, embed queries with `--prompt query` and documents with `--prompt document`
  (or `input_type` over HTTP/MCP). Without it the text is embedded exactly as given. Mixing prefixed and
  unprefixed vectors in one index hurts search quality.
- **Limits**: texts over the server's `max_context_length` (default 2048 tokens) are rejected with
  `context_length_exceeded`, never truncated. Split long documents into chunks first.
- **Vectors** are L2-normalized; use the dot product as cosine similarity. With `--dimensions 256` the
  vector is the re-normalized prefix; compare only vectors of the same size.
- **Cold start**: a one-off command loads the model each time (seconds). For loops, use `serve` or `mcp`.
- Images are resized by the official recipe (at most 280 soft tokens); any size and aspect ratio is fine.

## HTTP (aneembed serve, default http://127.0.0.1:18180)

```bash
curl -s localhost:18180/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["first", "second"], "input_type": "document", "dimensions": 256}'
curl -s -X QUERY 'localhost:18180/embed_text?input_type=query&format=json' -H 'Content-Type: text/plain' --data-binary 'my query'
curl -s -X QUERY 'localhost:18180/embed_image?format=json' -H 'Content-Type: image/jpeg' --data-binary @photo.jpg
```

`GET /health` is 503 while the model loads (the first start after install or a macOS update compiles for
the Neural Engine, which can take minutes); wait for 200.

## MCP

`aneembed mcp` (stdio) exposes `embed_text` (`text` or `texts`, `input_type`, `dimensions`,
`encoding_format`), `embed_image` (`data` base64, `images`, or a local `path`), and `similarity`
(`a`/`b` texts or `a_image`/`b_image`/`a_path`/`b_path`). Failures come back as `isError` results with a
structured reason; fix the input as the `reason` says rather than retrying unchanged.
