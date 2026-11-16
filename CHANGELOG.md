# Changelog

## Unreleased

`POST /tokenize` and `POST /detokenize` in llama-server's format (`add_special`, `parse_special`, `with_pieces`), so clients that measure or split their texts with llama-server's tokenizer API work unchanged; for EmbeddingGemma 2 the responses match llama-server's. Content is limited to 1 MiB, and invalid requests are refused with a `400` where llama-server is lenient. `docs/PERFORMANCE.md` now compares the M4, M5 Max and M5 Ultra in one table, measured the same way. See the README for details.

## v0.1.0

First release. EmbeddingGemma 2 text and image embeddings on the Apple Neural Engine from one dependency-free Swift binary: an OpenAI-compatible `/v1/embeddings` server, MCP over stdio and HTTP, and CLI commands. Nothing runs on the GPU and idle memory stays around 200 MB; texts up to 8192 tokens run entirely on the Neural Engine, and every function is checked against reference vectors before it serves. See the README for setup, limits and measurements.
