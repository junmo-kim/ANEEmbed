# Changelog

## v0.1.0

First release. EmbeddingGemma 2 text and image embeddings on the Apple Neural Engine from one dependency-free Swift binary: an OpenAI-compatible `/v1/embeddings` server, MCP over stdio and HTTP, and CLI commands. Nothing runs on the GPU and idle memory stays around 200 MB; texts up to 8192 tokens run entirely on the Neural Engine, and every function is checked against reference vectors before it serves. See the README for setup, limits and measurements.
