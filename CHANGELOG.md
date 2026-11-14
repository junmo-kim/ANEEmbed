# Changelog

## Unreleased

First version, before the model package is published: the tokenizer (checked token for token against
Hugging Face `tokenizers`), the official image preprocessing (bit-exact against `Gemma4ImageProcessor`),
packing of texts into uniform fixed-shape bucket functions with incrementally built biases, the OpenAI
`/v1/embeddings` server, MCP over stdio and HTTP, and the `embed`, `similarity`, `tokenize`, `doctor` and
`models` commands. The Core ML backend follows the bucket contract but has not run against the final
package yet.
