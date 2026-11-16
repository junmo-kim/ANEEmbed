"""HTTP contract of a running `aneembed serve`, stdlib only. Works against the fake runtime (CI) and a
real model; with a real model, pass --reference <json> ({"texts": [...], "vectors": [...]}, e.g. from Hugging
Face fp32) to also check accuracy.

Usage: python3 -I serve-contract.py http://127.0.0.1:18180 [--reference ref.json] [--min-cosine 0.9999]
"""
import argparse, base64, concurrent.futures, http.client, json, math, socket, struct, sys, urllib.error, urllib.request, zlib

parser = argparse.ArgumentParser()
parser.add_argument("base")
parser.add_argument("--reference")
parser.add_argument("--min-cosine", type=float, default=0.9999)
args = parser.parse_args()
BASE = args.base.rstrip("/")
failures = []


def call(path, payload=None, method=None, headers=None, raw=None):
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode())
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
            return r.status, (json.loads(body) if body[:1] in (b"{", b"[") else body.decode())
    except urllib.error.HTTPError as e:
        body = e.read() or b"{}"
        return e.code, (json.loads(body) if body[:1] in (b"{", b"[") else body.decode())


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def png_url(width, height, rgb=(200, 30, 90)):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def image_item(url):
    return {"content": [{"type": "image_url", "image_url": {"url": url}}]}


s, b = call("/health")
check("health 200", s == 200 and b == {"status": "ok"}, b)
s, b = call("/v1/models")
model = b["data"][0] if s == 200 and len(b.get("data", [])) == 1 else {}
max_tokens = model.get("max_context_length", 0)
check("models lists one id with limits", max_tokens >= 512 and model.get("dimensions") == [768, 512, 256, 128], b)
check("model object has OpenAI's fields", model.get("object") == "model" and isinstance(model.get("created"), int)
      and isinstance(model.get("owned_by"), str), b)

s, b = call("/v1/embeddings", {"input": "task: search result | query: hello"})
v = b["data"][0]["embedding"] if s == 200 else []
check("single string -> 768 unit floats", len(v) == 768 and abs(norm(v) - 1) < 1e-3, s)
check("usage reports tokens", b.get("usage", {}).get("prompt_tokens", 0) > 2, b.get("usage"))

batch = [f"query number {i} " + "word " * (i * 3) for i in range(20)]
s, b = call("/v1/embeddings", {"input": batch, "model": "anything"})
check("batch keeps order and count", s == 200 and [d["index"] for d in b["data"]] == list(range(len(batch))))
singles = [call("/v1/embeddings", {"input": t})[1]["data"][0]["embedding"] for t in batch[:5]]
sims = [cos(singles[i], b["data"][i]["embedding"]) for i in range(5)]
check("packed batch matches single calls", min(sims) > 0.999, f"min cos {min(sims):.6f}")

s, b64 = call("/v1/embeddings", {"input": batch[0], "encoding_format": "base64"})
vec = struct.unpack("<768f", base64.b64decode(b64["data"][0]["embedding"]))
check("base64 decodes to the float vector", max(abs(x - y) for x, y in zip(vec, singles[0])) < 1e-6)
s, d = call("/v1/embeddings", {"input": batch[0], "dimensions": 256})
v = d["data"][0]["embedding"]
head = singles[0][:256]
check("dimensions=256 is the renormalized prefix", len(v) == 256 and abs(norm(v) - 1) < 1e-5 and cos(v, head) > 0.99999)

for name, payload in [("bad dimensions", {"input": "x", "dimensions": 300}), ("token-id input", {"input": [1, 2, 3]}),
                      ("empty list", {"input": []}), ("bad encoding", {"input": "x", "encoding_format": "int8"})]:
    s, b = call("/v1/embeddings", payload)
    check(f"{name} -> 400 OpenAI error", s == 400 and b.get("error", {}).get("type") == "invalid_request_error", b)

too_long = "token " * (max_tokens + 10)
s, b = call("/v1/embeddings", {"input": ["ok", too_long]})
check(f"over {max_tokens} tokens -> 400, not truncated",
      s == 400 and b["error"].get("code") == "context_length_exceeded" and str(max_tokens) in b["error"]["message"], b)
def tokens(text):
    return call("/v1/embeddings", {"input": text})[1]["usage"]["prompt_tokens"]


if max_tokens > 600:
    per_unit = (tokens("long " * 40) - tokens("long " * 20)) / 20  # tokenizer-agnostic sizing
    long_text = "long " * int((min(max_tokens, 2048) - 40) / per_unit)
    check("long text is over 512 tokens", 512 < tokens(long_text) <= max_tokens, tokens(long_text))
    s, b = call("/v1/embeddings", {"input": ["short", long_text, "short too"]})
    check("mixed short + long batch keeps order", s == 200 and [d["index"] for d in b["data"]] == [0, 1, 2], s)

if model.get("max_image_tokens", 0) > 0:
    small, big, tiny, strip = png_url(320, 240), png_url(1600, 1200), png_url(1, 1), png_url(3000, 10)
    s, b = call("/v1/embeddings", {"input": [image_item(small)]})
    v = b["data"][0]["embedding"] if s == 200 else []
    check("image item -> 768 unit floats, usage 266 + 4", len(v) == 768 and b["usage"]["prompt_tokens"] == 270, s)
    s, b = call("/v1/embeddings", {"input": [image_item(big), image_item(tiny), image_item(strip)]})
    check("large, 1x1 and 3000x10 images -> 266 + 256 + 280 tokens (+ 4 each)",
          s == 200 and b["usage"]["prompt_tokens"] == 266 + 256 + 280 + 12, s)
    if s == 200:
        check("same picture at two sizes -> nearly the same vector", cos(b["data"][0]["embedding"], v) > 0.999)
    for name, payload in [("9 image items", {"input": [image_item(small)] * 9}),
                          ("mixed text and image", {"input": [image_item(small), "text"]}),
                          ("non-data image URL", {"input": [image_item("http://x/y.png")]}),
                          ("undecodable image", {"input": [image_item("data:image/png;base64,AAAA")]})]:
        s, b = call("/v1/embeddings", payload)
        check(f"{name} -> 400", s == 400, s)

text = "tokenize me, détokenize me"
s, plain = call("/tokenize", {"content": text})
s2, framed = call("/tokenize", {"content": text, "add_special": True})
check("POST /tokenize: add_special off by default, on adds <bos>/<eos> (the usage count)",
      s == 200 and s2 == 200 and len(framed["tokens"]) == len(plain["tokens"]) + 2 == tokens(text), (plain, framed))
s, b = call("/detokenize", {"tokens": plain.get("tokens", [])})
check("POST /detokenize gives the text back", s == 200 and b.get("content") == text, b)
s, b = call("/tokenize", {"content": "é\u0ff0", "with_pieces": True})  # U+0FF0: byte tokens for EmbeddingGemma 2
pieces = [t["piece"] for t in b.get("tokens", [])]
check("with_pieces joins back to the text, with byte-list pieces", s == 200 and any(isinstance(p, list) for p in pieces)
      and b"".join(bytes(p) if isinstance(p, list) else p.encode() for p in pieces) == "é\u0ff0".encode(), b)
s, b = call("/tokenize", {"content": "x" * (1 << 20 | 1)})
check("/tokenize content over 1 MiB -> 413 content_too_long", s == 413 and b["error"].get("code") == "content_too_long", s)
s, b = call("/detokenize", {"tokens": [-1]})
check("detokenize of a non-id -> 400 invalid_token", s == 400 and b["error"].get("code") == "invalid_token", b)

s, b = call("/embed_text?format=json&dimensions=128", method="QUERY", raw=b"hello", headers={"Content-Type": "text/plain"})
check("QUERY /embed_text raw body", s == 200 and len(b["data"][0]["embedding"]) == 128, s)
s, b = call("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
check("POST /mcp tools/list", s == 200 and len(b["result"]["tools"]) == 3, s)
s, _ = call("/nope")
check("unknown route -> 404", s == 404)

conn = http.client.HTTPConnection(BASE.split("//")[1])
statuses = []
for _ in range(3):
    conn.request("GET", "/health")
    r = conn.getresponse()
    r.read()
    statuses.append(r.status)
check("keep-alive: three requests on one connection", statuses == [200, 200, 200], statuses)

try:
    conn = http.client.HTTPConnection(BASE.split("//")[1], timeout=10)
    conn.request("HEAD", "/health")
    r = conn.getresponse()
    head = (r.status, r.getheader("Content-Length"), r.read())
    conn.request("GET", "/health")
    r = conn.getresponse()
    after = (r.status, r.read())
    conn.close()
    check("HEAD: GET's headers, no body; the next response on the connection parses",
          head == (200, "15", b"") and after == (200, b'{"status":"ok"}'), (head, after))
except (http.client.HTTPException, OSError) as e:
    check("HEAD: GET's headers, no body; the next response on the connection parses", False, repr(e))

def raw_exchange(data, timeout=10):
    """Sends raw bytes, returns everything the server sends until it closes the connection."""
    host, port = BASE.split("//")[1].rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall(data)
        received = b""
        while chunk := sock.recv(65536):
            received += chunk
        return received


smuggled = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"
for name, headers in [("CL.CL", b"Content-Length: 0\r\nContent-Length: %d\r\n" % len(smuggled)),
                      ("TE.CL", b"Transfer-Encoding: chunked\r\nContent-Length: %d\r\n" % len(smuggled))]:
    try:
        reply = raw_exchange(b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\n" + headers + b"\r\n" + smuggled)
        check(f"{name} framing -> one 400, then close", reply.startswith(b"HTTP/1.1 400") and reply.count(b"HTTP/1.1") == 1,
              reply[:80])
    except OSError as e:
        check(f"{name} framing -> one 400, then close", False, repr(e))

with concurrent.futures.ThreadPoolExecutor(32) as pool:
    statuses = list(pool.map(lambda q: call("/v1/embeddings", {"input": q})[0], batch * 3 + batch[:4]))
check("64 concurrent requests all 200", statuses.count(200) == 64, statuses.count(200))

if args.reference:
    ref = json.load(open(args.reference))
    s, b = call("/v1/embeddings", {"input": ref["texts"]})
    sims = [cos(d["embedding"], r) for d, r in zip(b["data"], ref["vectors"])]
    check(f"accuracy vs reference (min cos > {args.min_cosine})", min(sims) > args.min_cosine,
          f"min {min(sims):.6f} mean {sum(sims) / len(sims):.6f}")

print("FAILED:", failures if failures else "none")
sys.exit(1 if failures else 0)
