#!/usr/bin/env bash
set -euo pipefail

# Contracts that can regress even when a release build compiles and links: launch, version, help,
# doctor output, structured errors and exit codes, stdio MCP, and the HTTP listener lifecycle.
# Uses the fake runtime, so it needs neither a model nor a Neural Engine (CI runners have neither).
#
# Usage: Tests/Integration/release-binary-launch-smoke.sh [path-to-aneembed]

binary=${1:-.build/release/aneembed}
binary=$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/aneembed-smoke.XXXXXX")
pid=""
cleanup() {
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  rm -rf "$tmpdir"
}
trap cleanup EXIT
fail() { echo "smoke test failed: $*" >&2; exit 1; }
export ANEEMBED_RUNTIME=fake ANEEMBED_DATA_DIR="$tmpdir/data"

version=$("$binary" --version)
[[ "$version" =~ ^aneembed\ [0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "unexpected version output: $version"
"$binary" --help >/dev/null

"$binary" doctor --format json >"$tmpdir/doctor.json" 2>"$tmpdir/doctor.err" || fail "doctor exited nonzero"
python3 - "$tmpdir/doctor.json" <<'PY' || fail "doctor output failed validation"
import json, sys
doctor = json.load(open(sys.argv[1]))
missing = {"aneembed", "macos", "macos_build", "neural_engine", "model", "compile_cache", "ready", "problems"} - doctor.keys()
if missing:
    raise SystemExit(f"doctor is missing {sorted(missing)}")
PY

"$binary" embed "smoke test" --dimensions 128 --format json >"$tmpdir/embed.json" 2>/dev/null || fail "embed exited nonzero"
python3 - "$tmpdir/embed.json" <<'PY' || fail "embed output failed validation"
import json, math, sys
out = json.load(open(sys.argv[1]))
vector = out["data"][0]["embedding"]
assert len(vector) == 128 and abs(math.sqrt(sum(x * x for x in vector)) - 1) < 1e-4, out
PY

set +e
"$binary" embed x --dimensions 300 >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 64 ] && grep -q 'invalid_dimensions' "$tmpdir/err" || fail "bad --dimensions: exit $code"
set +e
"$binary" embed "$(head -c 3000 /dev/zero | tr '\0' 'x')" >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'context_length_exceeded' "$tmpdir/err" || fail "over-long text: exit $code"

printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"embed_text","arguments":{"text":"hi","dimensions":256}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"embed_text","arguments":{"dimensions":256}}}' \
  | "$binary" mcp >"$tmpdir/mcp.jsonl" 2>"$tmpdir/mcp.err" || fail "mcp exited nonzero"
python3 - "$tmpdir/mcp.jsonl" <<'PY' || fail "mcp output failed validation"
import json, sys
responses = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
assert set(responses) == {1, 2, 3, 4}, responses.keys()
assert responses[1]["result"]["serverInfo"]["name"] == "aneembed"
assert [t["name"] for t in responses[2]["result"]["tools"]] == ["embed_text", "embed_image", "similarity"]
ok = json.loads(responses[3]["result"]["content"][0]["text"])
assert len(ok["data"][0]["embedding"]) == 256 and not responses[3]["result"].get("isError")
err = json.loads(responses[4]["result"]["content"][0]["text"])
assert responses[4]["result"]["isError"] and err["reason"] == "missing_input", err
PY

port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$binary" serve --port "$port" 2>"$tmpdir/serve.err" &
pid=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 0.1
done
curl -sf "http://127.0.0.1:$port/health" | grep -q '"ok"' || { cat "$tmpdir/serve.err" >&2; fail "serve did not become healthy"; }
python3 "$(dirname "${BASH_SOURCE[0]}")/serve-contract.py" "http://127.0.0.1:$port" || fail "serve contract"
kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""
echo "release binary smoke test passed"
