#!/usr/bin/env bash
set -euo pipefail
EXTRA_ARGS="${EXTRA_ARGS:-}"

unset BASE_SAFE VLLM_ARGS BASE 2>/dev/null || true


# ------------------------------------------------------------------
# config — override with environment variables
# ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# vLLM flags. For trie cache metrics you MUST include
# --enable-prompt-tokens-details.
VLLM_ARGS="${VLLM_ARGS:---max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --enable-prompt-tokens-details}"

WORKLOAD="${WORKLOAD:-workloads/class_A.jsonl}"
CONCURRENCY="${CONCURRENCY:-8}"
DURATION="${DURATION:-300}"          # trie admission window, seconds
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)}"
API_KEY="${API_KEY:-}"               # leave empty if server has no auth
TRIE_BIN="${TRIE_BIN:-uv run trie}"

mkdir -p "$OUT_DIR"

VLLM_LOG="$OUT_DIR/vllm.log"
DMON_CSV="$OUT_DIR/dmon.csv"
METRICS_BEFORE="$OUT_DIR/metrics-before.txt"
METRICS_AFTER="$OUT_DIR/metrics-after.txt"
TRIE_LOG="$OUT_DIR/trie.log"

VLLM_PID=""
DMON_PID=""

cleanup() {
  echo ">>> cleanup"
  [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 1. launch vLLM
# ------------------------------------------------------------------
echo ">>> launching vLLM on :$PORT"
# shellcheck disable=SC2086
# shellcheck disable=SC2086

vllm serve "$MODEL" --port "$PORT" ${API_KEY:+--api-key "$API_KEY"} \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 \
  --enforce-eager \
  --enable-prompt-tokens-details \
  $EXTRA_ARGS \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!

# ------------------------------------------------------------------
# 2. wait for /health
# ------------------------------------------------------------------
echo ">>> waiting for /health"
for _ in $(seq 1 400); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo ">>> vLLM ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!!! vLLM died; tail of log:" >&2
    tail -50 "$VLLM_LOG" >&2
    exit 1
  fi
  sleep 2
done

# ------------------------------------------------------------------
# 3. start nvidia-smi dmon
# ------------------------------------------------------------------
echo ">>> starting nvidia-smi dmon -> $DMON_CSV"
nvidia-smi dmon \
  -i "$GPU_ID" \
  -s pucvmet \
  -d 1 \
  > "$DMON_CSV" 2>&1 &
DMON_PID=$!

# ------------------------------------------------------------------
# helper: curl /metrics with optional bearer token
# ------------------------------------------------------------------
metrics_curl() {
  if [[ -n "${API_KEY:-}" ]]; then
    curl -sf -H "Authorization: Bearer ${API_KEY}" \
      "http://localhost:${PORT}/metrics"
  else
    curl -sf "http://localhost:${PORT}/metrics"
  fi
}

# ------------------------------------------------------------------
# 4. dump pre-run /metrics
# ------------------------------------------------------------------
echo ">>> dumping pre-run /metrics"
metrics_curl > "$METRICS_BEFORE" || true

# ------------------------------------------------------------------
# 5. invoke trie
# ------------------------------------------------------------------
echo ">>> running trie (concurrency=$CONCURRENCY, duration=$DURATION)"
# shellcheck disable=SC2086
$TRIE_BIN \
  workload_path="$WORKLOAD" \
  endpoint="http://localhost:${PORT}/v1" \
  model="$MODEL" \
  concurrency="$CONCURRENCY" \
  duration="$DURATION" \
  stream=False \
  | tee "$TRIE_LOG"

# ------------------------------------------------------------------
# 6. dump post-run /metrics
# ------------------------------------------------------------------
echo ">>> dumping post-run /metrics"
metrics_curl > "$METRICS_AFTER" || true

# ------------------------------------------------------------------
# 7. stop dmon early; trap handles vLLM
# ------------------------------------------------------------------
kill "$DMON_PID" 2>/dev/null || true
DMON_PID=""

echo ">>> done. artifacts in $OUT_DIR"
