#!/usr/bin/env bash
set -euo pipefail

source env/vars.sh

CFG=$1
source "configs/$CFG.env"

OUT="runs/$GPU/$EXP/$CFG/rep${REP:-1}"
mkdir -p "$OUT"
cp "configs/$CFG.env" "$OUT/config.env"

P1=""
P2=""
P3=""
SP=""

cleanup() {
    kill ${P1:-} ${P2:-} ${P3:-} ${SP:-} 2>/dev/null || true
    pkill -f 'vllm serve' 2>/dev/null || true
    sleep 8
    nvidia-smi --query-gpu=memory.used --format=csv,noheader \
        >> "$OUT/vram-after.txt" 2>/dev/null || true
}

trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
    --format=csv -l 1 > "$OUT/gpu.csv" &
P1=$!

nvidia-smi dmon -s pucvmet -d 1 > "$OUT/dmon.csv" &
P2=$!

T0=$(date +%s)

export SERVE_ARGS WL_ARGS ENVV GPU

(
    eval "exec env $ENVV vllm serve $MODEL $SERVE_ARGS --port 8000 --api-key $KEY"
) > "$OUT/server.log" 2>&1 &
SP=$!

for i in $(seq 1 400); do
    if curl -sf http://localhost:8000/health > /dev/null; then
        break
    fi

    if ! kill -0 "$SP" 2>/dev/null; then
        echo "server died; see $OUT/server.log"
        exit 1
    fi

    sleep 2
done

if ! curl -sf http://localhost:8000/health > /dev/null; then
    echo "server did not become healthy; see $OUT/server.log"
    exit 1
fi

STARTUP=$(( $(date +%s) - T0 ))
export STARTUP

python - > "$OUT/meta.json" <<'PYEOF'
import hashlib
import json
import os
import subprocess
import torch
import vllm

trace_path = "traces/M020.jsonl"

print(json.dumps({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": os.environ["GPU"],
    "startup_s": int(os.environ["STARTUP"]),
    "serve_args": os.environ["SERVE_ARGS"],
    "envv": os.environ["ENVV"],
    "wl": os.environ["WL_ARGS"],
    "trace": trace_path,
    "trace_sha": hashlib.sha256(open(trace_path, "rb").read()).hexdigest()
}))
PYEOF

grep -iE 'kv cache size|maximum concurrency|model loading took|prefix caching|chunked prefill|cuda graph' \
    "$OUT/server.log" > "$OUT/startup-facts.txt" || true

python harness/scrape.py \
    http://localhost:8000/metrics \
    "$OUT/metrics.csv" &
P3=$!

python harness/run.py $WL_ARGS \
    --out "$OUT/requests.jsonl" \
    2>&1 | tee "$OUT/client.log"

[ -n "${PROBE:-}" ] && \
    python harness/quality_probe.py "$OUT/probe.json"

echo "done $CFG"
