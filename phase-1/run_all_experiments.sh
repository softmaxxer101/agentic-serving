#!/usr/bin/env bash
# run_all_experiments.sh
# Runs all E0-E13 experiments in order. Prints section header, divider, and live output.
# Expected failure: E13 c32 (burst overload) may 400-error on overflow rows — not fatal.

set -uo pipefail

divider() {
  printf '\n============================================================\n'
  printf '  %s\n' "$1"
  printf '============================================================\n'
}

# ---------------------------------------------------------------
divider "E0 — baseline scaling curve (light.jsonl, c1/2/4/8 × 3 reps)"
# ---------------------------------------------------------------
for rep in 1 2 3; do
  for c in 1 2 4 8; do
    VLLM_ARGS="$BASE" WORKLOAD=workloads/light.jsonl CONCURRENCY=$c DURATION=60 \
      OUT_DIR=runs/e0-c${c}-rep${rep} ./run_trie.sh
  done
done

# ---------------------------------------------------------------
divider "E3 — prefix caching on/off (light.jsonl, c4 × 3 reps)"
# ---------------------------------------------------------------
for rep in 1 2 3; do
  for mode in on off; do
    if [ "$mode" = on ]; then EXTRA_ARGS="--enable-prefix-caching"; else EXTRA_ARGS="--no-enable-prefix-caching"; fi
    EXTRA_ARGS="$EXTRA_ARGS" WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
      OUT_DIR=runs/e3-${mode}-rep${rep} ./run_trie.sh
  done
done

# ---------------------------------------------------------------
divider "E4 — batching knobs (light.jsonl, c8, 1 rep each)"
# ---------------------------------------------------------------
declare -A E4=(
  [e4-bt512]="--max-num-batched-tokens 512"
  [e4-bt2048]="--max-num-batched-tokens 2048"
  [e4-seq4]="--max-num-seqs 4"
  [e4-seq16]="--max-num-seqs 16"
  [e4-nochunk]="--no-enable-chunked-prefill"
)
for name in "${!E4[@]}"; do
  EXTRA_ARGS="${E4[$name]}" WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
    OUT_DIR=runs/$name-rep1 ./run_trie.sh
done

# ---------------------------------------------------------------
divider "E5a — KV capacity via gpu-memory-utilization (light.jsonl, c8)"
# ---------------------------------------------------------------
for g in 0.70 0.75 0.80 0.85; do
  EXTRA_ARGS="--gpu-memory-utilization $g" \
    WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
    OUT_DIR=runs/e5-gmu${g}-rep1 ./run_trie.sh
done

# ---------------------------------------------------------------
divider "E5b — quantization fp8 weight-only (light.jsonl, c8)"
# ---------------------------------------------------------------
EXTRA_ARGS="--quantization fp8" \
  WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e5-fp8w-rep1 ./run_trie.sh

# ---------------------------------------------------------------
divider "E8 — CUDA graphs vs eager (light.jsonl, c1 and c4)"
# ---------------------------------------------------------------
echo "--- E8.1 graphs c1 ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=1 DURATION=60 \
  OUT_DIR=runs/e8-graphs-c1-rep1 ./run_trie_noeager.sh

echo "--- E8.2 graphs c4 ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
  OUT_DIR=runs/e8-graphs-c4-rep1 ./run_trie_noeager.sh

echo "--- E8.3 eager c1 control ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=1 DURATION=60 \
  OUT_DIR=runs/e8-eager-c1-rep1 ./run_trie.sh

echo "--- E8.4 eager c4 control ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
  OUT_DIR=runs/e8-eager-c4-rep1 ./run_trie.sh

# ---------------------------------------------------------------
divider "E6 — ngram speculation on/off (light.jsonl, c1 & c4)"
# ---------------------------------------------------------------
echo "--- E6.1 ngram3 ---"
for c in 1 4; do
  EXTRA_ARGS='--speculative-config {"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    WORKLOAD=workloads/light.jsonl CONCURRENCY=$c DURATION=60 \
    OUT_DIR=runs/e6-ng3-c${c}-rep1 ./run_trie.sh
done

echo "--- E6.2 off (control) ---"
for c in 1 4; do
  WORKLOAD=workloads/light.jsonl CONCURRENCY=$c DURATION=60 \
    OUT_DIR=runs/e6-off-c${c}-rep1 ./run_trie.sh
done

# ---------------------------------------------------------------
divider "E13 — overload on burst.jsonl (c8/16/32; c32 may 400-error)"
# ---------------------------------------------------------------
for c in 8 16 32; do
  echo "--- E13 burst c${c} ---"
  WORKLOAD=workloads/burst.jsonl CONCURRENCY=$c DURATION=60 \
    OUT_DIR=runs/e13-burst-c${c}-rep1 ./run_trie.sh || echo "  [E13 c${c} exited nonzero — expected for c32]"
done

# ---------------------------------------------------------------
divider "E15 — KV offload to host RAM (burst.jsonl, c8)"
# ---------------------------------------------------------------
echo "--- E15.1 kv-offload OFF (control) ---"
WORKLOAD=workloads/burst.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e15-kv-off-c8-rep1 ./run_trie.sh

echo "--- E15.2 kv-offload ON (1 GiB host pool) ---"
EXTRA_ARGS="--kv-offloading-size 1 --kv-offloading-backend native" \
  WORKLOAD=workloads/burst.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e15-kv-on-c8-rep1 ./run_trie.sh

# ---------------------------------------------------------------
divider "E16 — Weight offload to CPU (light.jsonl, c8)"
# ---------------------------------------------------------------
echo "--- E16.1 weight-offload OFF (control) ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e16-woff-c8-rep1 ./run_trie.sh

echo "--- E16.2 weight-offload ON (0.5 GiB) ---"
EXTRA_ARGS="--cpu-offload-gb 0.5" \
  WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e16-woff-on-c8-rep1 ./run_trie.sh

# ---------------------------------------------------------------
divider "E17 — Scheduling policy & async scheduling (light.jsonl)"
# ---------------------------------------------------------------
echo "--- E17.1 fcfs c4 (control) ---"
WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
  OUT_DIR=runs/e17-fcfs-c4-rep1 ./run_trie.sh

echo "--- E17.2 priority c4 (expected no-op, trie sends no priorities) ---"
EXTRA_ARGS="--scheduling-policy priority" \
  WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
  OUT_DIR=runs/e17-prio-c4-rep1 ./run_trie.sh

echo "--- E17.3 async scheduling c4 ---"
EXTRA_ARGS="--async-scheduling" \
  WORKLOAD=workloads/light.jsonl CONCURRENCY=4 DURATION=60 \
  OUT_DIR=runs/e17-async-c4-rep1 ./run_trie.sh

echo "--- E17.4 async scheduling c8 ---"
EXTRA_ARGS="--async-scheduling" \
  WORKLOAD=workloads/light.jsonl CONCURRENCY=8 DURATION=60 \
  OUT_DIR=runs/e17-async-c8-rep1 ./run_trie.sh

divider "ALL EXPERIMENTS COMPLETE"
