#!/usr/bin/env bash

BASE_3050='--max-model-len 8192 --gpu-memory-utilization 0.75 --max-num-seqs 16 --enforce-eager --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details'

BASE_2050='--max-model-len 4096 --gpu-memory-utilization 0.80 --max-num-seqs 16 --enforce-eager --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details'

if [ "$GPU" = "3050" ]; then
    BASE="$BASE_3050"
else
    BASE="$BASE_2050"
fi

WL='--trace traces/M020.jsonl --mode replay --load closed --conc 8 --n 30 --warmup 2'

mk() {
    printf 'EXP=%s\nSERVE_ARGS=%q\nWL_ARGS=%q\nENVV=%q\n' \
        "$2" \
        "$BASE $3" \
        "${4:-$WL}" \
        "${5:-}" \
        > "configs/$1.env"
}
