#!/usr/bin/env bash

set -uo pipefail

for cfg in "$@"; do
    for rep in 1 2 3; do
        REP=$rep ./run_config.sh "$cfg"
    done
done
