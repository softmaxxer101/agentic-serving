# Agentic Serving Lab Kit

Pre-built harness for the **RTX 2050 4 GB + Qwen3-0.6B + vLLM 0.32.0** agentic serving lab.
You never edit harness code. You predict, run, read, explain.

## Install (once)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install vllm==0.32.0          # default 0.32.0 wheels target CUDA 13.0 (driver >= 580); older driver: use the cu129 build from the release page
pip install -e .                  # installs the `lab` command
export LAB_HOME=$PWD GPU=2050 KEY=lab-secret-key HF_HOME=$HOME/hf HF_HUB_ENABLE_HF_TRANSFER=1
lab selftest                      # proves the kit works (mock server, no GPU)
lab doctor                        # machine capture, bandwidth, KV bytes/token, roofline, flag detection
```

## The loop
```bash
lab notes E3          # creates notes/E3.md; write numeric predictions under "Prediction"
lab exp E3 --dry-run  # what will run and how long
lab exp E3            # refuses to run until the prediction is written; restarts server per run; resumes if interrupted
lab report E3         # tables from raw logs (+ vs-baseline and noise-floor flags)
lab plot E3
```
Run `lab exp E0` first: it measures the **noise floor** used by every later report.
`lab exp E5` measures saturation (SAT); E9, E12, E13 use it (auto-run if missing).
E13: `lab exp E13 --winner '--max-num-seqs 8' --winner '--no-enable-prefix-caching'`.

## Manual single runs
```bash
lab up base --arg "--max-num-seqs 8" --drop --enforce-eager
lab smoke
lab run mix --conc 8 --n 60            # or: --load open --rate 0.4
lab compare runs/adhoc/<a>/rep1 runs/adhoc/<b>/rep1
lab down
```
Workloads: mix, agent-code, agent-research, quick-tool, longdoc. All live (no replay).
If steps per session are near 1 (a 0.6B model often answers without calling tools), add `--tool-choice required-first`.

## What each run folder contains
config.json, cmd.txt, server.log, startup-facts.txt, requests.jsonl (per request), gpu.csv, dmon.csv, metrics.csv (1 Hz vLLM metrics),
cpu.csv (per core, server, client), vram-after.txt, meta.json, DONE, plus startup_failed.json (with diagnosis),
probe.json, copy_fresh.json, toolong.json, disconnect.json where relevant.

## vLLM 0.32.0 specifics baked in
- Flags and metric names are discovered at runtime (`vllm serve --help=all`, /metrics); unknown flags are dropped and recorded in config.json as findings.
- `max_tokens` is now an output ceiling, not reserved memory: E12 "toolong" records what 0.32.0 actually does.
- NVFP4 KV cache now reaches SM8x: E1 tries `--kv-cache-dtype nvfp4` if your build lists it.
- `--api-server-count` dispatches to the least-loaded API worker: E11.
- A GPU n-gram speculator was added; its method name is unverified here, so E7 tries `ngram_gpu` and a rejection is a finding.
- `lab doctor --warm` runs `vllm download-kernels`.

## Caveats
GPU-dependent parts (server start, nvidia-smi samplers, real metrics) could not be run in the build environment; the workload, runner,
analysis, reports and plots were verified against the built-in mock. Treat the first real `lab smoke` and `lab exp E0` as the integration test,
and read any FINDING / DRIFT lines the reports print.
