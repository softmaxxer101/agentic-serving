# Agentic Serving Lab — Phase 2



Benchmarking `Qwen3.8-27B` (NVFP4) on **dual RTX 5090 32 GB** under agentic serving workloads, using the [`trie`](https://github.com/Applied-Compute/trie) harness.



This repo holds the raw artifacts for **~49 runs** across **26 experiments** — the full ladder from baseline validation to overload, long-context, and a compound winner config. The workflow: `trie` drives realistic agent sessions (tool-call turns, retries, prompt growth) against a vLLM server, and each run dumps `vllm.log`, `metrics-after.txt`, and console output.



---



## Hardware & stack



| Component | Spec |

|---|---|

| GPU | 2 × NVIDIA RTX 5090 (32 GB, SM120, NVFP4-capable) |

| Model | `Qwen3.8-27B` quantized to NVFP4 |

| Serving | vLLM with tensor-parallel size 2 |

| Harness | [`trie`](https://github.com/Applied-Compute/trie) |

| Repo | [`yuro-py/agentic-serving-lab` / `phase-2`](https://github.com/yuro-py/agentic-serving-lab/tree/main/phase-2) |



Baseline recipe (all flags not overridden by a run's `EXTRA_ARGS`):



```bash

vllm serve unsloth/Qwen3.8-27B-NVFP4 \

  --tensor-parallel-size 2 \

  --max-model-len 262144 \

  --kv-cache-dtype fp8 \

  --enforce-eager \

  --language-model-only \

  --reasoning-parser qwen3 \

  --enable-auto-tool-choice \

  --tool-call-parser qwen3_coder \

  --enable-prompt-tokens-details

```



`--enforce-eager` is mandatory on a single 5090 and is kept on for the TP2 baseline in these runs; E20 tests lifting it.



---



## Workloads



Three synthetic workloads drive the experiments. All sessions include at least one tool call, as agentic traffic requires.



| File | Shape | Purpose |

|---|---|---|

| `workloads/light.jsonl` | Low concurrency, short prompts, single tool turn | Speculation and precision experiments where the per-step cost is the signal |

| `workloads/grow.jsonl` | Multi-turn prompts that grow across the session | Prefix-caching and long-context pressure |

| `workloads/burst.jsonl` | High fan-out short sessions | Batching saturation, scheduling, overload |



Every workload is validated before a run:



```bash

python tools/check_workload.py workloads/<file>.jsonl --max-model-len 262144

```



---



## Experiment index



| ID | Purpose | Workload | Key flags | Runs |

|---|---|---|---|---|

| E0 | Baseline validation | light | *(none)* | `e0-c1-rep1`, `e0-c1-rep2` |

| E1 | MTP speculation on/off | light | `--speculative-config {"method":"mtp","num_speculative_tokens":2}` | `e1-mtp-c1`, `e1-mtp-c8`, `e1-off-c1`, `e1-off-c8` |

| E2 | Linear kernel: flashinfer_cutlass vs marlin | burst | `--linear-backend flashinfer_cutlass` vs default | `e2-ficutlass-c8`, `e2-marlin-c8` |

| E3 | KV cache precision: nvfp4 vs fp8 | grow | `--kv-cache-dtype nvfp4` vs `fp8` | `e3-nvfp4-c4`, `e3-fp8-c4` |

| E4 | Batching / saturation knee | burst | `--max-num-seqs 64 --max-num-batched-tokens 16384` | `e4-c8`, `e4-c16`, `e4-c32` |

| E5 | Thinking budget: low vs xhigh | light | `--default-chat-template-kwargs {"reasoning_effort":"…"}` | `e5-low-c2`, `e5-xhigh-c2` |

| E6 | Prefix caching on/off | grow | `--no-enable-prefix-caching` | `e6-on-c4`, `e6-off-c4` |

| E7 | Scheduling policy: priority vs fcfs | burst | `--scheduling-policy priority` vs default | `e7-prio-c8`, `e7-prio-c16`, `e7-fcfs-c8`, `e7-fcfs-c16` |

| E8 | KV offloading to host RAM | grow | `--kv-offloading-size 32 --kv-offloading-backend native` | `e8-kvoff-c8`, `e8-kvoff-off-c8` |

| E11 | Long-context ceiling | grow | `--max-model-len 32768` / `65536` / `131072` / `262144` | `e11-len32k`, `e11-len64k`, `e11-len128k`, `e11-len262k` |

| E12 | GPU memory utilization sweep | burst | `--gpu-memory-utilization 0.85` / `0.90` / `0.93` | `e12-gmu085`, `e12-gmu090`, `e12-gmu093` |

| E13 | Language-model-only: text vs vision | light | `--language-model-only` vs full model | `e13-lmo-c8`, `e13-vision-c8` |

| E14 | Batch token budget: 2k / 4k / 8k / 16k | burst | `--max-num-seqs 64 --max-num-batched-tokens {2048…16384}` | `e14-bt2048` … `e14-bt16384` |

| E15 | `max-num-seqs` ceiling: 16 / 32 / 64 / 128 | burst | `--max-num-seqs {16…128}` | `e15-seq16` … `e15-seq128` |

| E16 | Chunked prefill on/off | burst | `--no-enable-chunked-prefill` | `e16-chunk-on-c16`, `e16-chunk-off-c16` |

| E17 | Async scheduling on/off | light | `--async-scheduling` vs default | `e17-async-c4`, `e17-async-c8`, `e17-nasync-c8` |

| E18 | MTP token-count sweep: 1 / 2 / 3 / 5 | light | `--speculative-config {"num_speculative_tokens": N}` | `e18-mtp1` … `e18-mtp5` |

| E19 | MTP × concurrency crossover | burst | MTP on at c=16, c=32 | `e19-mtp-c16`, `e19-mtp-c32` |

| E21 | Overload curve: 64 / 128 / 256 | burst | *(defaults)* | `e21-burst-c64`, `e21-burst-c128` |

| E25 | Hybrid-attention prefix-cache depth at c=16 | burst | cache on/off | `e25-off-c16` |



E9, E10, E20, E22, E23, E24, and E26 are specified in the ladder but not yet committed to this repo (see *Not yet run* below).



---



## Artifact layout



Every run lives under `phase-2/runs/<run-id>/` and is self-contained:



```

runs/

└── e1-mtp-c8-rep1/

    ├── vllm.log            # server log, including non-default args at startup

    ├── metrics-after.txt   # /metrics scrape at end of run

    ├── console.log         # trie output (benchmark complete line is authoritative)

    └── results.json        # per-session aggregates and percentiles

```



**Sanity check after every run:**



```bash

grep -E 'non-default args|GPU KV cache size|OutOfMemoryError|EngineDeadError' runs/<run-id>/vllm.log | head

grep -E 'failed_requests|completion_tok_per_s|ttft_s_p50|ttft_s_p99|ttfat_s_p50|cache_hit_rate_pct_mean' runs/<run-id>/console.log | tail -1

```



`failed_requests=0` and `completed=N/N` are required for a valid run. Anything else means the run is contaminated — rerun or record as a finding.



---



## What we're measuring



| Metric | Why it matters for agentic serving |

|---|---|

| `ttft_s_p50/p99` | Time to first token — user-perceived latency at the start of a turn |

| `ttfat_s_p50/p99` | Time to **first action token** — when the agent can start a tool call |

| `completion_tok_per_s` | Aggregate throughput |

| `spec_decode_num_accepted_tokens` / `…_draft_tokens` | MTP draft acceptance rate (Qwen3.8-27B recipe: ~0.79–0.90) |

| `cache_hit_rate_pct_mean` | Prefix-cache effectiveness on growing agent context |

| `vllm:num_requests_waiting` | Queue depth — graceful degradation vs preemption storm |

| `vllm:num_preemptions_total` | KV-pressure onset under overload |



`ttfat_s_p50/p99` matters more than `ttft` for agent workloads: the agent can't issue its tool call until the first action token arrives, and every model turn pays that latency.



---



## Not yet run (specified in the ladder)



| ID | Purpose | Blocked on |

|---|---|---|

| E9 | TP2 vs 2×TP1 replicas | Manual multi-server launch |

| E10 | Routing across replicas | E9 replicas up |

| E20 | CUDA graphs on/off | `run_trie_noeager.sh` harness variant |

| E22 | Client-disconnect resilience | Harness exit-metrics support |

| E23 | Weight offload (`--cpu-offload-gb`) | Optional, host RAM available |

| E24 | Long-context burst at 64k prompts | `workloads/long.jsonl` |

| E26 | Compound winner | Down-selection from E0–E25 |



---



## Reproducing a run



Each experiment is one `run_trie.sh` invocation:



```bash

WORKLOAD=workloads/burst.jsonl \

CONCURRENCY=8 \

DURATION=60 \

OUT_DIR=runs/e2-ficutlass-c8-rep1 \

EXTRA_ARGS="--linear-backend flashinfer_cutlass" \

./run_trie.sh

```



`run_trie.sh` handles model download, vLLM launch, `trie` invocation, and artifact collection. The baseline vLLM flags live in that script; `EXTRA_ARGS` are appended verbatim.



---



## Caveats and known issues



- **2×RTX 5090 is SM120, not Hopper.** Some kernels (flashinfer_cutlass linear, nvfp4 KV) fail to start on SM12x with an error. Those runs are recorded and the leg dropped — the failure is the finding, not a benchmark.

- **`--enforce-eager` is on for all runs.** Graph capture on a single 5090 crashes; E20 (not yet committed) is the test of whether 2×5090 avoids that.

- **PCIe all-reduce.** TP2 on a consumer board is a real cost. `light.jsonl` at c=1 and c=8 measures it against a hypothetical single-GPU baseline.

- **Host RAM 200+ GB.** E8 and E23 exercise host swap on this machine; the numbers won't transfer to a 64 GB box.

- **No sticky routing.** `trie` and vLLM don't coordinate prefix-aware routing across replicas; E10 would show this as a hit-rate collapse.



---



## Reading the results



Per-experiment findings live in `reports/` (one Markdown file per experiment, keyed by ID). The top-level takeaways — which flags actually move `ttfat_s_p99` on this hardware, whether MTP pays for itself past c=8, and the compound config that beats baseline — are summarized in `reports/SUMMARY.md`.



---



## Repo links



- Harness: [Applied-Compute/trie](https://github.com/Applied-Compute/trie)
