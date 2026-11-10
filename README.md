
```markdown

# Agentic Serving Benchmark



A reproducible benchmarking project for evaluating local LLM serving performance under agentic workloads.



The project studies how vLLM configuration choices affect latency, throughput, GPU memory usage, prefix caching, tool-call reliability, speculative decoding, scheduling, compilation, and overload behavior.



The current implementation uses Qwen3-0.6B served by vLLM on an NVIDIA RTX 3050 Laptop GPU. The design is intentionally modular so it can later scale to larger models, stronger GPUs, and production-like workloads.



## Project Goals



This project evaluates:



- Time to first token (TTFT)

- Inter-token latency (ITL)

- End-to-end request latency

- Session completion time

- Input and output token throughput

- Prefix-cache effectiveness

- Tool-call JSON validity

- Context-length failures

- GPU memory limits

- Scheduler behavior

- CUDA graph and torch compilation effects

- Speculative decoding

- CPU offloading

- Overload behavior

- Reproducibility of agentic request traces



The benchmark separates workload generation from model execution. This allows the same workload to be replayed across different server configurations.



## System Used



| Component | Value |

|---|---|

| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |

| GPU memory | 6 GiB |

| Driver | 610.88 |

| CUDA UMD | 13.3 |

| PCIe link | Gen4, x4 current, x16 maximum |

| Model | Qwen/Qwen3-0.6B |

| Serving engine | vLLM 0.31.0 |

| Python | 3.12 |

| Platform | Linux/WSL |

| Repository | `~/project/agentic-serving-lab` |



The GPU has approximately 5.03 GiB available to vLLM at startup because some memory is reserved by the system and runtime.



## Repository Layout



```text

agentic-serving-lab/

├── analysis/

│   ├── summarize.py

│   ├── hitrate.py

│   ├── compare_probe.py

│   └── *-summary.txt

├── configs/

│   ├── *.env

│   └── mk.sh

├── env/

│   ├── vars.sh

│   ├── gpu-summary.csv

│   ├── nvidia-smi.txt

│   ├── lscpu.txt

│   ├── free.txt

│   ├── corpus.sha

│   └── trace-hashes.txt

├── harness/

│   ├── common.py

│   ├── gen_trace.py

│   ├── check_trace.py

│   ├── run.py

│   ├── scrape.py

│   └── smoke.py

├── mock_server/

│   └── app.py

├── notes/

├── runs/

│   └── 3050/

├── traces/

│   ├── corpus.txt

│   ├── M.jsonl

│   ├── M020.jsonl

│   ├── M-p8.jsonl

│   ├── M-p16.jsonl

│   └── A/B/C/D.jsonl

├── run_config.sh

└── README.md

```



## Installation



Activate the existing project environment:



```bash

cd ~/project/agentic-serving-lab

source ~/project/venv/bin/activate

```



Verify the environment:



```bash

which python

which pip

python -m pip show vllm torch

python -c "import vllm; print(vllm.__version__)"

```



Expected vLLM version:



```text

0.31.0

```



Do not reinstall packages unnecessarily if they are already available in the environment.



## Environment Variables



Load the project variables:



```bash

source env/vars.sh

```



Important variables include:



- `MODEL`

- `KEY`

- `BASE_URL`

- `GPU`

- `MAX_MODEL_LEN`



Check them:



```bash

echo "$MODEL"

echo "$GPU"

echo "$BASE_URL"

```



The primary model is:



```text

Qwen/Qwen3-0.6B

```



## Starting the vLLM Server Manually



The standard server configuration is:



```bash

vllm serve "$MODEL" \

  --max-model-len 8192 \

  --gpu-memory-utilization 0.75 \

  --max-num-seqs 16 \

  --enforce-eager \

  --reasoning-parser qwen3 \

  --enable-auto-tool-choice \

  --tool-call-parser hermes \

  --enable-prompt-tokens-details \

  --port 8000 \

  --api-key "$KEY"

```



Check server health:



```bash

curl -s http://localhost:8000/health

```



Check metrics:



```bash

curl -s http://localhost:8000/metrics | head

```



Check GPU usage:



```bash

nvidia-smi

```



The server must be stopped before starting another configuration. The automated runner handles this automatically.



## Smoke Test



Run the smoke test against the running server:



```bash

python harness/smoke.py \

  | tee runs/smoke-test-output.txt

```



The smoke test checks:



1. Plain chat returns text.

2. Reasoning is separated from normal content.

3. Thinking can be disabled.

4. Tool calls are parsed.

5. Prompt-token details are returned.

6. Prefix caching can produce non-zero cached tokens on repeated prompts.



A successful tool test contains a parsed `tool_calls` list.



The calculator example is synthetic. The test prompt asks the model to use a calculator tool with the expression `123*45`; it is not an instruction from the user.



## Corpus



The corpus provides repeatable text for generated tool outputs:



```bash

wc -c traces/corpus.txt

sha256sum traces/corpus.txt | tee env/corpus.sha

```



The corpus is not intended to represent a production database. It is a deterministic workload source that makes experiments repeatable.



In production, tool results would come from real systems such as:



- Search services

- Databases

- File repositories

- APIs

- Retrieval systems

- Python execution workers



The corpus allows us to test serving behavior before integrating those external systems.



## Trace Generation



Generate the standard trace:



```bash

python harness/gen_trace.py \

  --n 300 \

  --seed 1 \

  --projects 4 \

  --scale 0.20 \

  --out traces/M020.jsonl

```



Validate it:



```bash

python harness/check_trace.py traces/M020.jsonl 8192 \

  | tee env/trace-check-M020.txt

```



The trace generator creates scripted agent conversations containing:



- System prompts

- Project context

- Tool calls

- Tool outputs

- Delays between steps

- Thinking and non-thinking workloads

- Final answers

- Repeated project contexts for prefix-cache testing



The tool calls and results are generated by the harness, not by the model. This makes replay deterministic.



## Determinism Check



Generate the same trace twice:



```bash

python harness/gen_trace.py \

  --n 300 --seed 1 --projects 4 \

  --out traces/M.jsonl



python harness/gen_trace.py \

  --n 300 --seed 1 --projects 4 \

  --out traces/M2.jsonl



sha256sum traces/M.jsonl traces/M2.jsonl \

  | tee env/trace-hashes.txt

```



Both hashes should be identical.



## Mock Server



The mock server tests the harness without using the GPU:



```bash

uvicorn mock_server.app:app --port 8000

```



In another terminal:



```bash

python harness/run.py \

  --trace traces/M020.jsonl \

  --mode replay \

  --load closed \

  --conc 8 \

  --n 30 \

  --warmup 2 \

  --out /tmp/mock.jsonl

```



The mock server is useful for verifying:



- Streaming response handling

- Closed-loop concurrency

- Open-loop request rates

- Session completion logging

- Error handling

- Metrics scraping



## Automated Experiments



The standard experiment runner is:



```bash

REP=1 ./run_config.sh CONFIG_NAME

```



Results are stored under:



```text

runs/3050/<experiment>/<configuration>/rep<repeat>/

```



Typical files include:



```text

requests.jsonl

server.log

metrics.csv

meta.json

config.env

vram-after.txt

```



Example:



```bash

REP=1 ./run_config.sh e0-base

```



## Result Summaries



Summarize a run:



```bash

python analysis/summarize.py \

  runs/3050/e0/e0-base/rep1/requests.jsonl

```



Summarize several runs:



```bash

python analysis/summarize.py \

  runs/3050/e0/e0-base/rep1/requests.jsonl \

  runs/3050/e11/e11-async/rep1/requests.jsonl \

  runs/3050/e14/e14-final/rep1/requests.jsonl

```



Important fields:



| Field | Meaning |

|---|---|

| `ttft_p50` | Median time to first token |

| `ttft_p99` | Tail time to first token |

| `itl_p50` | Median inter-token latency |

| `e2e_p50` | Median request latency |

| `sess_p50` | Median session duration |

| `out_tok_s` | Output throughput |

| `in_tok_s` | Input throughput |

| `json_ok` | Valid tool-call JSON fraction |

| `cached_frac` | Fraction of prompt tokens served from cache |

| `truncated` | Fraction ending because of token limits |

| `errors` | Number of failed records |



## Completed Experiments



### E0 — Baseline and Concurrency



Baseline configuration used the standard eager server settings.



Observed baseline:



| Metric | Result |

|---|---:|

| Sessions | 30 |

| Requests | 152 |

| Errors | 1 |

| Duration | 1068.5 s |

| TTFT p50 | 0.986 s |

| TTFT p99 | 3.716 s |

| ITL p50 | 0.183 s |

| E2E p50 | 44.973 s |

| E2E p99 | 109.010 s |

| Output throughput | 34.814 tok/s |

| Input throughput | 321.577 tok/s |

| Cached fraction | 0.808 |

| Truncated | 0.105 |



The baseline error and truncation were caused by context-length pressure. Some requests exceeded the usable context after reserving generation tokens.



### E3 — Prefix Caching



Prefix-cache experiments compared different project counts and cache settings.



The cache was effective, with cached fractions around 0.77–0.81 in most successful runs.



Increasing the number of projects reduced ideal cache reuse slightly because requests shared fewer identical prefixes.



### E4 — Batching



Batch-size experiments showed that batching changes the balance between:



- Throughput

- TTFT

- Tail latency

- GPU KV-cache usage



Higher batch capacity is not automatically better on a 6 GiB GPU because memory pressure and longer queues can increase tail latency.



### E5 — GPU Memory Utilization



A configuration using:



```text

--gpu-memory-utilization 0.85

```



failed during startup because:



```text

Free memory: 5.03 GiB

Requested memory: 5.10 GiB

```



A lower setting succeeded:



```text

--gpu-memory-utilization 0.80

```



This demonstrates that vLLM checks available memory at startup and rejects configurations that exceed the currently free GPU memory.



### E6 — N-gram Speculative Decoding



The n-gram speculative configuration used three draft tokens.



The server logs reported:



- Mean accepted draft length around 1.6–2.4 tokens

- Draft acceptance rates around 19.5%–45.5%

- Successful speculative decoding initialization



Speculative decoding was therefore active, but its benefit depends strongly on the workload and acceptance rate.



### E7 — Thinking On/Off



This experiment was not completed because the current `harness/run.py` does not support a `--think` command-line argument.



The smoke harness supports thinking behavior internally, but the full replay runner was not extended with a corresponding CLI flag.



### E8 — Torch Compilation and CUDA Graphs



The compiled configuration successfully used:



- vLLM compilation mode

- TorchInductor

- CUDA graphs



The server log showed:



```text

torch.compile took 35.05 s in total

```



CUDA graph capture completed successfully and used approximately 0.04 GiB.



Compilation adds startup time but may reduce repeated execution overhead after warm-up.



### E9 — NCCL and PCIe



The PCIe link was recorded as:



```text

Current generation: 4

Maximum generation: 4

Current width: x4

Maximum width: x16

```



NCCL initialized successfully, but the run used:



```text

world_size=1

```



Therefore, this experiment did not measure multi-GPU communication performance.



### E10 — CPU Offloading



CPU weight offloading completed successfully but was substantially slower than the baseline.



Observed offload result:



| Metric | Result |

|---|---:|

| Sessions | 5 |

| Requests | 18 |

| Errors | 0 |

| Duration | 827.1 s |

| TTFT p50 | 0.907 s |

| ITL p50 | 0.285 s |

| Output throughput | 3.614 tok/s |

| Cached fraction | 0.776 |



CPU offload is useful when the model cannot fit in GPU memory, but it introduces significant data-transfer overhead.



### E11 — Scheduling and Async Scheduling



The scheduler comparison used FCFS, priority scheduling, and async scheduling.



| Configuration | TTFT p50 | TTFT p99 | ITL p50 | E2E p50 | Output tok/s |

|---|---:|---:|---:|---:|---:|

| FCFS | 0.388 s | 0.733 s | 0.051 s | 3.217 s | 16.746 |

| Priority | 0.306 s | 0.634 s | 0.049 s | 2.497 s | 17.411 |

| Async | 0.234 s | 0.573 s | 0.037 s | 2.650 s | 22.172 |



The async scheduler produced the best TTFT and output throughput in this small test.



The priority run tested the scheduler policy itself. The request harness did not provide different per-request priority values.



### E12 — Multiple API Server Processes



The exact multi-API-server experiment was not available in vLLM 0.31.0. The installed version did not expose an `--api-server-count` option.



This experiment remains unavailable unless a different vLLM release or deployment architecture is used.



### E13 — Overload, Context Limits, and OOM



Overload testing increased request pressure while keeping the same small GPU.



The long-context test demonstrated predictable context failures when:



```text

prompt tokens + reserved output tokens > max-model-len

```



The forced OOM configuration also failed as expected:



```text

Free memory: 5.03 GiB

Requested memory: 5.88 GiB

```



This is a controlled startup validation failure rather than a runtime crash.



### E14 — Final Configuration



The final configuration used async scheduling and a larger small-run workload.



Observed result:



| Metric | Result |

|---|---:|

| Sessions | 10 |

| Requests | 45 |

| Errors | 0 |

| Duration | 581.5 s |

| TTFT p50 | 0.593 s |

| TTFT p99 | 2.305 s |

| ITL p50 | 0.080 s |

| E2E p50 | 19.567 s |

| E2E p99 | 53.621 s |

| Output throughput | 18.318 tok/s |

| Input throughput | 156.704 tok/s |

| JSON validity | 1.0 |

| Cached fraction | 0.790 |

| Truncated | 0.156 |



## Main Findings



1. Prefix caching works well for repeated project contexts.

2. Context length is a major limitation for multi-step agentic conversations.

3. GPU memory utilization must leave headroom for runtime allocations.

4. CPU offloading enables execution but can severely reduce throughput.

5. Async scheduling improved latency and throughput in the tested workload.

6. Torch compilation and CUDA graphs increase startup time but can improve steady-state execution.

7. Speculative decoding was active, but acceptance rates were workload-dependent.

8. A 6 GiB GPU is sufficient for controlled small-model experiments but not for unconstrained long-context serving.

9. Tool-call JSON remained valid in successful runs.

10. Results must be compared cautiously because some experiments used different request counts, concurrency levels, and trace sizes.



## Reproducing the Main Final Run



```bash

cd ~/project/agentic-serving-lab

source ~/project/venv/bin/activate

source env/vars.sh

source configs/mk.sh



mk e14-final e14 '--async-scheduling' \

  '--trace traces/M020.jsonl --mode replay --load closed --conc 2 --n 10 --warmup 1'



REP=1 ./run_config.sh e14-final



python analysis/summarize.py \

  runs/3050/e14/e14-final/rep1/requests.jsonl \

  | tee analysis/e14-final-summary.txt

```



## Troubleshooting



### Server dies during startup



Inspect the log:



```bash

tail -n 50 runs/3050/<experiment>/<config>/rep1/server.log

```



Check GPU memory:



```bash

nvidia-smi

```



Reduce memory pressure:



```text

--gpu-memory-utilization 0.70

--max-model-len 4096

--max-num-seqs 8

```



### Context-length errors



Check the trace:



```bash

python harness/check_trace.py traces/M020.jsonl 8192

```



Reduce workload scale:



```bash

python harness/gen_trace.py \

  --n 300 \

  --seed 1 \

  --projects 4 \

  --scale 0.20 \

  --out traces/M020.jsonl

```



Or reduce the generation-token budget.



### Missing summary data



Make sure the run completed and contains:



```text

run_start

session_done

run_end

```



Incomplete runs should not be included in aggregate summaries.



### Stale GPU memory



Check for leftover processes:



```bash

nvidia-smi

```



Stop stale vLLM processes before starting another run.



## Limitations



- The workload is synthetic and replay-based.

- The model is small and not representative of production model quality.

- Results were collected on one laptop GPU.

- Multi-GPU scaling was not measured.

- External tools were mocked rather than connected to real services.

- Some experiments used short speedrun workloads.

- Repeated runs are required for statistically stronger conclusions.

- The runner does not yet support all experiment-specific controls.

- E7 and E12 were not completed because the required interfaces were unavailable.



## Future Work



Possible extensions include:



- Test Qwen3-1.7B or larger models.

- Repeat experiments on stronger GPUs.

- Add real search and database tools.

- Add request-priority fields to the replay harness.

- Implement a proper thinking on/off comparison.

- Add burst and disconnect workloads.

- Add multi-GPU tensor-parallel experiments.

- Compare vLLM with another serving engine.

- Add confidence intervals and repeated trials.

- Export results to CSV or a dashboard.

- Add automated regression checks in CI.

- Compare local serving against a production inference endpoint.



## Project Status



The project successfully demonstrates a reproducible local agentic-serving benchmark with:



- Deterministic traces

- Mock-server validation

- GPU-backed vLLM serving

- Streaming request measurement

- Prometheus metric scraping

- Prefix-cache analysis

- Scheduling experiments

- Compilation experiments

- Speculative decoding

- Memory-failure characterization

- Final result aggregation



The benchmark is ready to evolve from a small local prototype into a larger serving-performance research project.

```



After saving it, verify:



```bash

cd ~/project/agentic-serving-lab

wc -l README.md

head -n 20 README.md

```
