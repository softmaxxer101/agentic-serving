Agentic Serving Benchmark

This project studies inference serving for multi-step agentic workloads across three phases. It begins with a customized local benchmark on Qwen3-0.6B, then moves to the upstream Trie trace-replay harness, and finally applies Trie to a larger Qwen3-27B deployment on stronger hardware.

The project is designed as a progressive inference-engineering study:

```text
Phase 0: Learn and validate serving behavior locally
    ↓
Phase 1: Reproduce the study with the upstream Trie harness
    ↓
Phase 2: Scale the same methodology to Qwen3-27B
```

## Project Goals

- Understand how agentic workloads stress inference servers.
- Measure TTFT, inter-token latency, end-to-end latency, throughput, cache reuse, and tail behavior.
- Compare vLLM serving configurations fairly using repeatable workloads.
- Learn the difference between request-level and session-level performance.
- Validate the benchmark methodology with a small model before scaling to a larger model.
- Produce evidence for future hardware, model, and serving-engine decisions.

## Phases

### Phase 0 Customized Harness on Qwen3-0.6B

Phase 0 uses a custom Python harness developed for this project and vLLM serving `Qwen/Qwen3-0.6B` on an NVIDIA RTX 3050 Laptop GPU with 6 GiB VRAM.

The harness contains:

- Deterministic agentic trace generation.
- Scripted tool calls and tool results.
- Streaming OpenAI-compatible requests.
- Closed-loop and open-loop replay.
- Prefix-cache measurements.
- Reasoning and tool-call parsing checks.
- Prometheus metric scraping.
- GPU and server-log collection.
- Run summaries with latency percentiles and throughput.
- A mock server for testing the driver without using the GPU.

The main Phase 0 questions are:

- How does context growth affect memory and latency?
- How much prefix caching is achieved?
- How does concurrency affect throughput and tail latency?
- What happens when GPU memory or context limits are exceeded?
- Do compilation, CUDA graphs, speculative decoding, offloading, and async scheduling help?

The main Phase 0 model and server configuration are:

```text
Model: Qwen/Qwen3-0.6B
Engine: vLLM 0.31.0
GPU: NVIDIA RTX 3050 Laptop GPU, 6 GiB
Context limit: 8192 tokens
GPU utilization target: 0.75
Max sequences: 16
Reasoning parser: qwen3
Tool parser: hermes
```

### Phase 1 Trie Harness in This Repository

Phase 1 uses the upstream [Applied-Compute Trie repository](https://github.com/Applied-Compute/trie), cloned into the `phase-1/trie` folder of this project.

Trie means **trace replay inference evaluation**. It is a lightweight harness for replaying synthetic workloads derived from production traces against OpenAI-compatible inference endpoints. The upstream project is designed to exercise agentic traffic against backends such as vLLM, SGLang, and TensorRT-LLM.

Unlike the customized Phase 0 harness, Phase 1 uses Trie’s workload format and measurement model. A trace describes:

- Number of tool-use turns.
- Initial prompt length.
- Assistant response length for every turn.
- Tool-result length for every turn.
- Simulated tool latency.
- Final assistant response length.

One trace produces multiple completion requests: one for each tool-use turn and one final response request.

Trie reports metrics including:

- End-to-end trace latency.
- TTFT.
- Time to first answer token after hidden tool turns.
- Decode throughput.
- Inter-token latency.
- Server-reported cache hit rate.
- Eligible cache hit rate.
- Prompt-token throughput.
- Cached and uncached prompt-token throughput.
- Completion-token throughput.
- Overall and steady-state throughput.

Phase 1 is a methodology-reproduction phase. The goal is to compare the customized Phase 0 harness against a maintained, general-purpose trace-replay harness using the same or closely matched serving endpoint and workload assumptions.

### Phase 2 Trie Harness on Qwen3-27B

Phase 2 keeps the Trie-based workload and measurement methodology but changes the serving scale:

```text
Model: Qwen3-27B
Harness: Trie
Backend: OpenAI-compatible inference server
Hardware: stronger GPU or multi-GPU system
```

Phase 2 investigates how conclusions from the small-model experiment change for a substantially larger model.

The focus shifts toward:

- Tensor or pipeline parallel serving.
- Model loading and startup time.
- GPU memory distribution.
- KV-cache capacity.
- Multi-GPU communication.
- Steady-state prompt and completion throughput.
- Queueing under higher concurrency.
- Cache reuse at larger scale.
- Cost and performance per GPU.

Phase 2 should not be treated as a direct speed comparison with Phase 0. The model size, hardware, parallelism, tokenizer behavior, and server configuration will differ. The comparison is primarily methodological and architectural.


## Phase 0 Experiment Coverage

The completed Phase 0 study included:

- Hardware and environment inspection.
- Deterministic corpus and trace generation.
- Mock-server validation.
- Baseline vLLM serving.
- Concurrency scaling.
- Prefix-cache experiments with different project counts.
- Batch-size comparisons.
- GPU-memory utilization tests.
- N-gram speculative decoding.
- Torch compilation and CUDA graphs.
- PCIe and NCCL inspection.
- CPU weight offloading.
- FCFS, priority, and async scheduling.
- Overload behavior.
- Long-context failures.
- Controlled out-of-memory failures.
- Final async-scheduling run.

The baseline produced approximately 34.8 output tokens per second with a cached-prompt fraction of approximately 0.808. The CPU-offload run fell to approximately 3.6 output tokens per second. Async scheduling produced the strongest result in the small scheduler comparison, with approximately 22.2 output tokens per second and a TTFT p50 of approximately 0.234 seconds.

The study also demonstrated that the 6 GiB GPU could not safely support every requested configuration. For example, vLLM rejected a utilization target of 0.85 because only about 5.03 GiB was free while approximately 5.10 GiB was requested.

## Reproducibility Principles

Every phase should follow the same principles:

1. Record the model, tokenizer, server version, GPU, driver, CUDA version, and configuration.
2. Use a fixed workload seed when comparing configurations.
3. Keep the request count, concurrency, and duration equal across comparisons.
4. Separate warm-up from steady-state measurement.
5. Run every important configuration more than once.
6. Record errors, truncation, cancellations, and incomplete runs.
7. Confirm that the server is idle before each benchmark.
8. Save raw request results and server metrics before generating summaries.
9. Compare p50, p95, and p99 instead of relying only on averages.
10. Distinguish synthetic workload accounting from server-reported token accounting.

## Core Metrics

| Metric | Meaning |
|---|---|
| TTFT | Time from request submission until the first streamed token |
| TTFAT | Time from the beginning of a hidden agent trace until the first token of the final answer |
| ITL | Client-observed time between streamed tokens |
| End-to-end latency | Time required to complete a request or complete trace |
| Decode throughput | Generated tokens per second during decoding |
| Cache hit rate | Server-reported cached prompt tokens divided by prompt tokens |
| Eligible cache hit rate | Cache reuse measured only over tokens expected to be reusable |
| Prompt throughput | Prompt tokens processed per second |
| Completion throughput | Generated completion tokens per second |
| Steady-state throughput | Throughput after excluding the initial warm-up period |

## Phase 1 Validation Plan

Before moving to Phase 2, Phase 1 should establish that:

- Trie can reach the local OpenAI-compatible endpoint.
- The tokenizer used by Trie matches the tokenizer used by the server.
- The server returns cached-token details when cache measurement is enabled.
- The same workload produces stable results across repeated runs.
- Trie’s output can be collected and summarized consistently.
- Phase 0 and Phase 1 metrics are clearly labeled rather than mixed without qualification.

Example Trie invocation from the upstream documentation:

```bash
uv run trie \
  workload_path=workload.jsonl \
  endpoint=http://localhost:8000/v1 \
  model=Qwen/Qwen3-0.6B \
  concurrency=8 \
  duration=300 \
  stream=True \
  num_gpus=1
```

The exact command should be adjusted to the actual Phase 1 workload and local model name.

## Phase 2 Validation Plan

Before running Qwen3-27B, record:

- Number and type of GPUs.
- Total and free VRAM per GPU.
- CUDA and driver versions.
- vLLM and Transformers versions.
- Tensor-parallel or pipeline-parallel configuration.
- Model loading time.
- Context length.
- KV-cache configuration.
- Maximum concurrent sequences.
- Whether prefix caching is enabled.
- Whether the workload uses streaming.

Phase 2 results should be compared using:

- Per-GPU throughput.
- Total throughput.
- TTFT and TTFAT.
- Decode TPOT or ITL.
- Cache hit rate.
- GPU memory utilization.
- Queueing behavior.
- Failed and cancelled traces.
- Cost or resource efficiency where available.

## Limitations

This project is an inference-engineering study, not a production capacity certification. Phase 0 uses synthetic and scripted tool behavior, a small model, and one laptop GPU. Phase 1 improves the benchmark methodology by using Trie, but still depends on the quality and representativeness of the selected workload. Phase 2 will be more realistic in model scale, but results will still depend on hardware, parallelism, traffic distribution, and backend configuration.

The benchmark should not be used to claim that one serving configuration is universally best. It is intended to show which configuration is best for a defined workload, hardware platform, and service objective.

## Future Extensions

- Connect real search, database, file, and retrieval tools.
- Add realistic production trace samples.
- Add burst, cancellation, disconnect, retry, and soak workloads.
- Add per-request priority classes.
- Add multi-GPU and multi-instance comparisons.
- Add confidence intervals and repeated-trial analysis.
- Export results to dashboards.
- Compare vLLM, SGLang, and TensorRT-LLM under the same Trie workload.
- Evaluate quantized and full-precision Qwen3-27B deployments.
- Measure cost per completed trace and cost per generated token.

## License and Attribution

Phase 1 uses the upstream Trie project from Applied Compute. Refer to the cloned repository for its Apache-2.0 license and attribution requirements:

https://github.com/Applied-Compute/trie

The customized Phase 0 harness and project-specific analysis should retain their own authorship and licensing information.
