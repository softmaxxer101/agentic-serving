# Agentic Serving Speedrun Results

## Hardware

- GPU: NVIDIA RTX 3050 Laptop GPU, 6 GB
- PCIe: Gen 4 x4
- Model: Qwen/Qwen3-0.6B
- vLLM: 0.31.0
- Context length: 8192 tokens

## Baseline

- 30 sessions
- 152 requests
- 1 error
- Output throughput: 34.8 tokens/s
- TTFT p50: 0.986 s
- Cached fraction: 80.8%
- Tool JSON validity: 100%
- Truncation: 10.5%

## Main findings

- Higher concurrency increased throughput.
- Prefix caching reached roughly 80%.
- CPU weight offload greatly reduced throughput.
- N-gram speculative decoding worked with variable acceptance.
- Compiled mode and CUDA graphs started successfully.
- Async scheduling improved the short-run latency/throughput comparison.
- PCIe operated at Gen4 x4.
- Long prompts exceeded the 8192-token context budget.
- Forced-OOM handling failed safely with a clear error.

## Limitations

- Most experiments used one repeat.
- Some full sweeps were skipped.
- E7 thinking comparison was unavailable in the current harness.
- E12 API-server scaling was unavailable in vLLM 0.31.0.
- Results are exploratory rather than production-grade benchmarks.
