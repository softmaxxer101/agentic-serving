"""Experiment matrices E0..E13. Edit nothing here: your job is predict, run, read, explain."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Cfg:
    name: str
    args: str = ""                       # extra serve flags (override the base preset)
    drop: tuple = ()                     # base flags to remove
    env: dict = field(default_factory=dict)
    wl: dict = field(default_factory=dict)   # RunCfg overrides; rate_frac = fraction of SAT
    model: str | None = None
    probe: bool = False
    micro: tuple = ()                    # copy_fresh | disconnect | toolong
    reps: int | None = None              # None = use --reps
    prefetch: str | None = None          # HF repo to download before start


MIX = dict(workload="mix", load="closed")

CONCEPT = {
    "E0": "Baseline, scaling curve, noise floor. Predict: ITL at c=1 (roofline floor x1.5-3 in eager), concurrency where throughput flattens, KV pool tokens.",
    "E1": "KV memory budget and preemption. Predict: KV tokens per gpu-memory-utilization (use bytes/token), the concurrency where preemptions start.",
    "E2": "Prefix caching and where it breaks. Predict: cached fraction at 4 projects; the project count where hit rate collapses (shared tokens x projects vs KV pool).",
    "E3": "Batching knobs. Predict: smaller max-num-batched-tokens lowers class C ITL p99 and raises class D TTFT.",
    "E4": "CUDA graphs vs eager. Predict: ITL gain at c=1, shrinking with concurrency; KV pool shrinks slightly; startup grows.",
    "E5": "Open-loop knee and goodput. Predict: knee in sessions/s and the offered fraction where p99 doubles.",
    "E6": "Thinking cost. Predict: completion tokens per step on/off ratio; the cap below which most thinking steps truncate.",
    "E7": "n-gram speculation. Predict: gain at c=1, crossover concurrency, acceptance higher on COPY than FRESH.",
    "E8": "Weight formats. Predict: weights GiB, KV tokens, and whether decode is faster or slower than bf16 on sm_86.",
    "E9": "Priority scheduling. Predict: class C p99 TTFT improves under overload, A/B/D get worse.",
    "E10": "CPU offload. Predict: hit rate recovers with KV offload at a TTFT cost set by host-to-device bandwidth H.",
    "E11": "API frontend overhead. Predict: where req/s stops growing while GPU util is under 90 percent.",
    "E12": "Stress and failure. Predict: graceful flat throughput or preemption storm at 2x SAT; what leaks after disconnects.",
    "E13": "Final config waterfall. Predict: ordered winners and the final goodput gain.",
}


def build(exp: str, caps, state: dict, opts: dict) -> list[Cfg]:
    full = opts.get("full", False)
    out: list[Cfg] = []
    a = out.append
    if exp == "E0":
        for c in (1, 2, 4, 8, 16):
            a(Cfg(f"c{c}", wl=dict(MIX, conc=c, n=30 if c <= 2 else 60)))
    elif exp == "E1":
        for g in ("0.70", "0.75", "0.80", "0.85", "0.90"):
            a(Cfg(f"gmu{g}", f"--gpu-memory-utilization {g}", wl=dict(MIX, conc=8, n=40)))
        kvd = caps.choices("--kv-cache-dtype")
        a(Cfg("kv-fp8", "--kv-cache-dtype fp8", probe=True, wl=dict(MIX, conc=8, n=40)))
        if not kvd or any("nvfp4" in x for x in kvd):  # 0.32.0: NVFP4 KV reaches SM8x
            a(Cfg("kv-nvfp4", "--kv-cache-dtype nvfp4", probe=True, wl=dict(MIX, conc=8, n=40)))
        a(Cfg("bf16-probe", probe=True, wl=dict(MIX, conc=8, n=40)))
        for L in (2048, 3072):
            a(Cfg(f"len{L}", f"--max-model-len {L}", wl=dict(MIX, conc=8, n=40, max_len=L)))
        for c in (8, 16, 24, 32):
            a(Cfg(f"press-c{c}", "--max-num-seqs 32", wl=dict(MIX, conc=c, n=60, projects=16)))
    elif exp == "E2":
        for p in (2, 4, 8, 16):
            w = dict(workload="agent-code", load="closed", conc=8, n=60, projects=p)
            a(Cfg(f"on-P{p}", wl=w))
            a(Cfg(f"off-P{p}", "--no-enable-prefix-caching", wl=w))
    elif exp == "E3":
        for t in (512, 1024, 2048, 4096):
            a(Cfg(f"bt{t}", f"--max-num-batched-tokens {t}", wl=dict(MIX, conc=16, n=60)))
        for s in (4, 8, 16, 32):
            a(Cfg(f"seq{s}", f"--max-num-seqs {s}", wl=dict(MIX, conc=32, n=60)))
        a(Cfg("nochunk", "--no-enable-chunked-prefill", wl=dict(MIX, conc=16, n=60)))
    elif exp == "E4":
        for c in (1, 4, 16):
            n = 30 if c == 1 else 60
            a(Cfg(f"eager-c{c}", wl=dict(MIX, conc=c, n=n)))
            a(Cfg(f"graphs-c{c}", drop=("--enforce-eager",), wl=dict(MIX, conc=c, n=n)))
        a(Cfg("O0-c1", "-O0", drop=("--enforce-eager",), wl=dict(MIX, conc=1, n=30)))
        a(Cfg("O3-c1", "-O3", drop=("--enforce-eager",), wl=dict(MIX, conc=1, n=30)))
        a(Cfg("piecewise-c1", '--compilation-config {"cudagraph_mode":"PIECEWISE"}', drop=("--enforce-eager",), wl=dict(MIX, conc=1, n=30)))
    elif exp == "E5":
        a(Cfg("sat-c16", wl=dict(MIX, conc=16, n=60)))
        for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            a(Cfg(f"load-{f}", wl=dict(workload="mix", load="open", rate_frac=f, n=100)))
    elif exp == "E6":
        w = dict(MIX, conc=4, n=50)
        a(Cfg("think-off", wl=dict(w, think="off")))
        a(Cfg("think-on", wl=dict(w, think="on")))
        for t in (128, 256, 512, 1024):
            a(Cfg(f"cap{t}", wl=dict(w, think="on", max_tokens=t)))
        a(Cfg("nudge1024", wl=dict(w, think="on", max_tokens=1024, nudge=" Think very briefly (at most 3 sentences) before answering.")))
        for m in ("off", "on"):
            a(Cfg(f"live-C-{m}", wl=dict(workload="quick-tool", load="closed", conc=2, n=40, think=m, max_tokens=1024)))
    elif exp == "E7":
        for c in (1, 4, 16):
            a(Cfg(f"off-c{c}", wl=dict(MIX, conc=c, n=40), probe=(c == 1), micro=("copy_fresh",) if c == 1 else ()))
        for n in ((1, 2, 3, 4, 5) if full else (1, 3, 5)):
            j = '{"method":"ngram","num_speculative_tokens":%d,"prompt_lookup_max":4,"prompt_lookup_min":2}' % n
            for c in (1, 4, 16):
                a(Cfg(f"ng{n}-c{c}", f"--speculative-config {j}", wl=dict(MIX, conc=c, n=40),
                      probe=(c == 1 and n == 3), micro=("copy_fresh",) if c == 1 else ()))
        # 0.32.0 notes a GPU n-gram speculator; method name is unverified, so a rejection is a finding.
        a(Cfg("ngram-gpu-c4", '--speculative-config {"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}',
              wl=dict(MIX, conc=4, n=40)))
    elif exp == "E8":
        variants = [("bf16", "", None, None), ("fp8ckpt", "", "Qwen/Qwen3-0.6B-FP8", "Qwen/Qwen3-0.6B-FP8"),
                    ("fp8dyn", "--quantization fp8", None, None), ("bnb4", "--quantization bitsandbytes", None, None)]
        for name, args, model, pre in variants:
            for c, n in ((1, 30), (8, 60)):
                a(Cfg(f"{name}-c{c}", args, model=model, prefetch=pre, probe=(c == 1), wl=dict(MIX, conc=c, n=n)))
        a(Cfg("live-C-bf16", wl=dict(workload="quick-tool", load="closed", conc=2, n=40, max_tokens=1024)))
    elif exp in ("E9", "E5X"):
        need_sat(state)
        w = dict(workload="mix", load="open", rate_frac=1.25, n=120)
        a(Cfg("fcfs", wl=w))
        a(Cfg("prio", "--scheduling-policy priority", wl=dict(w, prio=True)))
        a(Cfg("async", "--async-scheduling", wl=w))
    elif exp == "E10":
        w = dict(workload="agent-code", load="closed", conc=8, n=60, projects=32)
        a(Cfg("off", wl=w))
        if caps.has("--kv-offloading-size"):
            kv = "--kv-offloading-size 2" + (" --kv-offloading-backend native" if caps.has("--kv-offloading-backend") else "")
            a(Cfg("kv-offload", kv, wl=w))
        else:
            a(Cfg("kv-offload-MISSING", "--cpu-offload-gb 1", wl=w))  # finding: no KV offload flag in this build
        a(Cfg("weight-offload", "--cpu-offload-gb 1", wl=w))
    elif exp == "E11":
        for k in (1, 2, 4):
            a(Cfg(f"api{k}", f"--api-server-count {k} --max-num-seqs 64", wl=dict(workload="quick-tool", load="closed", conc=32, n=200)))
    elif exp == "E12":
        need_sat(state)
        a(Cfg("over", wl=dict(workload="mix", load="open", rate_frac=2.0, n=150), reps=1))
        a(Cfg("burst", wl=dict(workload="mix", load="open", rate_frac=0.5, n=150, burst=True), reps=1))
        a(Cfg("toolong", micro=("toolong",), wl=dict(workload="quick-tool", load="closed", conc=1, n=4, warmup=1), reps=1))
        a(Cfg("disconnect", micro=("disconnect",), wl=dict(workload="quick-tool", load="closed", conc=1, n=4, warmup=1), reps=1))
        a(Cfg("soak", wl=dict(MIX, conc=8, duration=opts.get("soak_min", 30) * 60), reps=1))
        a(Cfg("oom", "--gpu-memory-utilization 0.98 --max-model-len 32768", wl=dict(MIX, conc=2, n=10, max_len=32768), reps=1))
    elif exp == "E13":
        need_sat(state)
        ws = opts.get("winners") or []
        if not ws:
            raise SystemExit("E13 needs your winning flag sets: lab exp E13 --winner '<flags>' --winner '<flags>' ...")
        w = dict(workload="mix", load="open", rate_frac=1.0, n=120)
        a(Cfg("base", wl=w))
        acc = ""
        for i, x in enumerate(ws):
            acc = (acc + " " + x).strip()
            a(Cfg(f"add{i + 1}", acc, wl=w))
        for i in range(len(ws)):
            rest = " ".join(x for j, x in enumerate(ws) if j != i)
            a(Cfg(f"minus{i + 1}", rest, wl=w))
        a(Cfg("final", acc, probe=True, wl=w))
        a(Cfg("final-soak", acc, wl=dict(MIX, conc=8, duration=opts.get("soak_min", 30) * 60), reps=1))
    else:
        raise SystemExit(f"unknown experiment {exp}; choose from {sorted(CONCEPT)}")
    # drop graph-only duplicates if the user's flags no longer apply; flags unknown to this build are filtered at runtime
    return out


def need_sat(state):
    if not state.get("SAT"):
        state["_need_sat"] = True  # runner executes E5's saturation point first


EXPS = sorted(CONCEPT, key=lambda x: int(x[1:]))
