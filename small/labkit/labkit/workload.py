"""Live agent workload. The model really generates and picks tool calls; tools are mocked and seeded.

Classes: A agent-code (shared project context), B agent-research (bigger tool outputs),
C quick-tool (short, thinking off), D longdoc (single long-prompt request).
No replay mode: every request comes from a real model decision.
"""
from __future__ import annotations
import asyncio, hashlib, itertools, json, math, os, random, sysconfig, time
from dataclasses import dataclass, field
from . import core as C

TOOL_NAMES = ["search_docs", "read_file", "run_python", "sql_query", "calculator", "final_answer"]
TOOLS_SPEC = [{"type": "function", "function": {
    "name": n, "description": "Mock tool " + n,
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    for n in TOOL_NAMES]

PRESETS = {"mix": "A:0.4,B:0.2,C:0.3,D:0.1", "agent-code": "A:1", "agent-research": "B:1",
           "quick-tool": "C:1", "longdoc": "D:1"}

CLASSES = {  # token sizes tuned so sessions fit max-model-len 4096
    "A": dict(steps=(3, 6), tool=(150, 450), think=True, delay=(0.3, 2.0), ctx=900),
    "B": dict(steps=(3, 6), tool=(300, 700), think=True, delay=(0.5, 3.0), ctx=0),
    "C": dict(steps=(2, 4), tool=(60, 200), think=False, delay=(0.1, 0.5), ctx=0),
    "D": dict(steps=(1, 1), tool=None, think=False, delay=None, ctx=0),
}


@dataclass
class RunCfg:
    workload: str = "mix"
    load: str = "closed"        # closed | open
    conc: int = 4
    rate: float = 0.5           # sessions/s, open loop
    n: int = 60
    duration: float = 0.0       # seconds; overrides n when > 0
    warmup: int = 6
    think: str = "trace"        # trace | on | off
    max_tokens: int = 512
    nudge: str = ""
    projects: int = 4
    scale: float = 1.0
    prio: bool = False
    burst: bool = False
    tool_choice: str = "auto"   # auto | required-first
    seed: int = 1
    max_len: int = 4096
    port: int = C.PORT
    tool_latency: float = 1.0
    mix: str = ""


# ---------------- text bank ----------------
class TextBank:
    """Deterministic token-accurate text slices from the Python stdlib (tokenizer = the served model's)."""

    def __init__(self):
        self.tok, self.cpt = None, 3  # chars per unit in fallback mode
        if os.environ.get("LAB_TOKENIZER", "auto") != "none":
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(C.MODEL)
                self.cpt = 1
            except Exception:
                self.tok = None
        text = self._corpus()
        self.sha = hashlib.sha256(text.encode()).hexdigest()
        if self.tok:
            cache = C.P(".cache", "corpus_ids.json")
            d = C.jload(cache)
            if d and d.get("sha") == self.sha and d.get("model") == C.MODEL:
                self.ids = d["ids"]
            else:
                self.ids = self.tok(text)["input_ids"]
                C.jdump({"sha": self.sha, "model": C.MODEL, "ids": self.ids}, cache)
        else:
            self.ids = list(text)

    @staticmethod
    def _corpus(limit=1_200_000):
        d = sysconfig.get_paths()["stdlib"]
        buf, size = [], 0
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                try:
                    t = open(os.path.join(d, f), errors="ignore").read()
                except Exception:
                    continue
                buf.append(t)
                size += len(t)
                if size > limit:
                    break
        return "\n".join(buf)[:limit]

    def take(self, n, key):
        n = max(8, int(n)) * self.cpt
        start = int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % max(1, len(self.ids) - n - 1)
        sl = self.ids[start:start + n]
        return self.tok.decode(sl) if self.tok else "".join(sl)


def parse_mix(s):
    cl, w = zip(*[(c, float(x)) for c, x in (kv.split(":") for kv in s.split(","))])
    return list(cl), list(w)


def _lognorm(rng, med, p95):
    sig = (math.log(p95) - math.log(med)) / 1.645
    return max(1, int(med * math.exp(rng.gauss(0, sig))))


def make_session(bank: TextBank, cfg: RunCfg, idx: int) -> dict:
    mix = parse_mix(cfg.mix or PRESETS.get(cfg.workload, cfg.workload))
    rng = random.Random(f"{cfg.seed}:{idx}")
    c = rng.choices(mix[0], mix[1])[0]
    cp = CLASSES[c]
    s = dict(id=f"s{idx}", idx=idx, cls=c, gap=rng.expovariate(1.0), think=cp["think"], script=[], answer="")
    s["system"] = "" if c == "D" else "You are a tool-using assistant. Reference material follows.\n" + bank.take(350 * cfg.scale, "sys")
    s["ctx"], s["project"] = "", None
    if c == "A":
        s["project"] = rng.randrange(cfg.projects)
        s["ctx"] = bank.take(cp["ctx"] * cfg.scale, f"proj{s['project']}")
    if c == "D":
        s["ctx"] = bank.take(rng.randint(1500, 2800) * cfg.scale, f"doc{idx}")
        s["task"] = "Summarize the document in 3 sentences."
        s["max_steps"] = 1
        return s
    n = rng.randint(*cp["steps"])
    val = str(int(hashlib.sha256(f"v{cfg.seed}:{idx}".encode()).hexdigest()[:8], 16))
    key, hide = f"KEY_{idx}", rng.randrange(n)
    s["answer"] = val
    s["task"] = f"Use the tools to find {key} in the tool outputs, then reply exactly: Answer: <value>."
    for k in range(n):
        toks = min(_lognorm(rng, *cp["tool"]), 700)
        body = bank.take(toks, f"tool{idx}:{k}")
        if k == hide:
            body = f"{key} = {val} " + body
        d = min(_lognorm(rng, *[int(x * 1000) for x in cp["delay"]]) / 1000, 20) * cfg.tool_latency
        s["script"].append(dict(out=body, tokens=toks, delay=d))
    s["max_steps"] = n + 2
    return s


def prefix_msgs(s, cfg):
    m = []
    if s["system"]:
        m.append({"role": "system", "content": s["system"]})
    if s["ctx"]:
        m.append({"role": "user", "content": "Context: " + s["ctx"]})
    m.append({"role": "user", "content": s["task"] + cfg.nudge})
    return m


def think_of(s, cfg):
    return s["think"] if cfg.think == "trace" else cfg.think == "on"


def max_tokens_of(s, cfg, thinking):
    if s["cls"] == "D":
        return min(cfg.max_tokens, 160)
    return cfg.max_tokens if thinking else min(cfg.max_tokens, 256)


# ---------------- streaming client ----------------
_clients = {}


def client(port):
    from openai import AsyncOpenAI
    if port not in _clients:
        _clients[port] = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=C.KEY, timeout=900, max_retries=0)
    return _clients[port]


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def call(cfg, messages, tools, think, max_tokens, priority=None, tool_choice=None):
    kw = {"enable_thinking": False} if not think else {}
    samp = dict(temperature=0.6, top_p=0.95) if think else dict(temperature=0.7, top_p=0.8)
    extra = {"top_k": 20, "chat_template_kwargs": kw}
    if priority is not None:
        extra["priority"] = priority
    args = dict(model=C.SERVED, messages=messages, stream=True, stream_options={"include_usage": True},
                max_tokens=max_tokens, extra_body=extra, **samp)
    if tools:
        args["tools"] = tools
        if tool_choice:
            args["tool_choice"] = tool_choice
    ts, t0 = time.time(), time.perf_counter()
    first = answer = None
    last, gaps, content, rlen, calls, usage, finish = t0, [], "", 0, {}, None, None
    stream = await client(cfg.port).chat.completions.create(**args)
    async for ch in stream:
        now = time.perf_counter()
        if getattr(ch, "usage", None):
            usage = ch.usage
        if not ch.choices:
            continue
        d = ch.choices[0].delta
        if ch.choices[0].finish_reason:
            finish = ch.choices[0].finish_reason
        rt = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if rt or d.content or d.tool_calls:
            if first is None:
                first = now - t0
            else:
                gaps.append(now - last)
            last = now
        if rt:
            rlen += len(rt)
        if d.content:
            content += d.content
            answer = answer if answer is not None else now - t0
        for tc in d.tool_calls or []:
            answer = answer if answer is not None else now - t0
            s = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": ""})
            if tc.id:
                s["id"] = tc.id
            if tc.function and tc.function.name:
                s["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                s["args"] += tc.function.arguments
    e2e = time.perf_counter() - t0
    n = usage.completion_tokens if usage else None
    ok = True
    for s in calls.values():
        try:
            json.loads(s["args"] or "{}")
        except Exception:
            ok = False
    det = getattr(usage, "prompt_tokens_details", None) if usage else None
    return dict(
        t_submit=ts, ttft=first, t_answer=answer, e2e=e2e, finish=finish,
        prompt_tokens=usage.prompt_tokens if usage else None, completion_tokens=n,
        cached_tokens=getattr(det, "cached_tokens", None) if det else None,
        reasoning_chars=rlen, mean_itl=((e2e - first) / max(n - 1, 1)) if first and n else None,
        gap_p50=_pct(gaps, 50), gap_p99=_pct(gaps, 99), gap_max=max(gaps) if gaps else None,
        n_tool_calls=len(calls), tool_json_ok=ok, max_tokens=max_tokens,
        tool_calls=list(calls.values()), content=content)


# ---------------- agent loop ----------------
async def run_session(cfg, s, sink):
    msgs, t_start, steps, ok, ended = prefix_msgs(s, cfg), time.time(), 0, False, "max_steps"
    think = think_of(s, cfg)
    tools = TOOLS_SPEC if s["cls"] != "D" else None
    mt = max_tokens_of(s, cfg, think)
    last_total = 0
    for k in range(s["max_steps"]):
        tc_mode = "required" if (cfg.tool_choice == "required-first" and tools and k < len(s["script"]) - 1) else None
        prio = (0 if s["cls"] == "C" else 1) if cfg.prio else None
        try:
            r = await call(cfg, msgs, tools, think, mt, prio, tc_mode)
        except Exception as e:
            sink({"session": s["id"], "step": k, "cls": s["cls"], "error": repr(e)[:300], "t_submit": time.time()})
            ended = "error"
            break
        steps += 1
        sink({**{x: r[x] for x in r if x not in ("content", "tool_calls")}, "session": s["id"], "step": k,
              "cls": s["cls"], "think": think})
        if s["cls"] == "D":
            ended = "answer"
            break
        if not r["tool_calls"]:
            ok = bool(s["answer"]) and s["answer"] in (r["content"] or "")
            ended = "answer"
            break
        step = s["script"][k] if k < len(s["script"]) else dict(out="No further results.", tokens=6, delay=0.2)
        last_total = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        if last_total + step["tokens"] + 64 > cfg.max_len:
            ended = "context"
            break
        calls_out = []
        for j, t in enumerate(r["tool_calls"]):
            try:
                json.loads(t["args"] or "{}")
                args = t["args"] or "{}"
            except Exception:
                args = "{}"  # keep the conversation valid; validity is already recorded
            calls_out.append({"id": t["id"] or f"c{k}_{j}", "type": "function",
                              "function": {"name": t["name"] or "search_docs", "arguments": args}})
        msgs.append({"role": "assistant", "content": r["content"] or "", "tool_calls": calls_out})
        await asyncio.sleep(step["delay"])
        for j, c in enumerate(calls_out):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": step["out"] if j == 0 else "ok"})
    sink({"session": s["id"], "cls": s["cls"], "session_done": True, "steps": steps, "scripted": len(s["script"]),
          "dur": time.time() - t_start, "success": ok if s["cls"] != "D" else None, "ended": ended,
          "t_start": t_start})


async def run_workload(cfg: RunCfg, out_path, bank=None):
    bank = bank or TextBank()
    f = open(out_path, "a")

    def sink(r):
        f.write(json.dumps(r) + "\n")
        f.flush()

    nosink = lambda r: None
    counter = itertools.count(cfg.warmup)

    # warmup (discarded): fills caches, compiles lazily-built kernels
    if cfg.warmup:
        wi = iter(range(cfg.warmup))

        async def ww():
            for i in wi:
                await run_session(cfg, make_session(bank, cfg, i), nosink)
        await asyncio.gather(*[ww() for _ in range(min(4, max(1, cfg.conc)))])

    t_run = time.time()
    sink({"run_start": t_run, "cfg": cfg.__dict__, "corpus_sha": bank.sha})
    deadline = t_run + cfg.duration if cfg.duration > 0 else None
    limit = None if deadline else cfg.warmup + cfg.n

    def more(i):
        return (limit is None or i < limit) and (deadline is None or time.time() < deadline)

    if cfg.load == "closed":
        async def worker():
            while True:
                i = next(counter)
                if not more(i):
                    return
                await run_session(cfg, make_session(bank, cfg, i), sink)
        await asyncio.gather(*[worker() for _ in range(cfg.conc)])
    else:
        t_begin, tasks = time.time(), []
        mult = lambda: 4.0 if (cfg.burst and (time.time() - t_begin) % 120 < 20) else 1.0
        while True:
            i = next(counter)
            if not more(i):
                break
            s = make_session(bank, cfg, i)
            await asyncio.sleep(s["gap"] / (cfg.rate * mult()))
            tasks.append(asyncio.create_task(run_session(cfg, s, sink)))
        await asyncio.gather(*tasks)
    sink({"run_end": time.time()})
    f.close()
