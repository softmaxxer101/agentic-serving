"""Small controlled tests that run against a live server."""
from __future__ import annotations
import asyncio, json, random, re, time
from . import core as C
from .server import fetch_metrics, healthy
from .workload import RunCfg, TextBank, make_session, TOOLS_SPEC


def _client(port):
    from openai import OpenAI
    return OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=C.KEY, timeout=300, max_retries=0)


def _reasoning(msg):
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)


def smoke(port=C.PORT):
    """Five checks that tell you the server, parsers and cache work before you trust any latency."""
    c, res = _client(port), {}
    r = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": "Say hi in 5 words."}], max_tokens=400)
    m = r.choices[0].message
    res["1_plain_text"] = bool((m.content or "").strip())
    res["2_reasoning_separated"] = bool(_reasoning(m)) and "<think>" not in (m.content or "")
    n_think = r.usage.completion_tokens
    r2 = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": "Say hi in 5 words."}], max_tokens=100,
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    res["3_thinking_off_cheaper"] = r2.usage.completion_tokens < n_think and not _reasoning(r2.choices[0].message)
    ok_calls = 0
    for _ in range(5):
        r3 = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": "What is 123*45? Use the calculator tool."}],
                                       tools=TOOLS_SPEC, max_tokens=700)
        ok_calls += bool(r3.choices[0].message.tool_calls)
    res["4_tool_call_rate_of_5"] = ok_calls
    msgs = [{"role": "user", "content": "Repeat this sentence once: " + "the quick brown fox jumps over the lazy dog. " * 40}]
    c.chat.completions.create(model=C.SERVED, messages=msgs, max_tokens=8, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    r5 = c.chat.completions.create(model=C.SERVED, messages=msgs, max_tokens=8, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    det = getattr(r5.usage, "prompt_tokens_details", None)
    res["5_cached_tokens_on_repeat"] = getattr(det, "cached_tokens", None)
    try:
        txt = fetch_metrics(port)
        names = sorted({l.split("{")[0].split(" ")[0] for l in txt.splitlines() if l.startswith("vllm:")})
        (C.P("env") / "metric_names.txt").parent.mkdir(exist_ok=True)
        (C.P("env") / "metric_names.txt").write_text("\n".join(names))
        res["metric_names_saved"] = len(names)
    except Exception as e:
        res["metric_names_saved"] = f"failed: {e!r}"
    return res


def copy_fresh(out, port=C.PORT):
    """COPY text speeds up under n-gram speculation, FRESH text barely does. 8 trials each, greedy."""
    cfg = RunCfg(workload="agent-code")
    txt = make_session(TextBank(), cfg, 0)["ctx"][:1200]
    c, res = _client(port), {}
    for label, p in (("COPY", "Repeat the following text exactly, nothing else: " + txt),
                     ("FRESH", "Write a short story about a lighthouse keeper.")):
        tps = []
        for _ in range(8):
            t = time.perf_counter()
            r = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": p}], temperature=0, max_tokens=300,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            tps.append(r.usage.completion_tokens / (time.perf_counter() - t))
        res[label] = sum(tps) / len(tps)
    C.jdump(res, out / "copy_fresh.json")
    return res


def _cache_and_running(port):
    try:
        txt = fetch_metrics(port)
    except Exception:
        return None, None
    run = kv = None
    for l in txt.splitlines():
        if l.startswith("vllm:num_requests_running"):
            run = (run or 0) + float(l.rsplit(" ", 1)[1])
        if re.match(r"vllm:(kv|gpu)_cache_usage_perc", l):
            kv = (kv or 0) + float(l.rsplit(" ", 1)[1])
    return run, kv


def disconnect_storm(out, port=C.PORT):
    """64 streaming clients hang up early. Running requests and KV usage must return toward zero."""
    import httpx

    async def one():
        body = {"model": C.SERVED, "messages": [{"role": "user", "content": "Write a long essay about the sea."}],
                "max_tokens": 1500, "stream": True}
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                                headers={"Authorization": "Bearer " + C.KEY}, json=body) as r:
                n, lim = 0, random.randint(5, 50)
                async for _ in r.aiter_lines():
                    n += 1
                    if n > lim:
                        break

    async def main():
        await asyncio.gather(*[one() for _ in range(64)], return_exceptions=True)
    t0 = time.time()
    asyncio.run(main())
    res = {"storm_s": time.time() - t0, "timeline": []}
    for i in range(30):
        run, kv = _cache_and_running(port)
        res["timeline"].append((i, run, kv))
        if run == 0:
            res["time_to_idle_s"] = i
            break
        time.sleep(1)
    res["leak_suspected"] = "time_to_idle_s" not in res
    C.jdump(res, out / "disconnect.json")
    return res


def too_long(out, max_len, port=C.PORT):
    """Prompt beyond max-model-len: record the exact error and that the server is still alive after it."""
    c, res = _client(port), {}
    prompt = "word " * (max_len + 500)
    try:
        r = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": prompt}], max_tokens=16)
        res["outcome"] = "accepted"
        res["finish"] = r.choices[0].finish_reason
    except Exception as e:
        res["outcome"], res["error"] = "rejected", repr(e)[:400]
    res["server_alive_after"] = healthy(port)
    C.jdump(res, out / "toolong.json")
    return res


def quality_probe(out, port=C.PORT, n=40):
    """Greedy outputs + logprobs for the same prompts; compare across settings with compare_probes()."""
    c, bank, cfg = _client(port), TextBank(), RunCfg(workload="mix")
    rows, i = [], 0
    while len(rows) < n and i < n * 4:
        s = make_session(bank, cfg, 1000 + i)
        i += 1
        if s["cls"] == "D":
            continue
        msgs = [{"role": "user", "content": (s["ctx"][:1500] + " " + s["task"]).strip()}]
        r = c.chat.completions.create(model=C.SERVED, messages=msgs, temperature=0, max_tokens=64, logprobs=True, top_logprobs=1,
                                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        lp = [t.logprob for t in (r.choices[0].logprobs.content or [])]
        rows.append({"text": r.choices[0].message.content or "", "lp": lp})
    C.jdump(rows, out / "probe.json")
    return rows


def compare_probes(a_path, b_path):
    import numpy as np
    a, b = C.jload(a_path, []), C.jload(b_path, [])
    k = min(len(a), len(b))
    if not k:
        return {}
    return {"exact_match": float(np.mean([a[i]["text"] == b[i]["text"] for i in range(k)])),
            "first40_match": float(np.mean([a[i]["text"][:40] == b[i]["text"][:40] for i in range(k)])),
            "mean_logprob_base": float(np.mean([np.mean(x["lp"]) for x in a[:k] if x["lp"]])),
            "mean_logprob_other": float(np.mean([np.mean(x["lp"]) for x in b[:k] if x["lp"]]))}
