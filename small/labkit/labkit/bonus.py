"""Bonus exercises B1-B3. `profile` and `engines` need the GPU and must run with the vLLM server STOPPED."""
from __future__ import annotations
import json, time
from . import core as C


def profile():
    import torch
    from torch.profiler import profile as prof, ProfilerActivity
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    m = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16).cuda()
    x = tok("Hello " * 500, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = m(**x, use_cache=True)
        nxt = o.logits[:, -1:].argmax(-1)
        t = time.perf_counter()
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            m(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    gpu_us = sum(e.device_time_total for e in p.key_averages() if hasattr(e, "device_time_total"))
    print(f"wall {wall * 1000:.1f} ms vs GPU kernel time ~{gpu_us / 1000:.1f} ms -> if GPU << wall, decode is launch-bound (why graphs help)")


def structured(port=C.PORT, n=50):
    from openai import OpenAI
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=C.KEY)
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}
    for guided in (False, True):
        ok, t = 0, time.perf_counter()
        for _ in range(n):
            kw = dict(response_format={"type": "json_schema", "json_schema": {"name": "q", "schema": schema}}) if guided else {}
            r = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": "Return JSON with a key query containing a search string about GPUs."}],
                                          max_tokens=100, extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kw)
            try:
                json.loads(r.choices[0].message.content)
                ok += 1
            except Exception:
                pass
        print("guided" if guided else "free  ", f"validity {ok / n:.2f}  seconds/request {(time.perf_counter() - t) / n:.3f}")


def engines(port=C.PORT, n=8):
    """Transformers generate vs vLLM at c=1, same prompts, greedy. Stop the vLLM server first for the HF half."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    prompts = [f"Write {k} sentences about GPUs." for k in range(2, 2 + n)]
    m = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16).cuda()
    toks, t = 0, time.perf_counter()
    for p in prompts:
        x = tok(p, return_tensors="pt").to("cuda")
        o = m.generate(**x, max_new_tokens=128, do_sample=False)
        toks += o.shape[1] - x["input_ids"].shape[1]
    hf = toks / (time.perf_counter() - t)
    print(f"transformers: {hf:.1f} tok/s at c=1")
    del m
    torch.cuda.empty_cache()
    print("now start the server (lab up base) and run: lab bonus engines-vllm")


def engines_vllm(port=C.PORT, n=8):
    from openai import OpenAI
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=C.KEY)
    toks, t = 0, time.perf_counter()
    for k in range(2, 2 + n):
        r = c.chat.completions.create(model=C.SERVED, messages=[{"role": "user", "content": f"Write {k} sentences about GPUs."}],
                                      max_tokens=128, temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        toks += r.usage.completion_tokens
    print(f"vLLM: {toks / (time.perf_counter() - t):.1f} tok/s at c=1")
