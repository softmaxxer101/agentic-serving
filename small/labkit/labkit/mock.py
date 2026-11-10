"""Tiny OpenAI-compatible stand-in used ONLY by `lab selftest` to verify the kit itself (no GPU needed)."""
import asyncio, json, os, re, threading, time
from . import core as C

_STATE = {"running": 0, "queries": 0, "hits": 0, "prompt": 0, "gen": 0}


def make_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse
    app = FastAPI()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.get("/metrics")
    async def metrics():
        s = _STATE
        t = (f"vllm:num_requests_running {s['running']}\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc {min(1, s['running'] / 20)}\n"
             f"vllm:prefix_cache_queries_total {s['queries']}\nvllm:prefix_cache_hits_total {s['hits']}\n"
             f"vllm:prompt_tokens_total {s['prompt']}\nvllm:generation_tokens_total {s['gen']}\nvllm:num_preemptions_total 0\n")
        return PlainTextResponse(t)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        msgs = body["messages"]
        n_tool = sum(m["role"] == "tool" for m in msgs)
        text = " ".join(str(m.get("content") or "") for m in msgs)
        ptoks = max(1, len(text) // 3)
        cached = ptoks // 2 if n_tool else 0
        want_tool = bool(body.get("tools")) and n_tool < 2

        async def gen():
            _STATE["running"] += 1
            _STATE["queries"] += ptoks
            _STATE["hits"] += cached
            _STATE["prompt"] += ptoks
            await asyncio.sleep(0.02 + ptoks / 200000)
            ck = lambda d, fr=None: "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": "m",
                                                          "choices": [{"index": 0, "delta": d, "finish_reason": fr}]}) + "\n\n"
            ntok = 0
            if body.get("chat_template_kwargs") or True:
                yield ck({"reasoning": "thinking "})
                ntok += 1
            if want_tool:
                yield ck({"tool_calls": [{"index": 0, "id": f"call{n_tool}", "type": "function",
                                          "function": {"name": "search_docs", "arguments": '{"query":"x"}'}}]})
                ntok += 8
                fr = "tool_calls"
            else:
                m = re.search(r"KEY_\d+ = (\d+)", text)
                for tok in ("Answer: " + (m.group(1) if m else "0")).split(" "):
                    await asyncio.sleep(0.005)
                    yield ck({"content": tok + " "})
                    ntok += 1
                fr = "stop"
            yield ck({}, fr)
            _STATE["gen"] += ntok
            _STATE["running"] -= 1
            u = {"prompt_tokens": ptoks, "completion_tokens": ntok, "total_tokens": ptoks + ntok,
                 "prompt_tokens_details": {"cached_tokens": cached}}
            yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": "m", "choices": [], "usage": u}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return app


class MockServer:
    pid = None

    def __init__(self, out, port=None):
        self.port = port or C.PORT

    def start(self, timeout=30):
        import uvicorn
        cfg = uvicorn.Config(make_app(), host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(cfg)
        self.t = threading.Thread(target=self.server.run, daemon=True)
        self.t.start()
        t0 = time.time()
        from .server import healthy
        while time.time() - t0 < timeout:
            if healthy(self.port):
                return True, time.time() - t0
            time.sleep(0.2)
        return False, time.time() - t0

    def stop(self):
        self.server.should_exit = True
        self.t.join(timeout=5)
