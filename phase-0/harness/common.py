import time, json
from openai import AsyncOpenAI
NAMES = ['search_docs', 'read_file', 'run_python', 'sql_query', 'calculator', 'final_answer']
TOOLS_SPEC = [{'type': 'function', 'function': {'name': n, 'description': 'Mock tool ' + n, 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}} for n in NAMES]
_client = None
def client(a):
    global _client
    if _client is None: _client = AsyncOpenAI(base_url=a.base_url, api_key=a.api_key, timeout=3600)
    return _client
async def call(a, messages, tools, think):
    c = client(a)
    kw = {} if think else {'enable_thinking': False}
    samp = dict(temperature=0.6, top_p=0.95) if think else dict(temperature=0.7, top_p=0.8)
    args = dict(model=a.model, messages=messages, stream=True, stream_options={'include_usage': True}, max_tokens=a.max_tokens, extra_body={'top_k': 20, 'chat_template_kwargs': kw}, **samp)
    if tools: args['tools'] = tools
    ts = time.time(); t0 = time.perf_counter(); first = answer = None; last = t0
    gaps = []; content = ''; reasoning = ''; calls = {}; usage = None; finish = None
    stream = await c.chat.completions.create(**args)
    async for ch in stream:
        now = time.perf_counter()
        if ch.usage: usage = ch.usage
        if not ch.choices: continue
        d = ch.choices[0].delta
        if ch.choices[0].finish_reason: finish = ch.choices[0].finish_reason
        think_t = getattr(d, 'reasoning_content', None) or getattr(d, 'reasoning', None)
        if think_t or d.content or d.tool_calls:
            if first is None: first = now - t0
            else: gaps.append(now - last)
            last = now
            if think_t: reasoning += think_t
            if d.content:
                content += d.content
                if answer is None: answer = now - t0
            for tc in d.tool_calls or []:
                if answer is None: answer = now - t0
                s = calls.setdefault(tc.index, {'id': tc.id, 'name': '', 'args': ''})
                if tc.id: s['id'] = tc.id
                if tc.function and tc.function.name: s['name'] += tc.function.name
                if tc.function and tc.function.arguments: s['args'] += tc.function.arguments
    e2e = time.perf_counter() - t0
    n = usage.completion_tokens if usage else None
    ok = True
    for s in calls.values():
        try: json.loads(s['args'] or '{}')
        except Exception: ok = False
    det = getattr(usage, 'prompt_tokens_details', None) if usage else None
    cached = getattr(det, 'cached_tokens', None) if det else None
    return dict(t_submit=ts, ttft=first, t_answer=answer, e2e=e2e, finish=finish, prompt_tokens=usage.prompt_tokens if usage else None, completion_tokens=n, cached_tokens=cached, reasoning_chars=len(reasoning), mean_itl=((e2e - first) / max(n - 1, 1)) if first and n else None, chunk_gaps=gaps, tool_calls=list(calls.values()), tool_json_ok=ok, content=content)
