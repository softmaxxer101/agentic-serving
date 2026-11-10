import asyncio, json, time, argparse, uvloop
from common import call, TOOLS_SPEC
ap = argparse.ArgumentParser()
ap.add_argument('--trace', required=True); ap.add_argument('--mode', default='replay', choices=['replay', 'live'])
ap.add_argument('--load', default='closed', choices=['open', 'closed']); ap.add_argument('--rate', type=float, default=0.5)
ap.add_argument('--conc', type=int, default=4); ap.add_argument('--n', type=int, default=100); ap.add_argument('--warmup', type=int, default=8)
ap.add_argument('--max-tokens', dest='max_tokens', type=int, default=512)
ap.add_argument('--base-url', dest='base_url', default='http://localhost:8000/v1'); ap.add_argument('--api-key', dest='api_key', default='lab-secret-key')
ap.add_argument('--model', default='Qwen/Qwen3-0.6B'); ap.add_argument('--out', required=True)
a = ap.parse_args()
def prefix(s):
    m = []
    if s['system']: m.append({'role': 'system', 'content': s['system']})
    if s['ctx']: m.append({'role': 'user', 'content': 'Context: ' + s['ctx']})
    m.append({'role': 'user', 'content': s['task']}); return m
async def run_session(s, sink):
    msgs = prefix(s); t_start = time.time(); steps = 0; ok = False
    tools = TOOLS_SPEC if s['cls'] != 'D' else None
    for k in range(s['max_steps']):
        try: r = await call(a, msgs, tools, s['think'])
        except Exception as e:
            sink({'session': s['id'], 'step': k, 'cls': s['cls'], 'error': repr(e), 't_submit': time.time()}); break
        steps += 1
        sink({**{x: r[x] for x in r if x != 'content'}, 'session': s['id'], 'step': k, 'cls': s['cls']})
        if s['cls'] == 'D': break
        st = s['script'][min(k, len(s['script']) - 1)]
        if a.mode == 'replay':
            msgs.append(st['assistant'])
            if not st['assistant'].get('tool_calls'): break
        else:
            if not r['tool_calls']:
                ok = s['answer'] in r['content']; break
            msgs.append({'role': 'assistant', 'content': r['content'], 'tool_calls': [{'id': t['id'] or 'c%d' % k, 'type': 'function', 'function': {'name': t['name'], 'arguments': t['args']}} for t in r['tool_calls']]})
        await asyncio.sleep(st['delay'])
        if a.mode == 'replay': msgs.extend(st['tool_results'])
        else:
            for t in r['tool_calls']: msgs.append({'role': 'tool', 'tool_call_id': t['id'] or 'c%d' % k, 'content': st['tool_results'][0]['content'] if st['tool_results'] else 'ok'})
    sink({'session': s['id'], 'cls': s['cls'], 'session_done': True, 'steps': steps, 'dur': time.time() - t_start, 'success': ok if a.mode == 'live' else None, 't_start': t_start})
async def go(ss, sink, rate=None, conc=4):
    if rate:
        ts = []
        for s in ss:
            await asyncio.sleep(s['gap'] / rate); ts.append(asyncio.create_task(run_session(s, sink)))
        await asyncio.gather(*ts)
    else:
        it = iter(ss)
        async def worker():
            for s in it: await run_session(s, sink)
        await asyncio.gather(*[worker() for _ in range(conc)])
async def main():
    allss = [json.loads(l) for l in open(a.trace)]
    warm, ss = allss[:a.warmup], allss[a.warmup:a.warmup + a.n]
    f = open(a.out, 'a')
    def sink(r): f.write(json.dumps(r) + chr(10)); f.flush()
    if warm: await go(warm, lambda r: None, None, 4)
    sink({'run_start': time.time(), 'args': vars(a)})
    await go(ss, sink, a.rate if a.load == 'open' else None, a.conc)
    sink({'run_end': time.time()})
uvloop.run(main())
