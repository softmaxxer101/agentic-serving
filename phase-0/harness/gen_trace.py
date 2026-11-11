import json, math, random, hashlib, argparse
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--projects', type=int, default=4); ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--mix', default='A:0.4,B:0.2,C:0.3,D:0.1'); ap.add_argument('--out', default='traces/M.jsonl')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
ids = tok(open('traces/corpus.txt').read()[:3000000])['input_ids']
rng = random.Random(a.seed)
def text(n, seed):
    r = random.Random(seed); s = r.randrange(0, len(ids) - n - 1); return tok.decode(ids[s:s + n])
def sha(x): return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16)
def lognorm(med, p95):
    sig = (math.log(p95) - math.log(med)) / 1.645
    return max(1, int(med * math.exp(rng.gauss(0, sig))))
P = {'A': dict(steps=(6, 15), tool=(300, 1200), think=True, delay=(0.3, 2.0)),
     'B': dict(steps=(5, 12), tool=(600, 2000), think=True, delay=(0.5, 3.0)),
     'C': dict(steps=(3, 6), tool=(100, 300), think=False, delay=(0.1, 0.5)),
     'D': dict(steps=(1, 1), tool=None, think=False, delay=None)}
TOOLS = ['search_docs', 'read_file', 'run_python', 'sql_query', 'calculator']
SYS = text(int(600 * a.scale), 999)
PROJ = {p: text(int(rng.randint(2000, 3000) * a.scale), 5000 + p) for p in range(a.projects)}
cl, w = zip(*[(c, float(x)) for c, x in (kv.split(':') for kv in a.mix.split(','))])
out = open(a.out, 'w')
for i in range(a.n):
    c = rng.choices(cl, w)[0]; p = P[c]
    s = dict(id='s%d' % i, cls=c, gap=rng.expovariate(1.0), think=p['think'], script=[], answer='', system='' if c == 'D' else SYS, ctx='')
    if c == 'A': s['project'] = rng.randrange(a.projects); s['ctx'] = PROJ[s['project']]
    if c == 'D':
        s['ctx'] = text(int(rng.randint(3000, 6000) * a.scale), 7000 + i); s['task'] = 'Summarize the document in 3 sentences.'; s['max_steps'] = 1
    else:
        nsteps = max(1, min(p['steps'][1], round(rng.gauss(p['steps'][0], p['steps'][0] / 3))))
        val = str(sha('v%d' % i)); key = 'KEY_%d' % i; hide = rng.randrange(nsteps) if nsteps > 0 else 0
        s['answer'] = val; s['task'] = 'Use the tools to find %s in the tool outputs, then reply exactly: Answer: <value>.' % key
        for k in range(nsteps):
            name = rng.choice(TOOLS); args = json.dumps({'query': '%s-%d' % (s['id'], k)})
            body = text(lognorm(*p['tool']), sha(name + args))
            if k == hide: body = '%s = %s ' % (key, val) + body
            tid = 'call_%s_%d' % (s['id'], k)
            s['script'].append(dict(assistant={'role': 'assistant', 'content': '', 'tool_calls': [{'id': tid, 'type': 'function', 'function': {'name': name, 'arguments': args}}]}, tool_results=[{'role': 'tool', 'tool_call_id': tid, 'content': body}], delay=min(lognorm(*[int(x * 1000) for x in p['delay']]) / 1000, 20)))
        s['script'].append(dict(assistant={'role': 'assistant', 'content': 'Answer: ' + val}, tool_results=[], delay=0))
        s['max_steps'] = len(s['script'])
    out.write(json.dumps(s) + chr(10))
