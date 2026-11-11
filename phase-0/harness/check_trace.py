import json, sys, copy
from transformers import AutoTokenizer
from common import TOOLS_SPEC
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); LIM = int(sys.argv[2]); B = 16
seen = set(); tot = hit = over = mx = nreq = 0
for l in open(sys.argv[1]):
    s = json.loads(l); msgs = []
    if s['system']: msgs.append({'role': 'system', 'content': s['system']})
    if s['ctx']: msgs.append({'role': 'user', 'content': 'Context: ' + s['ctx']})
    msgs.append({'role': 'user', 'content': s['task']})
    for k in range(max(1, len(s['script']))):
        text = tok.apply_chat_template(msgs, tools=TOOLS_SPEC, add_generation_prompt=True, tokenize=False)
        ids = tok(text)['input_ids']; n = len(ids); mx = max(mx, n); over += n + 512 > LIM; nreq += 1
        h = 0; run = True; nb = n // B
        for b in range(nb):
            h = hash((h, tuple(ids[b * B:(b + 1) * B])))
            if run and h in seen: hit += 1
            else: run = False
            seen.add(h)
        tot += nb
        if k < len(s['script']):
            st = copy.deepcopy(s['script'][k])
            for t in st['assistant'].get('tool_calls', []): t['function']['arguments'] = json.loads(t['function']['arguments'])
            msgs.append(st['assistant']); msgs.extend(st['tool_results'])
print('requests', nreq, 'max prompt tokens', mx, 'over limit (prompt+512 > LIM)', over, 'ideal hit rate %.3f' % (hit / tot))
