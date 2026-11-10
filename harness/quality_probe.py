import json, sys, os
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key=os.environ['KEY'])
ps = [json.loads(l) for l in open('traces/M.jsonl')][:60]
out = []
for s in ps:
    msgs = [{'role': 'user', 'content': (s['ctx'][:1500] + ' ' + s['task'])}]
    r = c.chat.completions.create(model='Qwen/Qwen3-0.6B', messages=msgs, temperature=0, max_tokens=64, logprobs=True, top_logprobs=1, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
    out.append({'text': r.choices[0].message.content, 'lp': [t.logprob for t in r.choices[0].logprobs.content]})
json.dump(out, open(sys.argv[1], 'w'))
