import json, sys, numpy as np
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print('exact match', np.mean([x['text'] == y['text'] for x, y in zip(a, b)]))
print('first 40 chars match', np.mean([x['text'][:40] == y['text'][:40] for x, y in zip(a, b)]))
print('mean logprob base', np.mean([np.mean(x['lp']) for x in a]), 'other', np.mean([np.mean(y['lp']) for y in b]))
