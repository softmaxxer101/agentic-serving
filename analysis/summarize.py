import json, sys, pandas as pd, numpy as np
rows = []
for path in sys.argv[1:]:
    recs = [json.loads(l) for l in open(path)]
    req = pd.DataFrame([r for r in recs if 'ttft' in r and 'session_done' not in r])
    ses = pd.DataFrame([r for r in recs if r.get('session_done')])
    err = sum(1 for r in recs if 'error' in r)
    st = [r for r in recs if 'run_start' in r][0]['run_start']; en = [r for r in recs if 'run_end' in r][0]['run_end']; dur = en - st
    q = lambda s, p: float(np.nanpercentile(s.dropna(), p)) if len(s.dropna()) else None
    rows.append(dict(run=path.split('/')[-2], sessions=len(ses), requests=len(req), errors=err, dur_s=round(dur, 1),
        ttft_p50=q(req.ttft, 50), ttft_p99=q(req.ttft, 99), itl_p50=q(req.mean_itl, 50), itl_p95=q(req.mean_itl, 95),
        e2e_p50=q(req.e2e, 50), e2e_p99=q(req.e2e, 99), sess_p50=q(ses.dur, 50), sess_p99=q(ses.dur, 99),
        out_tok_s=req.completion_tokens.sum() / dur, in_tok_s=req.prompt_tokens.sum() / dur,
        json_ok=req.tool_json_ok.mean(), cached_frac=(req.cached_tokens.sum() / req.prompt_tokens.sum()) if 'cached_tokens' in req and req.cached_tokens.notna().any() else None,
        truncated=(req.finish == 'length').mean()))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
print(pd.DataFrame(rows).round(3).to_string(index=False))
