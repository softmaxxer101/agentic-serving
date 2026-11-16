#!/usr/bin/env python3
import argparse, csv, re, statistics
from pathlib import Path

def parse_prom(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m: out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out

def delta_sum(b, a, n): return sum(a.get(n, [0.0])) - sum(b.get(n, [0.0]))
def gauge_max(a, n):
    v = a.get(n, []); return max(v) if v else None

def parse_trie_line(text):
    """Grab the 'benchmark complete' logfmt line and split key=value pairs."""
    for line in text.splitlines():
        if "benchmark complete" in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            def f(k):
                try: return float(kv[k])
                except (KeyError, ValueError): return None
            return {
                "ttft_p50": f("ttft_s_p50"),
                "ttft_p99": f("ttft_s_p99"),
                "ttfat_p50": f("ttfat_s_p50"),
                "ttfat_p99": f("ttfat_s_p99"),
                "decode_tpot_p50": f("mean_turn_decode_tpot_ms_p50"),
                "itl_p50": f("inter_token_latency_ms_p50"),
                "itl_p99": f("inter_token_latency_ms_p99"),
                "cache_hit_rate": f("cache_hit_rate_pct_mean"),
                "eligible_cache_hit_rate": f("eligible_cache_hit_rate_pct_mean"),
                "completion_tok_s": f("completion_tok_per_s"),
                "prompt_tok_s": f("prompt_tok_per_s"),
                "completed": f("completed_requests"),
                "failed": f("failed_requests"),
            }
    return {}

def parse_dmon(path):
    pwr, sm, tmp = [], [], []
    if not Path(path).exists(): return pwr, sm, tmp
    for line in open(path):
        p = line.split()
        if len(p) < 8 or p[0] == "#": continue
        try: pwr.append(float(p[1])); sm.append(float(p[2]))
        except ValueError: pass
        try: tmp.append(float(p[3]))
        except ValueError: pass
    return pwr, sm, tmp

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
d = Path(a.run_dir)
trie = parse_trie_line((d / "trie.log").read_text(errors="ignore"))
before = parse_prom((d / "metrics-before.txt").read_text(errors="ignore"))
after  = parse_prom((d / "metrics-after.txt").read_text(errors="ignore"))
pwr, sm, tmp = parse_dmon(d / "dmon.csv")
row = {
    "run": d.name,
    **trie,
    "kv_usage_max": gauge_max(after, "vllm:kv_cache_usage_perc"),
    "preemptions_delta": delta_sum(before, after, "vllm:num_preemptions_total"),
    "prefix_queries_delta": delta_sum(before, after, "vllm:prefix_cache_queries"),
    "prefix_hits_delta": delta_sum(before, after, "vllm:prefix_cache_hits"),
    "power_mean_w": round(statistics.mean(pwr), 2) if pwr else None,
    "sm_clock_min_mhz": min(sm) if sm else None,
    "sm_clock_max_mhz": max(sm) if sm else None,
    "temp_max_c": max(tmp) if tmp else None,
}
w = csv.DictWriter(open(d / "summary-row.csv", "w"), fieldnames=list(row))
w.writeheader(); w.writerow(row)
print(",".join(str(row[k]) for k in row))
