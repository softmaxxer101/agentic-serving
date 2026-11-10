"""Turn raw run folders into tables. Every number printed comes from files under runs/."""
from __future__ import annotations
import json, pathlib, re
import numpy as np
import pandas as pd
from . import core as C
from .server import parse_facts

SLO = dict(ttft=2.0, itl=0.06)  # fixed BEFORE looking at results; change only via env LAB_SLO_TTFT / LAB_SLO_ITL
KEYM = ["out_tok_s", "itl_p50", "ttft_p99", "e2e_p99"]


def norm(n: str) -> str:
    n = n.replace("vllm:", "")
    return re.sub(r"_total$", "", n)


def pct(s, p):
    s = pd.Series(s).dropna()
    return float(np.percentile(s, p)) if len(s) else None


def run_dirs(exp):
    """Run folders in the order they were executed (= experiment definition order; first config = baseline)."""
    ds = [p for p in C.P("runs", exp).glob("*/rep*") if p.is_dir()]
    return sorted(ds, key=lambda p: (C.jload(p / "meta.json", {}).get("t_start", 0), p.name))


def load_recs(d):
    f = pathlib.Path(d) / "requests.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def frames(d):
    recs = load_recs(d)
    req = pd.DataFrame([r for r in recs if "ttft" in r and "session_done" not in r])
    ses = pd.DataFrame([r for r in recs if r.get("session_done")])
    err = [r for r in recs if "error" in r]
    st = next((r["run_start"] for r in recs if "run_start" in r), None)
    en = next((r["run_end"] for r in recs if "run_end" in r), None)
    return req, ses, err, st, en


def metric_series(m: pd.DataFrame, base: str):
    if m is None or m.empty:
        return None
    x = m[m.name.map(norm) == base]
    if x.empty:
        return None
    x = x.assign(value=x.value.astype(float))
    return x.groupby("ts").value.sum().sort_index()


def delta(s, t0, t1):
    if s is None or t0 is None:
        return None
    a = s[s.index <= t0]
    a = a.iloc[-1] if len(a) else s.iloc[0]
    b = s[s.index <= t1 + 2]
    return float(b.iloc[-1] - a) if len(b) else None


def window(s, t0, t1):
    return None if s is None else s[(s.index >= t0) & (s.index <= t1)]


def read_csv(d, name):
    f = pathlib.Path(d) / name
    try:
        return pd.read_csv(f) if f.exists() and f.stat().st_size > 10 else None
    except Exception:
        return None


def hist_avg(m, base, t0, t1):
    s, c = metric_series(m, base + "_sum"), metric_series(m, base + "_count")
    ds, dc = delta(s, t0, t1), delta(c, t0, t1)
    return ds / dc if ds is not None and dc else None


def summarize_run(d) -> dict:
    d = pathlib.Path(d)
    meta = C.jload(d / "meta.json", {})
    cfgj = C.jload(d / "config.json", {})
    row = dict(cfg=d.parent.name, rep=int(d.name.replace("rep", "")), status=meta.get("status"),
               startup_s=meta.get("startup_s"), vram_leak=meta.get("vram_leak"),
               predicted_first=meta.get("prediction_before_run"))
    facts = parse_facts((d / "startup-facts.txt").read_text()) if (d / "startup-facts.txt").exists() else {}
    row.update({k: facts.get(k) for k in ("weights_gib", "kv_gib", "kv_tokens", "max_conc")})
    row["kernel"] = ",".join(k[7:] for k in facts if k.startswith("kernel_") and facts[k]) or None
    if (d / "startup_failed.json").exists():
        row["failed"] = C.jload(d / "startup_failed.json", {}).get("diagnosis")
        return row
    req, ses, err, st, en = frames(d)
    if st is None or en is None or req.empty:
        return row
    dur = en - st
    ok_req = req
    row.update(sessions=len(ses), requests=len(req), errors=len(err), dur_s=round(dur, 1),
               ttft_p50=pct(req.ttft, 50), ttft_p99=pct(req.ttft, 99), itl_p50=pct(req.mean_itl, 50), itl_p95=pct(req.mean_itl, 95),
               gap_p99=pct(req.gap_p99, 50), e2e_p50=pct(req.e2e, 50), e2e_p99=pct(req.e2e, 99),
               sess_p50=pct(ses.dur, 50) if len(ses) else None, sess_p99=pct(ses.dur, 99) if len(ses) else None,
               out_tok_s=req.completion_tokens.sum() / dur, in_tok_s=req.prompt_tokens.sum() / dur,
               tok_per_step=req.completion_tokens.mean(), prompt_per_req=req.prompt_tokens.mean(),
               steps_per_session=ses.steps.mean() if len(ses) else None,
               json_ok=req.tool_json_ok.mean(), truncated=(req.finish == "length").mean(),
               cached_frac=(req.cached_tokens.sum() / req.prompt_tokens.sum()) if req.cached_tokens.notna().any() else None,
               think_s=pct((req.t_answer - req.ttft).dropna(), 50),
               success=ses.success.dropna().astype(float).mean() if len(ses) and ses.success.notna().any() else None,
               ctx_ended=(ses.ended == "context").mean() if len(ses) and "ended" in ses else None)
    row["req_s"] = len(req) / dur
    row["sess_s"] = len(ses) / dur
    m = read_csv(d, "metrics.csv")
    if m is not None:
        kv = metric_series(m, "kv_cache_usage_perc")
        if kv is None:
            kv = metric_series(m, "gpu_cache_usage_perc")
        run, wait = metric_series(m, "num_requests_running"), metric_series(m, "num_requests_waiting")
        w = lambda s: window(s, st, en)
        row["running_max"] = float(w(run).max()) if run is not None and len(w(run)) else None
        row["waiting_max"] = float(w(wait).max()) if wait is not None and len(w(wait)) else None
        row["kv_max"] = float(w(kv).max()) if kv is not None and len(w(kv)) else None
        row["preempt"] = delta(metric_series(m, "num_preemptions"), st, en)
        q, h = delta(metric_series(m, "prefix_cache_queries"), st, en), delta(metric_series(m, "prefix_cache_hits"), st, en)
        row["hit_rate"] = h / q if q else None
        dr, ac = delta(metric_series(m, "spec_decode_num_draft_tokens"), st, en), delta(metric_series(m, "spec_decode_num_accepted_tokens"), st, en)
        nd = delta(metric_series(m, "spec_decode_num_drafts"), st, en)
        row["accept_rate"] = ac / dr if dr else None
        row["accept_len"] = (ac / nd) if nd and ac is not None else None
        for k in ("queue", "prefill", "decode"):
            row[k + "_s"] = hist_avg(m, f"request_{k}_time_seconds", st, en)
    g = read_csv(d, "gpu.csv")
    if g is not None:
        g = g[(g.ts >= st) & (g.ts <= en)]
        if len(g):
            row.update(gpu_util=g.util.mean(), power_w=g.power_w.mean(), sm_min=g.sm_mhz.min(), sm_max=g.sm_mhz.max(),
                       temp_max=g.temp_c.max(), mem_mb=g.mem_mb.max())
            kw = row["power_w"] / 1000
            row["cost_per_1m_out"] = kw * C.PRICE_KWH / (row["out_tok_s"] * 3600) * 1e6 if row["out_tok_s"] else None
    cpu = read_csv(d, "cpu.csv")
    if cpu is not None:
        cpu = cpu[(cpu.ts >= st) & (cpu.ts <= en)]
        if len(cpu):
            row["client_cpu_pct"] = cpu.client_pct.mean()
            row["server_cpu_pct"] = cpu.server_pct.mean()
            row["hot_core_pct"] = float(np.max([np.mean(x) for x in zip(*[json.loads(s) for s in cpu.percpu_json])]))
    return row


# ---------- derived analyses ----------
def goodput(d, ttft=None, itl=None):
    import os
    ttft = ttft or float(os.environ.get("LAB_SLO_TTFT", SLO["ttft"]))
    itl = itl or float(os.environ.get("LAB_SLO_ITL", SLO["itl"]))
    req, ses, err, st, en = frames(d)
    if req.empty or not len(ses):
        return None
    bad = {e["session"] for e in err}
    viol = req[(req.ttft.fillna(99) > ttft) | (req.mean_itl.fillna(99) > itl) | (~req.tool_json_ok.astype(bool))]
    bad |= set(viol.session)
    good = ses[~ses.session.isin(bad)]
    return len(good) / (en - st)


def little(d):
    req, ses, err, st, en = frames(d)
    if ses.empty:
        return None
    arr = (ses.t_start.max() - ses.t_start.min()) / max(len(ses) - 1, 1)
    conc = [((ses.t_start <= t) & (ses.t_start + ses.dur > t)).sum() for t in np.arange(st, en, 1.0)]
    return dict(mean_concurrent=float(np.mean(conc)), lambda_x_duration=float(ses.dur.mean() / arr) if arr > 0 else None)


def class_table(dirs):
    parts = []
    for d in dirs:
        req = frames(d)[0]
        if not req.empty:
            parts.append(req.assign(cfg=pathlib.Path(d).parent.name))
    if not parts:
        return None
    r = pd.concat(parts)
    r["later"] = r.step > 0
    g = r.groupby(["cfg", "cls", "later"])
    t = g[["ttft", "mean_itl", "e2e", "prompt_tokens", "cached_tokens"]].median()
    t["ttft_p99"] = g.ttft.quantile(0.99)
    t["gap_p99_med"] = g.gap_p99.median()
    return t.round(3)


def load_noise():
    return C.jload(C.P("reports", "noise.json"), {})


def vs_base(agg: pd.DataFrame, base: str):
    noise = load_noise()
    if base not in agg.index:
        return agg
    for m in KEYM:
        if m in agg:
            b = agg.loc[base, m]
            agg[m + "_vs%"] = ((agg[m] / b - 1) * 100).round(1) if b else None
            if m in noise:
                agg[m + "_sig"] = agg[m + "_vs%"].abs() > noise[m] * 100
    return agg


NUMCOLS_SHOW = ["status", "sessions", "errors", "out_tok_s", "ttft_p50", "ttft_p99", "itl_p50", "itl_p95", "e2e_p99", "sess_p99",
                "json_ok", "success", "cached_frac", "hit_rate", "truncated", "running_max", "waiting_max", "kv_max", "preempt",
                "gpu_util", "power_w", "sm_min", "sm_max", "temp_max", "startup_s", "kv_tokens"]


def table(exp):
    dirs = run_dirs(exp)
    rows = [summarize_run(d) for d in dirs]
    df = pd.DataFrame(rows)
    return dirs, df


def report(exp, out=print):
    dirs, df = table(exp)
    if df.empty:
        out(f"no runs under runs/{exp}")
        return None
    C.P("reports").mkdir(exist_ok=True)
    df.to_csv(C.P("reports", f"{exp}_runs.csv"), index=False)
    num = df.drop(columns=[c for c in ("rep", "status", "failed", "kernel") if c in df], errors="ignore").select_dtypes("number")
    num = num.assign(cfg=df.cfg)
    agg = num.groupby("cfg", sort=False).median()
    rng = {m + "_rng": num.groupby("cfg", sort=False)[m].agg(lambda s: s.max() - s.min()) for m in KEYM if m in num}
    for k, v in rng.items():
        agg[k] = v
    agg["reps"] = df.groupby("cfg", sort=False).size()
    agg = vs_base(agg, df.cfg.iloc[0])
    agg.to_csv(C.P("reports", f"{exp}.csv"))
    cols = [c for c in ["reps"] + NUMCOLS_SHOW if c in agg and agg[c].notna().any()]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    out(f"\n== {exp}: median over repeats (first config is the baseline) ==")
    out(agg[cols].round(3).to_string())
    flags = [c for c in agg if c.endswith("_vs%") or c.endswith("_sig")]
    if flags:
        out("\nvs baseline (_sig=True means larger than the E0 noise floor):")
        out(agg[flags].to_string())
    failed = df[df.get("failed").notna()] if "failed" in df else df.iloc[0:0]
    for _, r in failed.iterrows():
        out(f"FINDING  {r.cfg}/rep{r.rep}: server did not start -> {r.failed}")
    base = agg.iloc[0]
    for cfg, r in agg.iterrows():
        if "prompt_per_req" in agg and base.get("prompt_per_req") and abs(r.get("prompt_per_req", 0) / base["prompt_per_req"] - 1) > 0.10:
            out(f"DRIFT    {cfg}: mean prompt tokens differ >10% from baseline; compare normalized metrics or step-0 only")
    EXTRAS.get(exp, lambda *a: None)(exp, dirs, df, agg, out)
    return agg


# ---------- experiment-specific extras ----------
def _x0(exp, dirs, df, agg, out):
    d = C.jload(C.P("env", "doctor.json"), {})
    if "c8" in set(df.cfg):
        c8 = df[df.cfg == "c8"]
        if len(c8) > 1:
            noise = {m: float((c8[m].max() - c8[m].min()) / c8[m].median()) for m in KEYM if m in c8 and c8[m].median()}
            C.jdump(noise, C.P("reports", "noise.json"))
            out("\nNOISE FLOOR (relative range across repeats at c=8): " + ", ".join(f"{k} {v:.1%}" for k, v in noise.items()))
    if d.get("itl_floor_ms") and "c1" in agg.index and pd.notna(agg.loc["c1", "itl_p50"]):
        out(f"\nOVERHEAD FACTOR at c=1: measured ITL {agg.loc['c1', 'itl_p50'] * 1000:.1f} ms / roofline floor {d['itl_floor_ms']:.1f} ms = "
            f"{agg.loc['c1', 'itl_p50'] * 1000 / d['itl_floor_ms']:.2f}x")


def _x1(exp, dirs, df, agg, out):
    d = C.jload(C.P("env", "doctor.json"), {})
    bpt, tot = d.get("kv_bytes_per_token"), (d.get("vram_total_mb") or 0) / 1024
    if not bpt or "kv_gib" not in agg:
        return
    t = agg[["kv_gib", "kv_tokens", "weights_gib"]].copy()
    t["tokens_from_gib"] = (t.kv_gib * 2 ** 30 / bpt).round(0)
    t["block_rounding_%"] = ((t.kv_tokens / t.tokens_from_gib - 1) * 100).round(1)
    gmu = t.index.to_series().str.extract(r"gmu([\d.]+)")[0].astype(float)
    if tot and gmu.notna().any():
        t["implied_overhead_gib"] = (tot * gmu - t.weights_gib - t.kv_gib).round(2)
    out("\nKV CAPACITY: logged vs predicted from your bytes/token (overhead should be ~constant across gmu):")
    out(t.round(3).to_string())


def _x5(exp, dirs, df, agg, out):
    rows = []
    for d in dirs:
        n = d.parent.name
        if not n.startswith("load-"):
            continue
        r = summarize_run(d)
        lt = little(d)
        rows.append(dict(cfg=n, frac=float(n.split("-")[1]), rep=d.name, goodput_s=goodput(d), sess_s=r.get("sess_s"), e2e_p99=r.get("e2e_p99"),
                         waiting_max=r.get("waiting_max"), **(lt or {})))
    if not rows:
        return
    t = pd.DataFrame(rows).groupby("frac").median(numeric_only=True)
    out(f"\nOPEN-LOOP CURVE (SAT = {state_sat():.3f} sessions/s):")
    out(t.round(3).to_string())
    base = t.e2e_p99.iloc[0]
    knee = next((f for f, v in t.e2e_p99.items() if v >= 2 * base), None)
    out(f"KNEE (rule: first load where e2e p99 >= 2x the lowest-load p99): {knee} x SAT" if knee else "KNEE: not reached in this sweep")
    t.to_csv(C.P("reports", "E5_curve.csv"))


def state_sat():
    return C.jload(C.P("runs", "_state.json"), {}).get("SAT", float("nan"))


def _x7(exp, dirs, df, agg, out):
    t = agg[[c for c in ("out_tok_s", "itl_p50", "accept_rate", "accept_len") if c in agg]].copy()
    rows = []
    for cfg in t.index:
        mm = re.match(r"(ng\d+|ngram-gpu)-c(\d+)", cfg)
        if mm and f"off-c{mm.group(2)}" in t.index:
            off = t.loc[f"off-c{mm.group(2)}"]
            rows.append(dict(cfg=cfg, conc=int(mm.group(2)), speedup=t.loc[cfg, "out_tok_s"] / off.out_tok_s,
                             accept_rate=t.loc[cfg].get("accept_rate"), accept_len=t.loc[cfg].get("accept_len")))
    if rows:
        s = pd.DataFrame(rows)
        out("\nSPECULATION speedup vs same-concurrency off (crossover = first conc with speedup < 1):")
        out(s.round(3).to_string(index=False))
        s.to_csv(C.P("reports", "E7_speedup.csv"), index=False)
    for d in dirs:
        cf = C.jload(d / "copy_fresh.json")
        if cf:
            out(f"COPY vs FRESH tok/s {d.parent.name}: {cf}")


def _xprobe(exp, dirs, df, agg, out):
    from .micro import compare_probes
    base = None
    for d in dirs:
        if (d / "probe.json").exists():
            base = base or d
            if d != base:
                out(f"PROBE {d.parent.name} vs {base.parent.name}: {compare_probes(base / 'probe.json', d / 'probe.json')}")


def _xclass(exp, dirs, df, agg, out):
    t = class_table(dirs)
    if t is not None:
        out("\nBY CLASS (later=False is step 0). Look for class C ITL spikes while D is in flight:")
        out(t.to_string())


def _x12(exp, dirs, df, agg, out):
    for d in dirs:
        for f in ("toolong.json", "disconnect.json"):
            j = C.jload(d / f)
            if j:
                j.pop("timeline", None)
                out(f"{d.parent.name}/{f}: {j}")
    for d in dirs:
        if d.parent.name == "soak":
            g = read_csv(d, "gpu.csv")
            if g is not None and len(g) > 60:
                h = len(g) // 2
                out(f"SOAK mem_mb first/second half max: {g.mem_mb[:h].max()} / {g.mem_mb[h:].max()}; sm_mhz mean {g.sm_mhz[:h].mean():.0f} -> {g.sm_mhz[h:].mean():.0f}")


EXTRAS = {"E0": _x0, "E1": lambda *a: (_x1(*a), _xprobe(*a)), "E3": _xclass, "E5": _x5, "E7": lambda *a: (_x7(*a), _xprobe(*a)),
          "E8": _xprobe, "E9": _xclass, "E12": _x12, "E2": _xclass, "E13": _xprobe}
