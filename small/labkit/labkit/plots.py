"""All figures regenerate from raw logs: `lab plot <E#|all>`."""
from __future__ import annotations
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from . import core as C
from . import analysis as A


def _save(name, title, xl, yl):
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.grid(alpha=0.3)
    C.P("plots").mkdir(exist_ok=True)
    f = C.P("plots", name)
    plt.tight_layout()
    plt.savefig(f, dpi=150)
    plt.close()
    print("wrote", f)


def _agg(exp):
    f = C.P("reports", f"{exp}.csv")
    if not f.exists():
        A.report(exp, out=lambda *a: None)
    return pd.read_csv(f, index_col=0) if f.exists() else None


def e0():
    g = _agg("E0")
    if g is None:
        return
    g = g.assign(c=g.index.str.extract(r"c(\d+)")[0].astype(float).values).sort_values("c")
    plt.errorbar(g.c, g.out_tok_s, yerr=g.get("out_tok_s_rng"), marker="o")
    plt.xscale("log", base=2)
    _save("scaling.png", "E0 throughput vs concurrency", "concurrency", "output tokens/s")
    d = C.jload(C.P("env", "doctor.json"), {})
    if d.get("itl_floor_ms"):
        plt.plot(g.c, g.itl_p50 * 1000, marker="o", label="measured ITL p50")
        plt.axhline(d["itl_floor_ms"], ls="--", color="r", label="roofline floor")
        plt.xscale("log", base=2)
        plt.legend()
        _save("roofline_vs_itl.png", "E0 ITL vs roofline floor", "concurrency", "ms/token")


def e1():
    g = _agg("E1")
    if g is None or "kv_tokens" not in g:
        return
    d = C.jload(C.P("env", "doctor.json"), {})
    bpt, tot = d.get("kv_bytes_per_token"), (d.get("vram_total_mb") or 0) / 1024
    s = g[g.index.str.startswith("gmu")].copy()
    if len(s) and bpt:
        s["gmu"] = s.index.str.extract(r"gmu([\d.]+)")[0].astype(float).values
        s["pred"] = (tot * s.gmu - s.weights_gib.iloc[0] - (tot * 0.80 - s.weights_gib.iloc[0] - s.kv_gib.get("gmu0.80", s.kv_gib.iloc[0]))) * 2 ** 30 / bpt
        plt.plot(s.gmu, s.kv_tokens, "o-", label="logged")
        plt.plot(s.gmu, s.pred, "s--", label="predicted (overhead calibrated at 0.80)")
        plt.legend()
        _save("kv_pred_vs_measured.png", "E1 KV tokens", "gpu-memory-utilization", "KV tokens")
    kv_timeline("E1", "press-c32")


def kv_timeline(exp, cfg):
    d = C.P("runs", exp, cfg, "rep1")
    m = A.read_csv(d, "metrics.csv")
    if m is None:
        return
    req, ses, err, st, en = A.frames(d)
    kv = A.metric_series(m, "kv_cache_usage_perc")
    if kv is None:
        kv = A.metric_series(m, "gpu_cache_usage_perc")
    pre = A.metric_series(m, "num_preemptions")
    run = A.metric_series(m, "num_requests_running")
    if kv is None:
        return
    fig, ax = plt.subplots()
    ax.plot(kv.index - st, kv.values, label="KV usage")
    if run is not None:
        ax.plot(run.index - st, run.values / max(run.max(), 1), label="running (norm)")
    if pre is not None:
        for t in pre.index[pre.diff().fillna(0) > 0]:
            ax.axvline(t - st, color="r", alpha=0.3)
    ax.legend()
    ax.set_xlabel("seconds")
    plt.title(f"{exp}/{cfg}: KV usage (red = preemption)")
    C.P("plots").mkdir(exist_ok=True)
    plt.savefig(C.P("plots", f"kv_timeline_{exp}_{cfg}.png"), dpi=150)
    plt.close()


def e2():
    g = _agg("E2")
    if g is None:
        return
    for mode in ("on", "off"):
        s = g[g.index.str.startswith(mode + "-P")].copy()
        if len(s):
            s["P"] = s.index.str.extract(r"P(\d+)")[0].astype(float).values
            plt.plot(s.P, s.hit_rate, "o-", label=f"cache {mode}")
    plt.xscale("log", base=2)
    plt.legend()
    _save("hitrate_vs_projects.png", "E2 prefix hit rate vs projects", "projects", "hit rate")


def e5():
    f = C.P("reports", "E5_curve.csv")
    if not f.exists():
        A.report("E5", out=lambda *a: None)
    if not f.exists():
        return
    t = pd.read_csv(f)
    fig, ax = plt.subplots()
    ax.plot(t.frac, t.e2e_p99, "o-", label="e2e p99 (s)")
    ax.set_xlabel("offered load / SAT")
    ax.set_ylabel("e2e p99 (s)")
    ax2 = ax.twinx()
    ax2.plot(t.frac, t.waiting_max, "s--", color="C1", label="waiting max")
    ax2.set_ylabel("waiting requests")
    plt.title("E5 latency vs offered load")
    C.P("plots").mkdir(exist_ok=True)
    plt.savefig(C.P("plots", "knee.png"), dpi=150)
    plt.close()
    plt.plot(t.frac, t.goodput_s, "o-", label="goodput")
    plt.plot(t.frac, t.sess_s, "s--", label="completed sessions/s")
    plt.legend()
    _save("goodput_vs_load.png", f"E5 goodput (SLO TTFT {A.SLO['ttft']}s, ITL {A.SLO['itl']}s; session good only if every step meets it)", "offered load / SAT", "sessions/s")


def latency_split(exp):
    g = _agg(exp)
    if g is None or not {"queue_s", "prefill_s", "decode_s"} <= set(g):
        return
    g[["queue_s", "prefill_s", "decode_s"]].fillna(0).plot(kind="bar", stacked=True)
    _save(f"latency_split_{exp}.png", f"{exp} mean queue / prefill / decode time (server histograms)", "config", "seconds")


def e7():
    f = C.P("reports", "E7_speedup.csv")
    if not f.exists():
        A.report("E7", out=lambda *a: None)
    if not f.exists():
        return
    s = pd.read_csv(f)
    s["n"] = s.cfg.str.extract(r"ng(\d+)")[0]
    for n, x in s.groupby("n", dropna=False):
        plt.plot(x.conc, x.speedup, "o-", label=f"N={n}")
    plt.axhline(1, color="k", lw=0.8)
    plt.xscale("log", base=2)
    plt.legend()
    _save("spec_speedup.png", "E7 n-gram speedup vs concurrency", "concurrency", "throughput vs off")
    s2 = s.dropna(subset=["accept_rate"])
    if len(s2):
        for n, x in s2.groupby("n"):
            plt.plot(x.conc, x.accept_rate, "o-", label=f"N={n}")
        plt.legend()
        _save("spec_acceptance.png", "E7 acceptance rate", "concurrency", "accepted / drafted")


def e13():
    g = _agg("E13")
    if g is None:
        return
    rows = g[g.index.str.match(r"base|add\d+")]
    if "sess_s" in rows and len(rows):
        plt.bar(rows.index, rows.sess_s)
        _save("waterfall.png", "E13 ablation waterfall (completed sessions/s at fixed load)", "config", "sessions/s")
    mm = g[g.index.str.match(r"minus\d+")]
    if len(mm):
        plt.bar(mm.index, mm.sess_s)
        _save("leave_one_out.png", "E13 leave-one-out", "config", "sessions/s")


def cost_quality():
    pts = []
    for e in ("E8", "E13"):
        g = _agg(e)
        if g is not None and "cost_per_1m_out" in g:
            for k, r in g.iterrows():
                pts.append((e + "/" + k, r.get("cost_per_1m_out"), r.get("success"), r.get("json_ok")))
    pts = [p for p in pts if p[1] == p[1]]
    for n, c, s, j in pts:
        y = s if s == s else j
        plt.scatter(c, y)
        plt.annotate(n, (c, y), fontsize=6)
    if pts:
        _save("cost_vs_quality.png", f"Energy cost per 1M output tokens (assumed ${C.PRICE_KWH}/kWh) vs quality", "$ per 1M out tokens", "success (or tool-JSON validity)")


def all_plots(which="all"):
    jobs = {"E0": e0, "E1": e1, "E2": e2, "E5": e5, "E7": e7, "E13": e13,
            "E3": lambda: latency_split("E3"), "E9": lambda: latency_split("E9"), "E12": lambda: kv_timeline("E12", "over"),
            "E8": cost_quality}
    for k, fn in jobs.items():
        if which in ("all", k):
            try:
                fn()
            except Exception as e:
                print(f"plot {k} skipped: {e!r}")
