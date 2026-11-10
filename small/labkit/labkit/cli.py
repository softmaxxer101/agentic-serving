"""`lab` command line."""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, tempfile, time
from . import core as C

HELP = """lab <command>
  doctor [--warm]       capture machine, measure bandwidth, print predictions
  smoke                 five sanity checks against a running server
  notes <E#>            create/show the notes template for an experiment
  up [preset] [--arg '--flag v']... [--env K=V]... [--drop --flag]...    start a server (stays up)
  run <workload> [--conc N | --load open --rate R] [--n N] [--duration S] [--think on|off|trace] [--max-tokens N] [--projects P]
  exp <E#> [--reps 3] [--only text] [--dry-run] [--force] [--quick] [--full] [--winner '<flags>']... [--no-predict-check]
  report <E#|run_dir>   tables from raw logs          plot <E#|all>   figures from raw logs
  compare <runA> <runB> side-by-side run summaries    probe <run_dir>  quality probe vs the experiment's first probe
  down                  stop the server, confirm VRAM is back to idle
  bonus profile|structured|engines|engines-vllm
  selftest              verify the kit itself on a built-in mock server (no GPU)
  reproduce             run E0..E12, report, plot
"""


def _srv_state():
    return C.jload(C.P("runs", ".server.json"), {})


def cmd_up(a):
    from .server import Caps, Server
    base = C.PRESETS.get(a.preset, a.preset)
    tokens = C.merge_args(base, " ".join(a.arg), a.drop)
    tokens, dropped = Caps.detect().filter(tokens)
    out = C.P("runs", "adhoc", "server-" + time.strftime("%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    env = dict(kv.split("=", 1) for kv in a.env)
    srv = Server(out, tokens, env)
    print("starting:", " ".join(tokens), ("\n(dropped unknown flags: %s)" % dropped) if dropped else "")
    ok, s = srv.start()
    if not ok:
        from .server import diagnose
        print("server failed; see", out / "server.log", "\n", diagnose(open(out / "server.log", errors="replace").read()[-4000:]))
        return 1
    C.jdump({"pid": srv.pid, "port": srv.port, "out": str(out), "vram_before": C.vram_used_mb()}, C.P("runs", ".server.json"))
    from .server import extract_facts
    print(f"up in {s:.0f}s; facts:", extract_facts(out / "server.log", out / "startup-facts.txt"))


def cmd_down(a):
    from .server import kill_tree, sweep_stragglers, wait_vram_idle
    st = _srv_state()
    if st.get("pid"):
        kill_tree(st["pid"])
    sweep_stragglers()
    v = wait_vram_idle(st.get("vram_before"))
    print("server stopped; VRAM now", v, "MB (idle was", st.get("vram_before"), ")")
    C.P("runs", ".server.json").unlink(missing_ok=True)


def cmd_run(a):
    import asyncio
    from .workload import RunCfg, TextBank, run_workload
    from .sidecars import Sidecars
    from . import analysis as A
    st = _srv_state()
    port = st.get("port", C.PORT)
    out = C.P("runs", "adhoc", f"{a.workload}-{time.strftime('%H%M%S')}", "rep1")
    out.mkdir(parents=True)
    rc = RunCfg(workload=a.workload, load=a.load, conc=a.conc, rate=a.rate, n=a.n, duration=a.duration, think=a.think,
                max_tokens=a.max_tokens, projects=a.projects, port=port, nudge=a.nudge, prio=a.prio, burst=a.burst,
                tool_choice=a.tool_choice, max_len=a.max_len)
    side = Sidecars(out, port)
    side.start_gpu()
    side.start_service(st.get("pid"))
    C.jdump({"adhoc": True, "cfg": rc.__dict__}, out / "config.json")
    asyncio.run(run_workload(rc, out / "requests.jsonl", TextBank()))
    side.stop()
    C.jdump({"status": "ok"}, out / "meta.json")
    import pandas as pd
    pd.set_option("display.width", 200)
    print(pd.Series(A.summarize_run(out)).dropna().to_string())
    print("run folder:", out)


def cmd_exp(a):
    from . import runner
    runner.run_exp(a.exp, reps=a.reps, opts=dict(only=a.only, quick=a.quick, full=a.full, winners=a.winner, soak_min=a.soak_min),
                   force=a.force, dry=a.dry_run, predict_check=not a.no_predict_check)


def cmd_report(a):
    from . import analysis as A
    import pandas as pd
    if os.path.isdir(a.target):
        pd.set_option("display.width", 200)
        print(pd.Series(A.summarize_run(a.target)).dropna().to_string())
    else:
        A.report(a.target)


def cmd_compare(a):
    from . import analysis as A
    import pandas as pd
    x, y = pd.Series(A.summarize_run(a.a)), pd.Series(A.summarize_run(a.b))
    df = pd.DataFrame({"A": x, "B": y})
    num = df.apply(pd.to_numeric, errors="coerce")
    df["B/A-1 %"] = ((num.B / num.A - 1) * 100).round(1)
    pd.set_option("display.width", 200)
    print(df.dropna(how="all").to_string())
    print("\nreminder: an effect smaller than your E0 noise floor (reports/noise.json) is not a result")


def cmd_probe(a):
    import pathlib
    from .micro import compare_probes, quality_probe
    d = pathlib.Path(a.run)
    if not (d / "probe.json").exists():
        st = _srv_state()
        quality_probe(d, st.get("port", C.PORT))
    sibs = sorted(p for p in d.parent.parent.glob("*/rep*/probe.json") if p.parent != d)
    if not sibs:
        print("no other probe.json in this experiment; run a baseline with probe=True")
        return
    print(f"{d} vs baseline {sibs[0].parent}:", compare_probes(sibs[0], d / "probe.json"))


def cmd_notes(a):
    from . import runner
    runner.check_prediction(a.exp, allow_missing=True)
    print(runner.notes_path(a.exp).read_text())


def cmd_selftest(a):
    """Runs the whole pipeline (workload -> sidecars -> report -> plots) against the built-in mock."""
    os.environ["LAB_TOKENIZER"] = "none"
    C.HOME = type(C.HOME)(tempfile.mkdtemp(prefix="labself-"))
    from . import analysis as A, plots, runner
    from .experiments import Cfg
    from .server import Caps
    print("selftest home:", C.HOME)
    cfgs = [Cfg("c1", wl=dict(workload="mix", conc=1, n=8, warmup=2, tool_latency=0.01)),
            Cfg("c8", wl=dict(workload="mix", conc=8, n=24, warmup=2, tool_latency=0.01)),
            Cfg("sat-c16", wl=dict(workload="mix", conc=16, n=24, warmup=2, tool_latency=0.01)),
            Cfg("load-1.0", wl=dict(workload="mix", rate_frac=1.0, n=20, warmup=2, tool_latency=0.01)),
            Cfg("load-2.0", wl=dict(workload="mix", rate_frac=2.0, n=20, warmup=2, tool_latency=0.01))]
    for c in cfgs:
        for rep in (1, 2):
            s = runner.run_one("E0", c, rep, Caps(), mock=True)
            print(f"  {c.name}/rep{rep}: {s}")
    A.report("E0")
    plots.all_plots("E0")
    ok = all((C.P("reports", f).exists()) for f in ("E0.csv", "E0_runs.csv")) and (C.P("plots", "scaling.png").exists())
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_reproduce(a):
    from . import runner, plots, analysis as A
    from .experiments import EXPS
    for e in EXPS:
        if e == "E13":
            continue
        runner.run_exp(e, reps=a.reps)
        A.report(e)
    plots.all_plots("all")


def main(argv=None):
    p = argparse.ArgumentParser(prog="lab", usage=HELP)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("doctor"); s.add_argument("--warm", action="store_true"); s.add_argument("--refresh", action="store_true")
    sp.add_parser("smoke")
    s = sp.add_parser("notes"); s.add_argument("exp")
    s = sp.add_parser("up"); s.add_argument("preset", nargs="?", default="base")
    s.add_argument("--arg", action="append", default=[]); s.add_argument("--env", action="append", default=[]); s.add_argument("--drop", action="append", default=[])
    s = sp.add_parser("run"); s.add_argument("workload")
    s.add_argument("--conc", type=int, default=4); s.add_argument("--load", default="closed", choices=["closed", "open"])
    s.add_argument("--rate", type=float, default=0.5); s.add_argument("--n", type=int, default=60); s.add_argument("--duration", type=float, default=0)
    s.add_argument("--think", default="trace", choices=["trace", "on", "off"]); s.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    s.add_argument("--projects", type=int, default=4); s.add_argument("--nudge", default=""); s.add_argument("--prio", action="store_true")
    s.add_argument("--burst", action="store_true"); s.add_argument("--tool-choice", default="auto", dest="tool_choice", choices=["auto", "required-first"])
    s.add_argument("--max-len", type=int, default=4096, dest="max_len")
    s = sp.add_parser("exp"); s.add_argument("exp"); s.add_argument("--reps", type=int, default=3); s.add_argument("--only", default="")
    s.add_argument("--dry-run", action="store_true"); s.add_argument("--force", action="store_true"); s.add_argument("--quick", action="store_true")
    s.add_argument("--full", action="store_true"); s.add_argument("--winner", action="append", default=[]); s.add_argument("--soak-min", type=int, default=30, dest="soak_min")
    s.add_argument("--no-predict-check", action="store_true", dest="no_predict_check")
    s = sp.add_parser("report"); s.add_argument("target")
    s = sp.add_parser("plot"); s.add_argument("which", nargs="?", default="all")
    s = sp.add_parser("compare"); s.add_argument("a"); s.add_argument("b")
    s = sp.add_parser("probe"); s.add_argument("run")
    sp.add_parser("down")
    s = sp.add_parser("bonus"); s.add_argument("which", choices=["profile", "structured", "engines", "engines-vllm"])
    sp.add_parser("selftest")
    s = sp.add_parser("reproduce"); s.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)
    if a.cmd == "doctor":
        from . import doctor
        doctor.run(a.warm, a.refresh)
    elif a.cmd == "smoke":
        from . import micro
        for k, v in micro.smoke().items():
            print(f"{k:32s} {v}")
    elif a.cmd == "plot":
        from . import plots
        plots.all_plots(a.which)
    elif a.cmd == "bonus":
        from . import bonus
        {"profile": bonus.profile, "structured": bonus.structured, "engines": bonus.engines, "engines-vllm": bonus.engines_vllm}[a.which]()
    else:
        return globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
