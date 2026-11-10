"""Runs one config (server restart, sidecars, workload, micro tests) and whole experiments (with resume)."""
from __future__ import annotations
import asyncio, dataclasses, json, re, shutil, sys, time
from . import core as C
from . import experiments as X
from . import micro
from .server import Caps, Server, extract_facts, diagnose, wait_vram_idle
from .sidecars import Sidecars
from .workload import RunCfg, TextBank, run_workload

MINUTES_PER_RUN = 12


# ---------- notes / prediction discipline ----------
def notes_path(exp):
    return C.P("notes", f"{exp}.md")


TEMPLATE = """# {exp}
Date / vLLM version / GPU:
Concept: {concept}
Prediction (numbers, written BEFORE running):

Result (table from lab report):

Explanation (tied to one metric or log line):

Surprises:

Negative results (what did nothing):

Next question:
"""


def check_prediction(exp, allow_missing=False):
    p = notes_path(exp)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TEMPLATE.format(exp=exp, concept=X.CONCEPT.get(exp, "")))
        if allow_missing:
            return False
        raise SystemExit(f"Created {p}. Write your numeric prediction under 'Prediction', then rerun. (Rule 1: predict before you measure.)")
    txt = p.read_text()
    m = re.search(r"^Prediction[^\n]*:(.*?)(?=^\S[^\n]{0,70}:\s*$|^#|\Z)", txt, re.S | re.M)
    body = re.sub(r"<!--.*?-->", "", m.group(1) if m else "", flags=re.S).strip()
    if not body:
        if allow_missing:
            return False
        raise SystemExit(f"{p} has an empty Prediction. Write numbers first, then rerun (or pass --no-predict-check).")
    return True


# ---------- state (saturation point) ----------
def state():
    return C.jload(C.P("runs", "_state.json"), {})


def save_state(d):
    C.jdump(d, C.P("runs", "_state.json"))


def update_sat(out):
    recs = [json.loads(l) for l in open(out / "requests.jsonl")]
    done = [r for r in recs if r.get("session_done")]
    st = next(r["run_start"] for r in recs if "run_start" in r)
    en = next(r["run_end"] for r in recs if "run_end" in r)
    if done and en > st:
        sat = len(done) / (en - st)
        s = state()
        s.setdefault("SAT_samples", []).append(sat)
        xs = sorted(s["SAT_samples"])
        s["SAT"] = xs[len(xs) // 2]
        save_state(s)


# ---------- one run ----------
def run_one(exp, cfg, rep, caps, force=False, mock=False, predicted=True):
    out = C.P("runs", exp, cfg.name, f"rep{rep}")
    if (out / "DONE").exists() and not force:
        return "skip"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tokens = C.merge_args(C.BASE_ARGS, cfg.args, cfg.drop)
    dropped = []
    if not mock:
        tokens, dropped = caps.filter(tokens)
    max_len = int(C.arg_value(tokens, "--max-model-len", 4096))
    wl = dict(cfg.wl)
    frac = wl.pop("rate_frac", None)
    if frac is not None:
        sat = state().get("SAT")
        if not sat:
            raise SystemExit("SAT unknown: run `lab exp E5` first (it measures saturation throughput).")
        wl.update(rate=round(frac * sat, 4), load="open")
    wl.setdefault("max_len", max_len)
    rc = RunCfg(**wl)
    t_start = time.time()
    C.jdump({"exp": exp, "cfg": dataclasses.asdict(cfg) | {"wl": wl}, "tokens": tokens, "dropped_flags": dropped,
             "rep": rep, "gpu": C.GPU, "model": cfg.model or C.MODEL}, out / "config.json")
    vram0 = C.vram_used_mb()
    if cfg.prefetch and not mock:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(cfg.prefetch)
        except Exception as e:
            (out / "prefetch_error.txt").write_text(repr(e))
    if mock:
        from .mock import MockServer
        srv = MockServer(out)
    else:
        srv = Server(out, tokens, cfg.env, cfg.model)
    side = Sidecars(out, srv.port)
    side.start_gpu()
    ok, startup_s = srv.start()
    status = "ok"
    if not ok:
        tail = ""
        try:
            tail = "".join(open(out / "server.log", errors="replace").readlines()[-60:])
        except Exception:
            pass
        C.jdump({"startup_failed": True, "startup_s": startup_s, "diagnosis": diagnose(tail), "log_tail": tail[-3000:]},
                out / "startup_failed.json")
        status = "failed"
    else:
        facts = extract_facts(out / "server.log", out / "startup-facts.txt")
        side.start_service(srv.pid)
        if not (C.P("env", "metric_names.txt")).exists():
            try:
                from .server import fetch_metrics
                names = sorted({l.split("{")[0].split(" ")[0] for l in fetch_metrics(srv.port).splitlines() if l.startswith("vllm:")})
                C.P("env").mkdir(exist_ok=True)
                C.P("env", "metric_names.txt").write_text("\n".join(names))
            except Exception:
                pass
        try:
            asyncio.run(run_workload(rc, out / "requests.jsonl", TextBank()))
            for m in cfg.micro:
                if m == "copy_fresh":
                    micro.copy_fresh(out, srv.port)
                elif m == "disconnect":
                    micro.disconnect_storm(out, srv.port)
                elif m == "toolong":
                    micro.too_long(out, max_len, srv.port)
            if cfg.probe and not mock:
                micro.quality_probe(out, srv.port)
        except Exception as e:
            (out / "run_error.txt").write_text(repr(e))
            status = "run-error"
    side.stop()
    srv.stop()
    vram1 = wait_vram_idle(vram0)
    (out / "vram-after.txt").write_text(f"before={vram0}\nafter={vram1}\n")
    C.jdump({"vllm": C.vllm_version(), "target": C.TARGET_VLLM, "startup_s": startup_s, "vram_before_mb": vram0,
             "vram_after_mb": vram1, "vram_leak": bool(vram0 and vram1 and vram1 > vram0 + 200),
             "prediction_before_run": predicted, "t_start": t_start, "t_end": time.time(), "status": status},
            out / "meta.json")
    if status == "ok" and cfg.name.startswith("sat-") and (out / "requests.jsonl").exists():
        update_sat(out)
    (out / "DONE").write_text(status)
    return status


def plan(exp, reps, opts, caps):
    st = state()
    cfgs = X.build(exp, caps, st, opts)
    if opts.get("only"):
        cfgs = [c for c in cfgs if opts["only"] in c.name]
    runs = [(c, r) for c in cfgs for r in range(1, (c.reps or reps) + 1)]
    return cfgs, runs, st


def run_exp(exp, reps=3, opts=None, force=False, dry=False, predict_check=True, mock=False):
    opts = opts or {}
    caps = Caps() if mock else Caps.detect()
    predicted = True if mock else (check_prediction(exp) if predict_check else check_prediction(exp, allow_missing=True))
    if opts.get("quick"):
        reps = 1
    cfgs, runs, st = plan(exp, reps, opts, caps)
    if st.get("_need_sat") and exp != "E5" and not dry:
        print("SAT unknown -> running E5 saturation point first")
        run_exp("E5", reps=1, opts={"only": "sat-"}, force=False, predict_check=False, mock=mock)
        cfgs, runs, st = plan(exp, reps, opts, caps)
    print(f"{exp}: {len(cfgs)} configs, {len(runs)} runs, ~{len(runs) * MINUTES_PER_RUN} min (est.)")
    for c, r in runs:
        print(f"  {c.name} rep{r}: {c.args or '(base)'}")
    if dry:
        return
    for i, (c, r) in enumerate(runs, 1):
        t = time.time()
        print(f"[{i}/{len(runs)}] {exp}/{c.name}/rep{r} ...", flush=True)
        s = run_one(exp, c, r, caps, force=force, mock=mock, predicted=predicted)
        print(f"    -> {s} ({time.time() - t:.0f}s)", flush=True)
    print(f"done. next: lab report {exp}; lab plot {exp}")
