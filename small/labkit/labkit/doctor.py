"""`lab doctor`: capture the machine, measure bandwidth, read the model config, print your predictions."""
from __future__ import annotations
import glob, json, os, platform, shutil, sys, time
from . import core as C
from .server import Caps


def _write(name, text):
    C.P("env").mkdir(exist_ok=True)
    C.P("env", name).write_text(text)


def bandwidth():
    import torch
    n = 256 * 1024 * 1024
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    B = x.numel() * 2 * 2 * 20 / 1e9 / (time.perf_counter() - t)
    h = torch.empty(n, dtype=torch.bfloat16).pin_memory()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        x.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    H = x.numel() * 2 * 20 / 1e9 / (time.perf_counter() - t)
    del x, y, h
    torch.cuda.empty_cache()
    return B, H


def run(warm=False, refresh=False):
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "target_vllm": C.TARGET_VLLM}
    for name, cmd in (("nvidia-smi.txt", "nvidia-smi"), ("nvidia-smi-q.txt", "nvidia-smi -q"), ("lscpu.txt", "lscpu"),
                      ("free.txt", "free -h"), ("df.txt", "df -h"), ("uname.txt", "uname -a")):
        _write(name, C.sh(cmd))
    out["python"] = sys.version.split()[0]
    out["wsl"] = "microsoft" in platform.release().lower()
    out["vllm"] = C.vllm_version()
    print(f"vLLM installed: {out['vllm']}  (kit targets {C.TARGET_VLLM})")
    if out["vllm"] != C.TARGET_VLLM:
        print(f"  WARNING: flags/metrics may differ. Pin it: pip install vllm=={C.TARGET_VLLM}")
    py = sys.version_info
    if not ((3, 10) <= (py.major, py.minor) < (3, 14)):
        print(f"  WARNING: Python {py.major}.{py.minor}; vLLM 0.32.0 wheels expect 3.10-3.13 (the omni add-on needs 3.11+).")
    drv = C.sh("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,power.limit --format=csv,noheader").strip()
    out["gpu_summary"] = drv
    print("GPU:", drv or "NOT FOUND")
    try:
        major = int(drv.split(",")[1].strip().split(".")[0])
        print(f"  driver {major}: " + ("ok for CUDA 13.0 wheels (default in 0.32.0)" if major >= 580 else
              "OLDER than 580. 0.32.0 default wheels target CUDA 13.0; use the cu129 build from the release page or update the driver."))
    except Exception:
        pass
    out["vram_idle_mb"], out["vram_total_mb"] = C.vram_used_mb(), C.vram_total_mb()
    if out["vram_idle_mb"] and out["vram_idle_mb"] > 300:
        print(f"  WARNING: {out['vram_idle_mb']} MB already used with nothing running; close browsers/video NOW (it is stolen from your KV pool).")
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            out["capability"] = list(torch.cuda.get_device_capability(0))
            print(f"  compute capability {out['capability']} (8.6 = Ampere: no FP8/FP4 tensor cores)")
            B, H = bandwidth()
            out["bandwidth_gbs"], out["h2d_gbs"] = B, H
            print(f"  device copy B = {B:.1f} GB/s ; host-to-device H = {H:.1f} GB/s")
    except Exception as e:
        print("  torch check failed:", repr(e))
    # model facts
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(C.MODEL)
        cfg = json.load(open(os.path.join(path, "config.json")))
        _write("model-config.json", json.dumps(cfg, indent=2))
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        bpt = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2
        wbytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors")))
        out.update(kv_bytes_per_token=bpt, weights_bytes=wbytes, layers=cfg["num_hidden_layers"], kv_heads=cfg["num_key_value_heads"], head_dim=hd)
        print(f"model: {C.MODEL}  layers={cfg['num_hidden_layers']} kv_heads={cfg['num_key_value_heads']} head_dim={hd}")
        print(f"PREDICTION 1  KV bytes/token (bf16) = {bpt:,}   fp8 = {bpt // 2:,}   (expect 114,688 / 57,344)")
        if out.get("bandwidth_gbs"):
            fl = wbytes / (out["bandwidth_gbs"] * 1e9) * 1000
            out["itl_floor_ms"] = fl
            print(f"PREDICTION 2  decode floor = {wbytes / 1e9:.2f} GB / {out['bandwidth_gbs']:.0f} GB/s = {fl:.1f} ms/token at c=1 (real ITL is higher; the gap is overhead)")
        if out.get("vram_total_mb"):
            tot = out["vram_total_mb"] / 1024
            est = (tot * 0.80 - wbytes / 2 ** 30 - 0.9) * 2 ** 30 / bpt
            out["kv_tokens_estimate_base"] = est
            print(f"PREDICTION 3  rough KV pool at gmu 0.80 ~ ({tot:.1f} GiB x 0.80 - weights - ~0.9 GiB overhead) / bytes-per-token = {est:,.0f} tokens"
                  f"  (overhead is a guess; E1 measures it)")
    except Exception as e:
        print("  model facts failed (network? run `hf download` first):", repr(e))
    if warm and shutil.which("vllm"):
        print("warming kernels: vllm download-kernels (0.32.0)")
        print(C.sh(["vllm", "download-kernels"], timeout=1800)[-400:])
    caps = Caps.detect(refresh=True)
    out["flags_detected"] = len(caps.flags)
    print(f"flags detected from `vllm serve --help=all`: {len(caps.flags)}")
    for f in ("--kv-cache-dtype", "--speculative-config", "--scheduling-policy", "--async-scheduling", "--api-server-count",
              "--cpu-offload-gb", "--kv-offloading-size", "--no-enable-chunked-prefill", "--compilation-config"):
        print(f"  {f:32s} {'yes' if caps.has(f) else 'MISSING (record as finding)'}")
    print("  kv-cache-dtype choices:", caps.choices("--kv-cache-dtype") or "unknown")
    C.jdump(out, C.P("env", "doctor.json"))
    print("saved env/doctor.json. Now write your predictions in notes/E0.md, then: lab smoke ; lab exp E0")
    return out
