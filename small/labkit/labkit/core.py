"""Shared constants and small utilities for the Lab Kit (target: vLLM 0.32.0, RTX 2050 4GB)."""
from __future__ import annotations
import json, os, pathlib, re, shlex, subprocess, time

TARGET_VLLM = "0.32.0"
HOME = pathlib.Path(os.environ.get("LAB_HOME", ".")).resolve()
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
KEY = os.environ.get("KEY", "lab-secret-key")
PORT = int(os.environ.get("LAB_PORT", "8000"))
GPU = os.environ.get("GPU", "2050")
SERVED = "lab"  # served-model-name used by every client in the kit
PRICE_KWH = float(os.environ.get("LAB_PRICE_KWH", "0.15"))  # ASSUMPTION, label it in the write-up

# Preset "base": 4 GB card, eager mode first (graphs are tested in E4).
BASE_ARGS = (
    "--max-model-len 4096 --gpu-memory-utilization 0.80 --max-num-seqs 16 --enforce-eager "
    "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes "
    "--enable-prompt-tokens-details"
)
PRESETS = {"base": BASE_ARGS}


def P(*a) -> pathlib.Path:
    return HOME.joinpath(*a)


def jdump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def jload(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception:
        return default


def sh(cmd, timeout=30) -> str:
    """Run a command, return stdout+stderr text ('' on failure)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def vram_used_mb():
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    try:
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def vram_total_mb():
    out = sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    try:
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def vllm_version():
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return None


# ---------- serve-argument merging ----------
def _is_flag(t: str) -> bool:
    return t.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", t)


def _key(flag: str) -> str:
    if re.fullmatch(r"-O\d*", flag):
        return "-O"
    return re.sub(r"^--no-", "--", flag)


def to_pairs(tokens):
    pairs = []
    for t in tokens:
        if _is_flag(t):
            pairs.append([t, []])
        elif pairs:
            pairs[-1][1].append(t)
        else:
            raise ValueError(f"value without flag: {t}")
    return pairs


def merge_args(base: str, extra: str = "", drop=()) -> list[str]:
    """Later flags override earlier ones; `drop` removes flags (with or without --no- form)."""
    pairs = to_pairs(shlex.split(base))
    for f, v in to_pairs(shlex.split(extra)):
        k = _key(f)
        pairs = [p for p in pairs if _key(p[0]) != k]
        pairs.append([f, v])
    dk = {_key(d if d.startswith("-") else "--" + d) for d in drop}
    pairs = [p for p in pairs if _key(p[0]) not in dk]
    out = []
    for f, v in pairs:
        out.append(f)
        out.extend(v)
    return out


def arg_value(tokens, flag, default=None):
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return tokens[i + 1]
    return default


def fmt_s(x, nd=3):
    return None if x is None else round(float(x), nd)
