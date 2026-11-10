"""vLLM server lifecycle, flag detection, startup-fact parsing."""
from __future__ import annotations
import os, re, signal, subprocess, time, urllib.request
from . import core as C


class Caps:
    """What this vLLM build accepts, discovered from `vllm serve --help=all` (cached per version)."""

    def __init__(self, text: str = ""):
        self.text = text
        self.flags = set(re.findall(r"(?<![\w-])(-{1,2}[A-Za-z][\w-]*)", text))

    @classmethod
    def detect(cls, refresh=False):
        cache = C.P("env", "flags.json")
        ver = C.vllm_version()
        d = C.jload(cache)
        if d and not refresh and d.get("vllm") == ver and d.get("text"):
            return cls(d["text"])
        text = ""
        for cmd in (["vllm", "serve", "--help=all"], ["vllm", "serve", "--help"]):
            text = C.sh(cmd, timeout=180)
            if len(text) > 500:
                break
        if text:
            C.jdump({"vllm": ver, "text": text}, cache)
        return cls(text)

    def has(self, flag: str) -> bool:
        if not self.text:
            return True  # cannot filter; let the server judge
        return C._key(flag) in {C._key(f) for f in self.flags} or flag in self.flags

    def choices(self, flag: str) -> list[str]:
        m = re.search(re.escape(flag) + r"\s+\{([^}]*)\}", self.text)
        return [x.strip() for x in m.group(1).split(",")] if m else []

    def filter(self, tokens):
        """Drop flags this build does not know. Returns (kept_tokens, dropped_flags)."""
        pairs, dropped, out = C.to_pairs(tokens), [], []
        for f, v in pairs:
            if self.has(f):
                out.append(f)
                out.extend(v)
            else:
                dropped.append(f)
        return out, dropped


def healthy(port, timeout=2) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout).status == 200
    except Exception:
        return False


def fetch_metrics(port, timeout=3) -> str:
    return urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout).read().decode()


def kill_tree(pid, waits=(25, 10, 5)):
    import psutil
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for sig, w in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), waits):
        for pr in procs:
            try:
                pr.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=w)
        procs = alive
        if not procs:
            return


def sweep_stragglers():
    import psutil
    for pr in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(pr.info["cmdline"] or [])
            if "VLLM::" in cl or "vllm serve" in cl or "vllm.entrypoints" in cl:
                pr.kill()
        except Exception:
            pass


def wait_vram_idle(baseline_mb, timeout=60):
    """Wait until VRAM falls back near baseline. Returns final used MB (None if no GPU)."""
    t0 = time.time()
    v = C.vram_used_mb()
    while v is not None and baseline_mb is not None and v > baseline_mb + 150 and time.time() - t0 < timeout:
        time.sleep(2)
        v = C.vram_used_mb()
    return v


class Server:
    def __init__(self, out, tokens, env=None, model=None, port=None):
        self.out, self.tokens, self.env = out, list(tokens), env or {}
        self.model, self.port = model or C.MODEL, port or C.PORT
        self.proc = None
        self.pid = None

    def start(self, timeout=900):
        cmd = ["vllm", "serve", self.model, *self.tokens, "--served-model-name", C.SERVED,
               "--host", "127.0.0.1", "--port", str(self.port), "--api-key", C.KEY]
        (self.out / "cmd.txt").write_text(" ".join(cmd) + "\n" + repr(self.env) + "\n")
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.env.items()})
        log = open(self.out / "server.log", "wb")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.pid = self.proc.pid
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False, time.time() - t0
            if healthy(self.port):
                return True, time.time() - t0
            time.sleep(2)
        return False, time.time() - t0

    def stop(self):
        if self.pid:
            kill_tree(self.pid)
        sweep_stragglers()
        self.proc = None


# ---------- log parsing ----------
FACT_PAT = re.compile(
    r"kv cache|maximum concurrency|prefix|chunked|cuda graph|cudagraph|enforce|backend|quantization|dtype|"
    r"max_model_len|model loading took|memory|compil|marlin|fp8|nvfp4|bitsandbytes|speculative|offload",
    re.I)


def extract_facts(log_path, out_path):
    try:
        lines = [l for l in open(log_path, errors="replace") if FACT_PAT.search(l)]
    except FileNotFoundError:
        lines = []
    open(out_path, "w").writelines(lines)
    return parse_facts("".join(lines))


def parse_facts(text: str) -> dict:
    d = {}
    m = re.search(r"Model loading took ([\d.]+)\s*GiB", text)
    d["weights_gib"] = float(m.group(1)) if m else None
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    d["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
    d["kv_gib"] = float(m.group(1)) if m else None
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", text)
    d["max_conc"] = float(m.group(2)) if m else None
    d["graphs"] = bool(re.search(r"graph capturing finished|Capturing CUDA graph", text, re.I))
    for k in ("marlin", "bitsandbytes", "fp8", "nvfp4"):
        d["kernel_" + k] = bool(re.search(k, text, re.I))
    return d


def diagnose(log_tail: str) -> str:
    t = log_tail.lower()
    rules = [
        ("unrecognized arguments|invalid choice|no such option", "bad-flag: this build rejects a flag (record as finding)"),
        ("no available memory for the cache blocks|larger than the maximum number of tokens that can be stored",
         "kv-budget: weights+activations leave too little for KV (lower max-model-len/max-num-seqs or raise gmu)"),
        ("out of memory|cuda error: out of memory|cuda out of memory", "oom: weights, activation peak or graph capture exceeded VRAM"),
        ("not supported|unsupported|does not support", "unsupported: feature not available on this GPU/build"),
        ("no module named|modulenotfounderror", "missing-package: pip install the missing dependency"),
    ]
    for pat, msg in rules:
        if re.search(pat, t):
            return msg
    return "unknown: read the last lines of server.log"
