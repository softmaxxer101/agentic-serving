"""Background samplers written next to every run: gpu.csv, dmon.csv, metrics.csv, cpu.csv."""
from __future__ import annotations
import csv, json, re, shutil, subprocess, threading, time
from . import core as C
from .server import fetch_metrics

_LINE = re.compile(r"([^{ ]+)(\{.*\})? (\S+)$")


class Sidecars:
    def __init__(self, out, port):
        self.out, self.port = out, port
        self.stop_ev = threading.Event()
        self.threads, self.procs = [], []

    def start_gpu(self):
        if not shutil.which("nvidia-smi"):
            return
        gpu = open(self.out / "gpu.csv", "w", buffering=1)
        gpu.write("ts,util,mem_mb,power_w,sm_mhz,temp_c\n")
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits", "-l", "1"]
        p = subprocess.Popen(q, stdout=subprocess.PIPE, text=True)
        self.procs.append(p)

        def rd():
            for line in p.stdout:
                gpu.write(f"{time.time()},{line.strip().replace(' ', '')}\n")
        t = threading.Thread(target=rd, daemon=True)
        t.start()
        self.threads.append(t)
        try:  # dmon is not available everywhere (e.g. some WSL builds); best effort
            d = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
                                 stdout=open(self.out / "dmon.csv", "w"), stderr=subprocess.DEVNULL)
            self.procs.append(d)
        except Exception:
            pass

    def start_service(self, server_pid=None):
        for fn in (self._metrics, lambda: self._cpu(server_pid)):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self.threads.append(t)

    def _metrics(self):
        f = open(self.out / "metrics.csv", "w", buffering=1, newline="")
        w = csv.writer(f)
        w.writerow(["ts", "name", "labels", "value"])
        while not self.stop_ev.is_set():
            try:
                txt = fetch_metrics(self.port)
                ts = time.time()
                for l in txt.splitlines():
                    if l.startswith("vllm:") and "_bucket" not in l and "_created" not in l:
                        m = _LINE.match(l)
                        if m:
                            w.writerow([ts, m.group(1), m.group(2) or "", m.group(3)])
            except Exception:
                pass
            self.stop_ev.wait(1.0)

    def _cpu(self, server_pid):
        import psutil
        f = open(self.out / "cpu.csv", "w", buffering=1)
        f.write("ts,percpu_json,server_pct,client_pct\n")
        me = psutil.Process()
        me.cpu_percent(None)
        srv = None
        if server_pid:
            try:
                srv = psutil.Process(server_pid)
            except Exception:
                srv = None
        while not self.stop_ev.is_set():
            per = psutil.cpu_percent(interval=1.0, percpu=True)
            sp = 0.0
            if srv is not None:
                try:
                    sp = sum(p.cpu_percent(None) for p in [srv] + srv.children(recursive=True))
                except Exception:
                    pass
            f.write(f"{time.time()},{json.dumps(per)},{sp:.1f},{me.cpu_percent(None):.1f}\n")

    def stop(self):
        self.stop_ev.set()
        for p in self.procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1.2)
