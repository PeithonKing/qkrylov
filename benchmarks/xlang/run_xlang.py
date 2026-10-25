"""Run the same Heisenberg-ring Lanczos problem in Python, Julia and C++ and compare them.

    python benchmarks/xlang/run_xlang.py L ITERS [REPEAT] [--julia PATH] [--lib PATH] [--cpp-dir DIR]

Each cell runs in its own process (REPEAT times; the fastest solve is reported). Checks that the
energies and eigenvector checksums agree across languages and prints time and memory side by side.
"""
import os
import re
import subprocess
import sys
import threading
import time

L = int(sys.argv[1]); iters = int(sys.argv[2])
rest = sys.argv[3:]
repeat = int(rest[0]) if rest and rest[0].isdigit() else 3
opt = {a: rest[i + 1] for i, a in enumerate(rest) if a.startswith("--") and i + 1 < len(rest)}
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
julia = opt.get("--julia", os.path.expanduser("~/.juliaup/bin/julia"))
cpp_dir = opt.get("--cpp-dir", os.path.join(root, "build"))
py = opt.get("--python", sys.executable)
device = opt.get("--device", "cpu")
env = dict(os.environ, OMP_PROC_BIND=os.environ.get("OMP_PROC_BIND", "spread"),
           OMP_PLACES=os.environ.get("OMP_PLACES", "threads"))
# Julia calls the C API, which runs on the build's default execution space (the GPU in a CUDA build).
# For a CPU comparison on a GPU machine pass a CPU-only library with --lib-cpu.
lib = opt.get("--lib-cpu") if device == "cpu" and "--lib-cpu" in opt else opt.get("--lib")
if lib:
    env["QKRYLOV_LIB_PATH"] = lib


class poll_gpu:
    """Peak GPU memory used while a cell runs, above the idle baseline (nvidia-smi, 20 ms polling)."""
    def __init__(self):
        self.peak = self.base = self._now()
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    @staticmethod
    def _now():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.split()
            return int(out[0])
        except Exception:
            return 0

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self._now())
            time.sleep(0.02)

    def stop(self):
        self._run = False
        self._t.join()
        return max(0, self.peak - self.base)


def cell(lang, prec, mode):
    if lang == "python":
        cmd = [py, os.path.join(here, "bench.py"), str(L), str(prec), mode, str(iters), device]
    elif lang == "julia":
        cmd = [julia, f"--project={os.path.join(root, 'bindings', 'julia')}", os.path.join(here, "bench.jl"),
               str(L), str(prec), mode, str(iters), device]
    else:
        cmd = [os.path.join(cpp_dir, f"xlang_fp{prec}"), str(L), mode, str(iters), device]
    best = None
    for _ in range(repeat):
        gpu = poll_gpu() if device != "cpu" else None
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("lang=")), None)
        gpu_mb = gpu.stop() if gpu else 0
        if line is None:
            return {"error": (out.stderr or out.stdout)[-400:]}
        r = dict(kv.split("=", 1) for kv in line.split())
        r["gpu_mb"] = str(gpu_mb)
        if best is None or float(r["t_solve_ms"]) < float(best["t_solve_ms"]):
            best = r
    return best


print(f"Heisenberg ring, Sz=0, L={L}, exactly {iters} Lanczos steps, device={device}, "
      f"best of {repeat}, OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'default')}\n")
ok = True
for prec in (64, 32):
    for mode in ("energy", "state"):
        rows = {lang: cell(lang, prec, mode) for lang in ("cpp", "python", "julia")}
        print(f"--- {prec}-bit, {mode} ---")
        print(f"{'lang':7} {'E0':>22} {'iters':>5} {'setup ms':>10} {'solve ms':>10} {'RSS base MB':>12} {'RSS peak MB':>12} {'problem MB':>11} {'GPU MB':>8} {'checksum':>16}")
        for lang, r in rows.items():
            if "error" in r:
                print(f"{lang:7} ERROR: {r['error']!r}"); ok = False; continue
            extra = float(r["rss_peak_mb"]) - float(r["rss_base_mb"])
            print(f"{lang:7} {r['E0']:>22} {r['iters']:>5} {r['t_setup_ms']:>10} {r['t_solve_ms']:>10} "
                  f"{r['rss_base_mb']:>12} {r['rss_peak_mb']:>12} {extra:>11.0f} {r['gpu_mb']:>8} {r['checksum']:>16}")
        good = [r for r in rows.values() if "error" not in r]
        if len(good) == 3:
            e = [float(r["E0"]) for r in good]
            cs = [float(r["checksum"]) for r in good]
            tol = 1e-9 if prec == 64 else 1e-4
            same_e = max(e) - min(e) <= tol * max(1.0, abs(e[0]))
            same_i = len({r["iters"] for r in good}) == 1
            same_c = mode == "energy" or (max(cs) - min(cs)) <= (1e-6 if prec == 64 else 1e-1) * max(1.0, abs(cs[0]))
            print(f"identical energy: {same_e}   identical iteration count: {same_i}   identical eigenvector checksum: {same_c}\n")
            ok = ok and same_e and same_i and same_c
sys.exit(0 if ok else 1)
