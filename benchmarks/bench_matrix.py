#!/usr/bin/env python3
"""Full benchmark matrix for the Lanczos ground-state solvers, written as JSON for the viewer.

One invocation covers one device and one stopping mode, and sweeps every system size, precision and
number of passes (1 = Lanczos, 2 = LanczosTwoPass) on a periodic Heisenberg chain (Sz = 0 sector).

    python benchmarks/bench_matrix.py --device cpu  --mode fixed --sizes 16 18 20 22 24 --out cpu_fixed.json
    python benchmarks/bench_matrix.py --device cuda --mode early --sizes 16 18 20 22 24 --out gpu_early.json

Modes
    fixed  no early stopping, exactly --maxiter (100) iterations: identical work everywhere, so times compare fairly
    early  stops as soon as the energy has converged (tolerance per precision), at most --early-maxiter iterations

Merge the outputs into the viewer's data file with  benchmarks/viewer/build_data.py.
"""
import argparse
import gc
import json
import os
import platform
import time

import numpy as np

import qkrylov as qk

# Convergence tolerance used in "early" mode. float32 cannot reach the float64 value.
EARLY_TOL = {32: 1e-5, 64: 1e-10}
FIXED_TOL = 1e-12  # never reached before maxiter in practice; early stopping is off anyway
TIME_BUDGET_S = 90.0  # once a single run takes longer than this, do not repeat it


def build(L, dtype, device):
    t0 = time.perf_counter()
    basis = qk.basis.SpinHalf(L, sz=0)
    ops = qk.OpSum()
    for i in range(L):
        j = (i + 1) % L
        ops += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))
    H = qk.MatrixFreeHamiltonian(basis, qk.site.SpinHalf(), ops, device=device, dtype=dtype)
    return H, time.perf_counter() - t0


def solve_once(L, precision, passes, device, mode, maxiter):
    dtype = np.float64 if precision == 64 else np.float32
    H, t_build = build(L, dtype, device)
    cls = qk.solvers.Lanczos if passes == 1 else qk.solvers.LanczosTwoPass
    early = mode == "early"
    solver = cls(maxiter=maxiter, tol=EARLY_TOL[precision] if early else FIXED_TOL, early_stopping=early)
    t0 = time.perf_counter()
    res = solver.solve(H)
    t_solve = time.perf_counter() - t0
    out = {"solve_s": t_solve, "build_s": t_build, "energy": float(res.energy),
           "iterations": int(res.iterations), "dim": int(H.dimension)}
    del res, H, solver
    gc.collect()  # release device memory now, or repeated runs at large L exhaust the GPU
    return out


def measure(L, precision, passes, device, mode, maxiter, repeat):
    runs = [solve_once(L, precision, passes, device, mode, maxiter)]
    while len(runs) < repeat and runs[0]["solve_s"] < TIME_BUDGET_S:
        runs.append(solve_once(L, precision, passes, device, mode, maxiter))
    t = np.array([r["solve_s"] for r in runs])
    last = runs[-1]
    return {
        "L": L, "precision": precision, "passes": passes, "device": "gpu" if device != "cpu" else "cpu",
        "mode": mode, "dim": last["dim"], "iterations": last["iterations"], "energy": last["energy"],
        "solve_s_mean": float(t.mean()), "solve_s_min": float(t.min()), "solve_s_std": float(t.std()),
        "build_s": float(np.mean([r["build_s"] for r in runs])), "runs": len(runs),
    }


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--device", default="cpu", help='"cpu" (OpenMP) or "cuda"')
ap.add_argument("--mode", choices=["fixed", "early"], required=True)
ap.add_argument("--sizes", type=int, nargs="+", required=True)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=100, help="iterations in fixed mode")
ap.add_argument("--early-maxiter", type=int, default=500, help="iteration cap in early mode")
ap.add_argument("--machine", default=platform.node())
ap.add_argument("--out", required=True)
a = ap.parse_args()

meta = {"machine": a.machine, "arch": platform.machine(), "cores": os.cpu_count(), "device": a.device,
        "mode": a.mode, "maxiter": a.maxiter if a.mode == "fixed" else a.early_maxiter,
        "tol": FIXED_TOL if a.mode == "fixed" else EARLY_TOL, "qkrylov": getattr(qk, "__version__", "?"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"), "date": time.strftime("%Y-%m-%d")}
results = []
maxiter = a.maxiter if a.mode == "fixed" else a.early_maxiter

# One small run per configuration first so kernel compilation / thread start-up is not timed.
for precision in (32, 64):
    for passes in (1, 2):
        solve_once(12, precision, passes, a.device, a.mode, 20)

for L in a.sizes:
    for precision in (32, 64):
        for passes in (1, 2):
            try:
                r = measure(L, precision, passes, a.device, a.mode, maxiter, a.repeat)
            except RuntimeError as e:
                # Single-pass Lanczos keeps every Krylov vector: 100 fp64 vectors at L=28 do not fit in 24 GB.
                if "allocate" not in str(e):
                    raise
                gc.collect()
                r = {"L": L, "precision": precision, "passes": passes, "device": "gpu" if a.device != "cpu" else "cpu",
                     "mode": a.mode, "error": "out of memory"}
                print(f"L={L} fp{precision} {passes}-pass {r['device']} {a.mode}: out of memory", flush=True)
                results.append(r)
                with open(a.out, "w") as f:
                    json.dump({"meta": meta, "results": results}, f, indent=1)
                continue
            results.append(r)
            print(f"L={L} fp{precision} {passes}-pass {r['device']} {a.mode}: {r['solve_s_mean']:.3f} s "
                  f"({r['iterations']} it, E={r['energy']:.8f}, runs={r['runs']})", flush=True)
            with open(a.out, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=1)
