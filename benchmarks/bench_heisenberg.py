#!/usr/bin/env python3
"""Phase-timed benchmark for the periodic Heisenberg chain (Sz=0 sector).

Mirrors the format of the reference log /extra/alterego_test/20.log:
    1_basis, 2_hamiltonian, 3_solver_setup, 4_execution (mean/std over --repeat runs
    after one warm-up), and additionally reports the energy, iteration count and
    time per iteration so runs with different iteration counts compare fairly.

Usage:
    python benchmarks/bench_heisenberg.py -L 20 --precision 64 --passes 1 --device cpu
    python benchmarks/bench_heisenberg.py -L 20 --all          # 32/64-bit x 1/2-pass
"""
import argparse
import os
import platform
import time

import numpy as np

import qkrylov as qk


def build(L, dtype, device):
    t0 = time.perf_counter()
    basis = qk.basis.SpinHalf(L, sz=0)
    t1 = time.perf_counter()
    ops = qk.OpSum()
    for i in range(L):
        j = (i + 1) % L
        ops += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))
    H = qk.MatrixFreeHamiltonian(basis, qk.site.SpinHalf(), ops, device=device, dtype=dtype)
    t2 = time.perf_counter()
    return H, (t1 - t0), (t2 - t1)


def run_once(L, dtype, passes, device, maxiter, tol, early_stopping=True):
    H, t_basis, t_ham = build(L, dtype, device)
    t0 = time.perf_counter()
    solver_cls = qk.solvers.Lanczos if passes == 1 else qk.solvers.LanczosTwoPass
    solver = solver_cls(maxiter=maxiter, tol=tol, early_stopping=early_stopping)
    t1 = time.perf_counter()
    res = solver.solve(H)
    t2 = time.perf_counter()
    return {
        "1_basis": t_basis, "2_hamiltonian": t_ham, "3_solver_setup": t1 - t0, "4_execution": t2 - t1,
        "energy": res.energy, "iterations": res.iterations, "dim": H.dimension,
    }


def bench(L, precision, passes, device, repeat, maxiter, tol, early_stopping=True):
    dtype = np.float64 if precision == 64 else np.float32
    print(f"--- QKrylov Benchmark ---")
    print(f"L={L} | {precision}-bit | {passes}-pass | {device.upper()}")
    print("Warming up...")
    run_once(L, dtype, passes, device, maxiter, tol, early_stopping)
    runs = []
    for r in range(repeat):
        print(f"Run {r + 1}/{repeat}...", end="", flush=True)
        runs.append(run_once(L, dtype, passes, device, maxiter, tol, early_stopping))
    print("\n")
    print(f"{'Step':<20} | {'Mean (ms)':<15} | {'Std (ms)':<15}")
    print("-" * 55)
    for step in ("1_basis", "2_hamiltonian", "3_solver_setup", "4_execution"):
        v = np.array([r[step] for r in runs]) * 1e3
        print(f"{step:<20} | {v.mean():<15.3f} | {v.std():<15.3f}")
    it = runs[-1]["iterations"]
    ex = np.mean([r["4_execution"] for r in runs]) * 1e3
    print(f"{'energy':<20} | {runs[-1]['energy']:.10f}")
    print(f"{'iterations':<20} | {it}")
    print(f"{'ms / iteration':<20} | {ex / max(it, 1):.3f}")
    print(f"{'dimension':<20} | {runs[-1]['dim']:,}")
    print("-" * 48)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("-L", type=int, default=20)
ap.add_argument("--precision", type=int, choices=[32, 64], default=64)
ap.add_argument("--passes", type=int, choices=[1, 2], default=1)
ap.add_argument("--device", default="cpu")
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--maxiter", type=int, default=100)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--all", action="store_true", help="run 32/64-bit x 1/2-pass")
ap.add_argument("--no-early-stopping", action="store_true",
                help="run exactly --maxiter iterations (fixed work: fair across precisions/versions)")
a = ap.parse_args()

print(f"--- Benchmarking L={a.L} on {platform.machine()} ({os.cpu_count()} cores), "
      f"OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND', 'unset')} "
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} ---\n")
configs = [(p, n) for p in (32, 64) for n in (1, 2)] if a.all else [(a.precision, a.passes)]
for precision, passes in configs:
    print(f"Running: --precision {precision} --passes {passes} --device {a.device} -L {a.L}")
    bench(a.L, precision, passes, a.device, a.repeat, a.maxiter, a.tol, not a.no_early_stopping)
