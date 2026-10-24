"""One cell of the cross-language benchmark (Python side).

    python bench.py L PREC MODE ITERS         e.g.  python bench.py 24 64 state 40

PREC is 32 or 64; MODE is "energy" (energy only) or "state" (two-pass, eigenvector returned).
Prints one key=value line that run_xlang.py parses. bench.jl and bench.cpp solve the same
problem with the same settings: Heisenberg ring, Sz=0, exactly ITERS Lanczos steps, default seed.
"""
import resource
import sys
import time

import numpy as np

import qkrylov as qk

L, prec, mode, iters = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
device = sys.argv[5] if len(sys.argv) > 5 else "cpu"
dtype = np.float64 if prec == 64 else np.float32


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_and_solve(L):
    t0 = time.perf_counter()
    basis = qk.basis.SpinHalf(L, sz=0)
    ops = qk.OpSum()
    for i in range(L):
        j = (i + 1) % L
        ops += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))
    H = qk.MatrixFreeHamiltonian(basis, qk.site.SpinHalf(), ops, device=device, dtype=dtype)
    t1 = time.perf_counter()
    res = qk.solvers.LanczosTwoPass(maxiter=iters if L == main_L else 8, tol=1e-12, early_stopping=False,
                                    return_state=(mode == "state")).solve(H)
    t2 = time.perf_counter()
    return H, res, (t1 - t0) * 1e3, (t2 - t1) * 1e3


main_L = L
build_and_solve(8)                 # warm-up (runtime start-up, Kokkos / CUDA context): outside the timings,
rss_base = rss_mb()                # as in bench.jl and bench.cpp
H, res, t_setup, t_solve = build_and_solve(L)

checksum = 0.0
if res.eigenvector is not None:
    psi = np.asarray(res.eigenvector)
    for lo in range(0, psi.size, 1 << 20):      # chunked: no full-size temporaries inflate the memory reading
        part = psi[lo:lo + (1 << 20)].real.astype(np.float64)
        checksum += float(np.sum(part * np.sin(np.arange(lo, lo + part.size, dtype=np.float64))))

print(f"lang=python prec={prec} mode={mode} L={L} dim={H.dimension} E0={res.energy:.12e} iters={res.iterations} "
      f"t_setup_ms={t_setup:.2f} t_solve_ms={t_solve:.2f} "
      f"rss_base_mb={rss_base:.0f} rss_peak_mb={rss_mb():.0f} checksum={checksum:.10e}")
