"""Ground state of the periodic Heisenberg chain, Sz = 0 sector (Python).

The same problem is solved in examples/quickstart.jl (Julia) and examples/quickstart.cpp (C++).

    python examples/quickstart.py [L] [device]       e.g.  python examples/quickstart.py 20 cpu

For a GPU build use device="cuda"; add dtype=np.float32 to MatrixFreeHamiltonian for single precision.
"""
import sys

import qkrylov as qk

L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

basis = qk.basis.SpinHalf(L, sz=0)          # describes the sector only: no states are built yet
ops = qk.OpSum()
for i in range(L):
    j = (i + 1) % L
    ops += qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))

H = qk.MatrixFreeHamiltonian(basis, qk.site.SpinHalf(), ops, device=device)    # float64 by default
energy, psi = qk.solvers.LanczosTwoPass().solve(H)

print(f"L = {L}, device = {device}, dimension = {basis.size:,}")
print(f"ground-state energy  E0 = {energy:.10f}   (E0/L = {energy / L:.8f})")
print(f"state number 5 as bits: {basis.state(5):0{L}b}   |psi| = {(psi @ psi) ** 0.5:.6f}")
