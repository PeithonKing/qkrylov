# Ground state of the periodic Heisenberg chain, Sz = 0 sector (Julia).
# The same problem is solved in examples/quickstart.py (Python) and examples/quickstart.cpp (C++).
#
#     julia --project=bindings/julia examples/quickstart.jl [L] [device]     e.g.  ... 20 cpu
#
# For a GPU build use device = "gpu" (or "cuda"); pass precision=Float32 to MatrixFreeHamiltonian for single precision.
using QuantumKrylov, Printf, LinearAlgebra

L = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 20
device = length(ARGS) > 1 ? ARGS[2] : "cpu"

function heisenberg_ring(L)
    ops = OpSum()
    for i in 0:(L - 1)
        j = mod(i + 1, L)
        ops += Sz(i) * Sz(j) + 0.5 * (Sp(i) * Sm(j) + Sm(i) * Sp(j))
    end
    return ops
end

basis = SpinHalfBasis(L; sz=0)            # describes the sector only: no states are built yet
ops = heisenberg_ring(L)
H = MatrixFreeHamiltonian(basis, SpinHalfSite(), ops; device=device)      # Float64 by default
sol = solve(GroundStateProblem(H), Lanczos(variation=TwoPass(), return_state=true))
energy, psi = sol.energy, sol.state

@printf("L = %d, device = %s, dimension = %s\n", L, device, string(Int(dimension(basis))))
@printf("ground-state energy  E0 = %.10f   (E0/L = %.8f)\n", energy, energy / L)
@printf("state number 5 as bits: %s   |psi| = %.6f\n", string(state(basis, 5); base=2, pad=L), norm(psi))
