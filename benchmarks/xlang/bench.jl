# One cell of the cross-language benchmark (Julia side). See bench.py for the contract.
#   julia --project=bindings/julia bench.jl L PREC MODE ITERS [DEVICE]
using QuantumKrylov, Printf

const L = parse(Int, ARGS[1])
const prec = parse(Int, ARGS[2])
const mode = ARGS[3]
const iters = parse(Int, ARGS[4])
const T = prec == 64 ? Float64 : Float32
const device = length(ARGS) > 4 ? ARGS[5] : "cpu"

function build_and_solve(L, T, mode, iters, device)
    t0 = time_ns()
    basis = SpinHalfBasis(L; sz=0)
    ops = OpSum()
    for i in 0:(L - 1)
        j = mod(i + 1, L)
        ops += 1.0 * Sz(i) * Sz(j) + 0.5 * (Sp(i) * Sm(j) + Sm(i) * Sp(j))
    end
    H = MatrixFreeHamiltonian(basis, SpinHalfSite(), ops; precision=T, device=device)
    t1 = time_ns()
    res = solve(GroundStateProblem(H), Lanczos(variation=TwoPass(), maxiter=iters, tol=1e-12,
                                               return_state=(mode == "state"), early_stopping=false))
    t2 = time_ns()
    return H, res, (t1 - t0) / 1e6, (t2 - t1) / 1e6
end

build_and_solve(8, T, mode, 8, device)          # warm-up: compile everything once, outside the timings
GC.gc()
rss_base = Sys.maxrss() / 2^20

H, res, t_setup, t_solve = build_and_solve(L, T, mode, iters, device)

# plain loop: a generator-based sum() allocates temporaries and would inflate the memory reading
function position_checksum(psi)
    acc = 0.0
    @inbounds for k in eachindex(psi)
        acc += Float64(real(psi[k])) * sin(k - 1)
    end
    return acc
end

checksum = res.has_state ? position_checksum(res.state) : 0.0
rss_peak = Sys.maxrss() / 2^20

println("lang=julia prec=$prec mode=$mode L=$L dim=$(Int(dimension(H))) E0=", @sprintf("%.12e", res.energy),
        " iters=$(res.iterations) t_setup_ms=", @sprintf("%.2f", t_setup), " t_solve_ms=", @sprintf("%.2f", t_solve),
        " rss_base_mb=", @sprintf("%.0f", rss_base), " rss_peak_mb=", @sprintf("%.0f", rss_peak),
        " checksum=", @sprintf("%.10e", checksum))
