"""
    QuantumKrylov

Native high-performance Julia interface to the `qkrylov` C++20 matrix-free Krylov subspace methods and quantum dynamics library.

# Features
- Zero-copy native `ccall` interface into Kokkos-accelerated C++ kernels.
- Dual-precision numerical execution (`Float64` and `Float32`).
- Hardware device traits for OpenMP CPU, NVIDIA CUDA, AMD HIP, and Intel SYCL.
- Full integration with the Julia SciML Common Interface (`solve(prob, alg)`).

# Notes
<TODO-LLM: Explain the Julia module architecture, native ccall binding layer, and SciML common interface integration here>
"""
module QuantumKrylov

using qkrylov_jll

const VERSION = v"0.1.0"

include("libqkrylov.jl")
include("device.jl")
include("sector.jl")
include("site.jl")
include("basis.jl")
include("opsum.jl")
include("hamiltonian.jl")
include("solvers.jl")
include("vector_ops.jl")

export Sector, set_sz!, set_hubbard_particles!, set_n!, set_nb!, get_sz, get_hubbard_particles, get_n, get_nb
export AbstractSite, SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite, LocalAction, apply, site_type
export AbstractBasis, SpinHalfBasis, SpinSBasis, FermionBasis, HubbardBasis, TJBasis, dimension, nsites, state, basis_index, basis_type, sector
export spin, dimension_per_site
export OpSum, add_term!, clear!, OpTerm, OpExpr, opsum_size, opsum_get_term_info, opsum_get_factor
export Sz, Sp, Sm, Sx, Sy, n, c, cdag
export CdagUp, CUp, CdagDn, CDn, Nup, Ndn, Nupdn, Bdag, B, N
export validate, validate!
export MatrixFreeHamiltonian, diagonal, diagonal_device
export DeviceVector, mul!
export solve
export AbstractQuantumProblem, GroundStateProblem, ExcitedStatesProblem, ThermalProblem, DynamicsProblem, SpectralProblem
export AbstractQuantumAlgorithm, AbstractLanczosVariation, OnePass, TwoPass, OnePass_DKGS
export Lanczos, Davidson, FTLM, ContinuedFraction, CorrectionVector
export AbstractQuantumSolution, GroundStateSolution, LanczosResult, ExcitedStatesSolution
export lanczos_ground_state, lanczos_lowest, LanczosLowestResult
export davidson_lowest, DavidsonResult
export continued_fraction_coeffs, ContinuedFractionResult, evaluate_spectral_function
export ftlm, FTLMResult, ftlm_sweep, ftlm_sweep_streamed, FTLMSweepResult, FTLMSamples, ftlm_sample, ftlm_evaluate_sweep
export RealTimeResult, time_evolve, FTLMDynamicsResult, ftlm_dynamics
export solver_correction_vector, CorrectionVectorResult
export vector_dot, vector_norm, vector_axpy!, vector_scal!, vector_normalize!, vector_zero_fill!, vector_copy!
export AbstractDevice, CPUDevice, CUDADevice, HIPDevice, SYCLDevice
export find_gpu, gpu_count, is_gpu_build, initialize_device!
export get_last_error_message, clear_last_error

end
