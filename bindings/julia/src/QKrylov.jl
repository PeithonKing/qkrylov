module QKrylov

using qkrylov_jll

const VERSION = v"0.1.0"

include("libqkrylov.jl")
include("sector.jl")
include("site.jl")
include("basis.jl")
include("opsum.jl")
include("hamiltonian.jl")
include("solvers.jl")

export Sector, set_sz!, set_hubbard_particles!
export AbstractSite, SpinHalfSite, FermionSite, HubbardSite, TJSite
export AbstractBasis, SpinHalfBasis, FermionBasis, HubbardBasis, TJBasis, dimension, nsites
export OpSum, add_term!, clear!
export MatrixFreeHamiltonian
export lanczos_ground_state, LanczosResult

end
