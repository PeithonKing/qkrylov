# Lanczos solver wrappers

struct LanczosResultC
    energy::Cfloat
end

struct LanczosResult
    energy::Float64
    _state::Union{Vector{ComplexF64}, Nothing}
    has_state::Bool

    function LanczosResult(energy::Float64, state::Union{Vector{ComplexF64}, Nothing}=nothing)
        return new(energy, state, state !== nothing)
    end
end

function Base.getproperty(res::LanczosResult, sym::Symbol)
    if sym === :state || sym === :eigenvector || sym === :vector
        if !getfield(res, :has_state) || getfield(res, :_state) === nothing
            error("Ground state wavefunction was not computed. Pass `return_state=true` to `lanczos_ground_state` to compute the state vector.")
        end
        return getfield(res, :_state)
    end
    return getfield(res, sym)
end

function Base.propertynames(res::LanczosResult, private::Bool=false)
    return private ? fieldnames(LanczosResult) : (:energy, :state, :eigenvector)
end

function Base.iterate(res::LanczosResult, state=1)
    if state == 1
        return (res.energy, 2)
    elseif state == 2
        if !res.has_state
            error("Ground state wavefunction was not computed. Pass `return_state=true` to `lanczos_ground_state` to compute the state vector.")
        end
        return (res.state, 3)
    else
        return nothing
    end
end

function lanczos_ground_state(
    H::MatrixFreeHamiltonian;
    maxiter::Integer=100,
    tol::Real=1e-12,
    return_state::Bool=false,
    compute_eigenvector::Bool=return_state
)::LanczosResult
    should_compute = return_state || compute_eigenvector
    res_c = Ref{LanczosResultC}(LanczosResultC(0.0f0))

    if should_compute
        dim = Int(dimension(H))
        vec_buf = Vector{Float32}(undef, 2 * dim)

        GC.@preserve vec_buf begin
            status = ccall(
                (:qkrylov_lanczos_ground_state_complex, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultC}, Ptr{Cfloat}),
                H.ptr, Cint(maxiter), Cfloat(tol), res_c, pointer(vec_buf)
            )
        end
        status != QKRYLOV_SUCCESS && error("Lanczos ground state solver failed with status code $status")

        psi = Vector{ComplexF64}(undef, dim)
        @inbounds for i in 1:dim
            re = Float64(vec_buf[2i - 1])
            im = Float64(vec_buf[2i])
            psi[i] = ComplexF64(re, im)
        end
        return LanczosResult(Float64(res_c[].energy), psi)
    else
        status = ccall(
            (:qkrylov_lanczos_ground_state, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultC}),
            H.ptr, Cint(maxiter), Cfloat(tol), res_c
        )
        status != QKRYLOV_SUCCESS && error("Lanczos ground state solver failed with status code $status")
        return LanczosResult(Float64(res_c[].energy), nothing)
    end
end

# Davidson Solver
struct DavidsonResult
    eigenvalues::Vector{Float64}
    eigenvectors::Union{Vector{Vector{ComplexF64}}, Nothing}
end

function davidson_lowest(
    H::MatrixFreeHamiltonian;
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-6,
    compute_eigenvectors::Bool=true
)::DavidsonResult
    dim = Int(dimension(H))
    evals_buf = Vector{Float32}(undef, n_eig)
    evecs_buf = compute_eigenvectors ? Vector{Float32}(undef, n_eig * 2 * dim) : Float32[]

    GC.@preserve evals_buf evecs_buf begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_buf) : Ptr{Cfloat}(C_NULL)
        status = ccall(
            (:qkrylov_davidson_lowest_complex, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}),
            H.ptr, Cint(n_eig), Cint(max_subspace), Cfloat(tol), pointer(evals_buf), evecs_ptr
        )
        status != QKRYLOV_SUCCESS && error("Davidson solver failed with status code $status")
    end

    evals = Vector{Float64}(evals_buf)
    if !compute_eigenvectors
        return DavidsonResult(evals, nothing)
    end

    evecs = Vector{Vector{ComplexF64}}(undef, n_eig)
    for idx in 1:n_eig
        v = Vector{ComplexF64}(undef, dim)
        offset = (idx - 1) * 2 * dim
        @inbounds for i in 1:dim
            re = Float64(evecs_buf[offset + 2i - 1])
            im = Float64(evecs_buf[offset + 2i])
            v[i] = ComplexF64(re, im)
        end
        evecs[idx] = v
    end
    return DavidsonResult(evals, evecs)
end

# Dynamics & Spectral Function
struct ContinuedFractionResult
    alphas::Vector{Float64}
    betas::Vector{Float64}
    norm_phi0::Float64
end

function continued_fraction_coeffs(
    H::MatrixFreeHamiltonian,
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult
    dim = Int(dimension(H))
    @assert length(phi0) == dim "Initial vector phi0 size $(length(phi0)) does not match Hamiltonian dimension $dim"

    phi0_c = Vector{ComplexF64}(phi0)
    phi0_buf = Vector{Float32}(undef, 2 * dim)
    @inbounds for i in 1:dim
        phi0_buf[2i - 1] = Float32(real(phi0_c[i]))
        phi0_buf[2i]     = Float32(imag(phi0_c[i]))
    end

    alphas_buf = Vector{Float32}(undef, n_iter)
    betas_buf  = Vector{Float32}(undef, n_iter)
    norm_ref   = Ref{Cfloat}(0.0f0)
    num_coeffs = Ref{Cint}(0)

    GC.@preserve phi0_buf alphas_buf betas_buf begin
        status = ccall(
            (:qkrylov_continued_fraction_coeffs_complex, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ref{Cfloat}, Ref{Cint}),
            H.ptr, pointer(phi0_buf), Cint(n_iter), pointer(alphas_buf), pointer(betas_buf), norm_ref, num_coeffs
        )
        status != QKRYLOV_SUCCESS && error("Continued fraction solver failed with status code $status")
    end

    k = Int(num_coeffs[])
    alphas = Vector{Float64}(alphas_buf[1:k])
    betas  = Vector{Float64}(betas_buf[1:max(0, k - 1)])
    return ContinuedFractionResult(alphas, betas, Float64(norm_ref[]))
end

function evaluate_spectral_function(
    cfr::ContinuedFractionResult,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    alphas_buf = Vector{Float32}(cfr.alphas)
    betas_buf  = Vector{Float32}(cfr.betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function, libqkrylov),
            Cfloat,
            (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat, Cfloat, Cfloat),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cfloat(cfr.norm_phi0), Cfloat(omega), Cfloat(E0), Cfloat(eta)
        )
    end
    return Float64(val)
end

function evaluate_spectral_function(
    alphas::AbstractVector{<:Real},
    betas::AbstractVector{<:Real},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    alphas_buf = Vector{Float32}(alphas)
    betas_buf  = Vector{Float32}(betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function, libqkrylov),
            Cfloat,
            (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat, Cfloat, Cfloat),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cfloat(norm_phi0), Cfloat(omega), Cfloat(E0), Cfloat(eta)
        )
    end
    return Float64(val)
end

# FTLM (Finite Temperature Lanczos) Solver
struct FTLMResultC
    beta::Cfloat
    partition_function::Cfloat
    internal_energy::Cfloat
    specific_heat::Cfloat
end

struct FTLMResult
    beta::Float64
    partition_function::Float64
    internal_energy::Float64
    specific_heat::Float64
end

function ftlm(
    H::MatrixFreeHamiltonian;
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult
    res_c = Ref{FTLMResultC}(FTLMResultC(0.0f0, 0.0f0, 0.0f0, 0.0f0))
    status = ccall(
        (:qkrylov_ftlm, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cfloat, Cint, Cint, Ref{FTLMResultC}),
        H.ptr, Cfloat(beta), Cint(n_random), Cint(n_steps), res_c
    )
    status != QKRYLOV_SUCCESS && error("FTLM solver failed with status code $status")
    return FTLMResult(
        Float64(res_c[].beta),
        Float64(res_c[].partition_function),
        Float64(res_c[].internal_energy),
        Float64(res_c[].specific_heat)
    )
end
