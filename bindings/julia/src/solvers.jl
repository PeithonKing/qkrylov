# Lanczos solver wrappers

struct LanczosResultC
    energy::Cdouble
end

struct LanczosResult
    energy::Float64
end

function lanczos_ground_state(H::MatrixFreeHamiltonian; maxiter::Integer=100, tol::Real=1e-12)::LanczosResult
    res_c = Ref{LanczosResultC}(LanczosResultC(0.0))
    status = ccall(
        (:qkrylov_lanczos_ground_state, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultC}),
        H.ptr, Cint(maxiter), Cdouble(tol), res_c
    )
    status != QKRYLOV_SUCCESS && error("Lanczos ground state solver failed with status code $status")
    return LanczosResult(res_c[].energy)
end
