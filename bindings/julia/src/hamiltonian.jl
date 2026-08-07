# MatrixFreeHamiltonian & operator overloading

mutable struct MatrixFreeHamiltonian
    ptr::Ptr{Cvoid}
    basis::AbstractBasis
    site::AbstractSite
    opsum::OpSum

    function MatrixFreeHamiltonian(basis::AbstractBasis, site::AbstractSite, opsum::OpSum)
        ptr = ccall(
            (:qkrylov_hamiltonian_create, libqkrylov),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            basis.ptr, site.ptr, opsum.ptr
        )
        ptr == C_NULL && error("Failed to create MatrixFreeHamiltonian")
        
        obj = new(ptr, basis, site, opsum)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_hamiltonian_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function dimension(H::MatrixFreeHamiltonian)::UInt64
    return ccall((:qkrylov_hamiltonian_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), H.ptr)
end

Base.size(H::MatrixFreeHamiltonian) = (Int(dimension(H)), Int(dimension(H)))
Base.size(H::MatrixFreeHamiltonian, d::Integer) = (d == 1 || d == 2) ? Int(dimension(H)) : 1

function Base.:*(H::MatrixFreeHamiltonian, x::AbstractVector{<:Number})::Vector{ComplexF64}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"
    
    x_c = Vector{ComplexF64}(x)
    x_re = Vector{Float64}(real.(x_c))
    x_im = Vector{Float64}(imag.(x_c))
    y_re = Vector{Float64}(undef, dim)
    y_im = Vector{Float64}(undef, dim)

    GC.@preserve x_re x_im y_re y_im begin
        status = ccall(
            (:qkrylov_hamiltonian_apply, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
            H.ptr, pointer(x_re), pointer(x_im), pointer(y_re), pointer(y_im)
        )
        status != QKRYLOV_SUCCESS && error("MatrixFreeHamiltonian apply failed with status code $status")
    end

    return complex.(y_re, y_im)
end
