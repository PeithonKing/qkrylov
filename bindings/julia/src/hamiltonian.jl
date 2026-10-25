# MatrixFreeHamiltonian & operator overloading

"""
    MatrixFreeHamiltonian{T<:Union{Float32, Float64}}

Matrix-free quantum Hamiltonian linear operator.
Evaluates matrix-vector products \$H |v\\rangle\$ on-the-fly without allocating sparse or dense matrices in RAM.
Supports dual-precision floating-point arithmetic (`Float64` and `Float32`) across CPU and GPU hardware.

# Constructors
- `MatrixFreeHamiltonian{T}(basis, site, opsum; device=CPUDevice())`
- `MatrixFreeHamiltonian(basis, site, opsum; device=CPUDevice(), dtype=Float64)`
- `MatrixFreeHamiltonian(basis, opsum; device=CPUDevice(), dtype=Float64)`

`dtype` is `Float64` or `Float32` and defaults to `Float64` on every device; `H.dtype` returns the element type.

"""
mutable struct MatrixFreeHamiltonian{T<:Union{Float32, Float64}}
    ptr::Ptr{Cvoid}
    basis::AbstractBasis
    site::AbstractSite
    opsum::OpSum
    device::AbstractDevice
    dtype::Type{T}

    function MatrixFreeHamiltonian{T}(
        basis::AbstractBasis,
        site::AbstractSite,
        opsum::OpSum;
        device::Union{AbstractDevice, AbstractString} = CPUDevice()
    ) where {T<:Union{Float32, Float64}}
        validate!(opsum, nsites(basis))

        dev_str = device_string(device)
        d_lower = lowercase(dev_str)
        requested = occursin("cuda", d_lower) ? "cuda" :
                    occursin("hip", d_lower)  ? "hip"  :
                    occursin("sycl", d_lower) ? "sycl" :
                    d_lower == "gpu"          ? "gpu"  : nothing
        if requested !== nothing
            if !is_gpu_build()
                throw(ArgumentError("QKrylov was not built with GPU support. Install/compile a GPU build of libqkrylov with Kokkos CUDA/HIP enabled."))
            end
            # A GPU build serves exactly one backend: asking for another one is an error, not a silent substitution.
            backend = lowercase(something(find_gpu(), ""))
            if requested != "gpu" && !isempty(backend) && !occursin(requested, backend)
                throw(ArgumentError("This libqkrylov was built for the '$backend' backend; the '$requested' device is not available. Use device=\"gpu\" or device=\"$backend\"."))
            end
        end

        dev_trait = if device isa AbstractDevice
            device
        else
            if occursin("cuda", d_lower) || d_lower == "gpu"
                CUDADevice()
            elseif occursin("hip", d_lower)
                HIPDevice()
            elseif occursin("sycl", d_lower)
                SYCLDevice()
            else
                CPUDevice()
            end
        end

        ptr = if T === Float32
            ccall(
                (:qkrylov_hamiltonian_create_device_fp32, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                basis.ptr, site.ptr, opsum.ptr, dev_str
            )
        else
            ccall(
                (:qkrylov_hamiltonian_create_device_fp64, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                basis.ptr, site.ptr, opsum.ptr, dev_str
            )
        end
        if ptr == C_NULL
            err = get_last_error_message()
            if !isempty(err)
                error("Failed to create MatrixFreeHamiltonian on device: $dev_str with dtype: $T: $err")
            else
                error("Failed to create MatrixFreeHamiltonian on device: $dev_str with dtype: $T")
            end
        end

        obj = new{T}(ptr, basis, site, opsum, dev_trait, T)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_hamiltonian_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function MatrixFreeHamiltonian(
    basis::AbstractBasis,
    site::AbstractSite,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice(),
    dtype = Float64
)
    return MatrixFreeHamiltonian{check_dtype(dtype)}(basis, site, opsum; device=device)
end

default_site(b::SpinHalfBasis) = SpinHalfSite()
default_site(b::SpinSBasis)    = SpinSSite(b.spin_s)
default_site(b::FermionBasis)  = FermionSite()
default_site(b::HubbardBasis)  = HubbardSite()
default_site(b::TJBasis)       = TJSite()
function default_site(b::AbstractBasis)
    error("Cannot automatically infer Site model for basis type $(typeof(b)). Please provide the site argument explicitly: MatrixFreeHamiltonian(basis, site, opsum)")
end

function MatrixFreeHamiltonian(
    basis::AbstractBasis,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice(),
    dtype = Float64
)
    site = default_site(basis)
    return MatrixFreeHamiltonian{check_dtype(dtype)}(basis, site, opsum; device=device)
end

function MatrixFreeHamiltonian{T}(
    basis::AbstractBasis,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice()
) where {T<:Union{Float32, Float64}}
    site = default_site(basis)
    return MatrixFreeHamiltonian{T}(basis, site, opsum; device=device)
end

function dimension(H::MatrixFreeHamiltonian)::UInt64
    return ccall((:qkrylov_hamiltonian_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), H.ptr)
end

Base.size(H::MatrixFreeHamiltonian) = (Int(dimension(H)), Int(dimension(H)))
Base.size(H::MatrixFreeHamiltonian, d::Integer) = (d == 1 || d == 2) ? Int(dimension(H)) : 1

function Base.:*(H::MatrixFreeHamiltonian{Float64}, x::AbstractVector{<:Number})::Vector{ComplexF64}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"

    x_c = (x isa Vector{ComplexF64}) ? x : Vector{ComplexF64}(x)
    y_c = Vector{ComplexF64}(undef, dim)

    GC.@preserve x_c y_c begin
        status = ccall(
            (:qkrylov_hamiltonian_apply_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
            H.ptr, pointer(x_c), pointer(y_c)
        )
        _check_status(status, "MatrixFreeHamiltonian apply failed")
    end
    return y_c
end

function Base.:*(H::MatrixFreeHamiltonian{Float32}, x::AbstractVector{<:Number})::Vector{ComplexF32}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"

    x_c = (x isa Vector{ComplexF32}) ? x : Vector{ComplexF32}(x)
    y_c = Vector{ComplexF32}(undef, dim)

    GC.@preserve x_c y_c begin
        status = ccall(
            (:qkrylov_hamiltonian_apply_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}),
            H.ptr, pointer(x_c), pointer(y_c)
        )
        _check_status(status, "MatrixFreeHamiltonian apply failed")
    end
    return y_c
end

function diagonal(H::MatrixFreeHamiltonian{Float64})::Vector{Float64}
    dim = Int(dimension(H))
    diag_buf = Vector{Float64}(undef, dim)

    GC.@preserve diag_buf begin
        status = ccall(
            (:qkrylov_hamiltonian_diagonal_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}),
            H.ptr, pointer(diag_buf)
        )
        _check_status(status, "Failed to extract Hamiltonian diagonal")
    end
    return diag_buf
end

function diagonal(H::MatrixFreeHamiltonian{Float32})::Vector{Float32}
    dim = Int(dimension(H))
    diag_buf = Vector{Float32}(undef, dim)

    GC.@preserve diag_buf begin
        status = ccall(
            (:qkrylov_hamiltonian_diagonal_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}),
            H.ptr, pointer(diag_buf)
        )
        _check_status(status, "Failed to extract Hamiltonian diagonal")
    end
    return diag_buf
end

# Pure Device Matrix-Vector Multiplication: y_dev = H * x_dev (zero copies)
function mul!(y::DeviceVector{Float64}, H::MatrixFreeHamiltonian{Float64}, x::DeviceVector{Float64})
    dim = Int(dimension(H))
    (length(x) == dim && length(y) == dim) || throw(DimensionMismatch("Vector dimensions ($(length(x)), $(length(y))) do not match Hamiltonian dimension $dim"))
    status = ccall((:qkrylov_hamiltonian_apply_device_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, x.ptr, y.ptr)
    _check_status(status, "qkrylov_hamiltonian_apply_device_fp64 failed")
    return y
end

function mul!(y::DeviceVector{Float32}, H::MatrixFreeHamiltonian{Float32}, x::DeviceVector{Float32})
    dim = Int(dimension(H))
    (length(x) == dim && length(y) == dim) || throw(DimensionMismatch("Vector dimensions ($(length(x)), $(length(y))) do not match Hamiltonian dimension $dim"))
    status = ccall((:qkrylov_hamiltonian_apply_device_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, x.ptr, y.ptr)
    _check_status(status, "qkrylov_hamiltonian_apply_device_fp32 failed")
    return y
end

function Base.:*(H::MatrixFreeHamiltonian{T}, x::DeviceVector{T})::DeviceVector{T} where {T}
    y = DeviceVector{T}(length(x))
    mul!(y, H, x)
    return y
end

"""
    diagonal_device(H::MatrixFreeHamiltonian{T}) -> DeviceVector{T}

Extract the diagonal of the Hamiltonian directly into device-resident memory with zero host copies.
"""
function diagonal_device(H::MatrixFreeHamiltonian{T})::DeviceVector{T} where {T}
    dim = Int(dimension(H))
    diag_dev = DeviceVector{T}(dim)
    status = if T === Float32
        ccall((:qkrylov_hamiltonian_diagonal_device_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, diag_dev.ptr)
    else
        ccall((:qkrylov_hamiltonian_diagonal_device_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, diag_dev.ptr)
    end
    _check_status(status, "Failed to extract device Hamiltonian diagonal")
    return diag_dev
end

function Base.show(io::IO, H::MatrixFreeHamiltonian{T}) where {T}
    d = dimension(H)
    print(io, "MatrixFreeHamiltonian{$T}(dim = $d, device = \"$(H.device)\", basis = $(H.basis))")
end

"""
    isreal(H::MatrixFreeHamiltonian) -> Bool

`true` if every coefficient of `H` is real (a real symmetric matrix in this basis). Real
Hamiltonians run the Krylov solvers in real arithmetic and return real eigenvectors.
"""
function Base.isreal(H::MatrixFreeHamiltonian)
    status = ccall((:qkrylov_hamiltonian_is_real, libqkrylov), Cint, (Ptr{Cvoid},), H.ptr)
    status < 0 && error("qkrylov_hamiltonian_is_real failed: $(get_last_error_message())")
    return status == 1
end

"""
    active_device(H::MatrixFreeHamiltonian) -> String

The device `H` really runs on: `"cpu"`, or the GPU backend of this build (`"cuda"`, `"hip"`, `"sycl"`).
One GPU-enabled library serves both: `device="cpu"` runs on the host, `device="gpu"` (or the
build's backend name) on the GPU, and the choice is made per Hamiltonian.
"""
function active_device(H::MatrixFreeHamiltonian)::String
    c_str = ccall((:qkrylov_hamiltonian_device, libqkrylov), Cstring, (Ptr{Cvoid},), H.ptr)
    c_str == C_NULL && error("qkrylov_hamiltonian_device failed: $(get_last_error_message())")
    return unsafe_string(c_str)
end
