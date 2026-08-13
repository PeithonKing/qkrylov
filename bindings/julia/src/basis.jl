# Hilbert space basis wrappers

abstract type AbstractBasis end

function dimension(b::AbstractBasis)::UInt64
    return ccall((:qkrylov_basis_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), b.ptr)
end

function nsites(b::AbstractBasis)::Int
    return Int(ccall((:qkrylov_basis_nsites, libqkrylov), Cint, (Ptr{Cvoid},), b.ptr))
end

function state(b::AbstractBasis, index::Integer)::UInt64
    return ccall((:qkrylov_basis_state, libqkrylov), UInt64, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(index))
end

function basis_index(b::AbstractBasis, state_bitstring::Unsigned)::Int64
    return ccall((:qkrylov_basis_index, libqkrylov), Int64, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(state_bitstring))
end

function Base.in(state_bitstring::Unsigned, b::AbstractBasis)::Bool
    res = ccall((:qkrylov_basis_contains, libqkrylov), Cint, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(state_bitstring))
    return res != 0
end

Base.size(b::AbstractBasis) = (Int(dimension(b)), Int(dimension(b)))
Base.length(b::AbstractBasis) = Int(dimension(b))
Base.getindex(b::AbstractBasis, i::Integer) = state(b, i - 1)

mutable struct SpinHalfBasis <: AbstractBasis
    ptr::Ptr{Cvoid}

    function SpinHalfBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
        sec_ptr = sector === nothing ? C_NULL : sector.ptr
        ptr = ccall((:qkrylov_spinhalf_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create SpinHalfBasis")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct FermionBasis <: AbstractBasis
    ptr::Ptr{Cvoid}

    function FermionBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
        sec_ptr = sector === nothing ? C_NULL : sector.ptr
        ptr = ccall((:qkrylov_fermion_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create FermionBasis")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct HubbardBasis <: AbstractBasis
    ptr::Ptr{Cvoid}

    function HubbardBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
        sec_ptr = sector === nothing ? C_NULL : sector.ptr
        ptr = ccall((:qkrylov_hubbard_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create HubbardBasis")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct TJBasis <: AbstractBasis
    ptr::Ptr{Cvoid}

    function TJBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
        sec_ptr = sector === nothing ? C_NULL : sector.ptr
        ptr = ccall((:qkrylov_tj_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create TJBasis")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end
