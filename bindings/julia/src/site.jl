# Site wrapper structs

abstract type AbstractSite end

mutable struct SpinHalfSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function SpinHalfSite()
        ptr = ccall((:qkrylov_spinhalf_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create SpinHalfSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct FermionSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function FermionSite()
        ptr = ccall((:qkrylov_fermion_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create FermionSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct HubbardSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function HubbardSite()
        ptr = ccall((:qkrylov_hubbard_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create HubbardSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct TJSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function TJSite()
        ptr = ccall((:qkrylov_tj_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create TJSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

Base.show(io::IO, ::SpinHalfSite) = print(io, "SpinHalfSite(dim = 2, states = [↑, ↓])")
Base.show(io::IO, ::FermionSite)  = print(io, "FermionSite(dim = 2, states = [0, 1])")
Base.show(io::IO, ::HubbardSite)  = print(io, "HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])")
Base.show(io::IO, ::TJSite)       = print(io, "TJSite(dim = 3, states = [0, ↑, ↓])")
