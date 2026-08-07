# Sector symmetry wrapper

mutable struct Sector
    ptr::Ptr{Cvoid}

    function Sector()
        ptr = ccall((:qkrylov_sector_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create Sector")
        
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_sector_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function set_sz!(sec::Sector, sz2::Integer)
    status = ccall((:qkrylov_sector_set_sz, libqkrylov), Cint, (Ptr{Cvoid}, Cint), sec.ptr, Cint(sz2))
    status != QKRYLOV_SUCCESS && error("Failed to set sz sector (status code $status)")
    return sec
end

function set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)
    status = ccall((:qkrylov_sector_set_hubbard_particles, libqkrylov), Cint, (Ptr{Cvoid}, Cint, Cint), sec.ptr, Cint(nup), Cint(ndn))
    status != QKRYLOV_SUCCESS && error("Failed to set hubbard particles sector (status code $status)")
    return sec
end
