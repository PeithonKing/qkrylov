# OpSum term construction interface

mutable struct OpSum
    ptr::Ptr{Cvoid}

    function OpSum()
        ptr = ccall((:qkrylov_opsum_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create OpSum")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_opsum_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function clear!(op::OpSum)
    status = ccall((:qkrylov_opsum_clear, libqkrylov), Cint, (Ptr{Cvoid},), op.ptr)
    status != QKRYLOV_SUCCESS && error("Failed to clear OpSum (status code $status)")
    return op
end

function add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer)
    c = ComplexF64(coeff)
    status = ccall(
        (:qkrylov_opsum_add_term_1body, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble, Cstring, Cint),
        op.ptr, real(c), imag(c), string(op1), Cint(site1)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 1-body term to OpSum (status code $status)")
    return op
end

function add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer, op2::AbstractString, site2::Integer)
    c = ComplexF64(coeff)
    status = ccall(
        (:qkrylov_opsum_add_term_2body, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble, Cstring, Cint, Cstring, Cint),
        op.ptr, real(c), imag(c), string(op1), Cint(site1), string(op2), Cint(site2)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 2-body term to OpSum (status code $status)")
    return op
end
