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
        (Ptr{Cvoid}, Cfloat, Cfloat, Cstring, Cint),
        op.ptr, Cfloat(real(c)), Cfloat(imag(c)), string(op1), Cint(site1)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 1-body term to OpSum (status code $status)")
    return op
end

function add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer, op2::AbstractString, site2::Integer)
    c = ComplexF64(coeff)
    status = ccall(
        (:qkrylov_opsum_add_term_2body, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cfloat, Cfloat, Cstring, Cint, Cstring, Cint),
        op.ptr, Cfloat(real(c)), Cfloat(imag(c)), string(op1), Cint(site1), string(op2), Cint(site2)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 2-body term to OpSum (status code $status)")
    return op
end

function add_term!(op::OpSum, coeff::Number, ops::AbstractVector{<:AbstractString}, sites::AbstractVector{<:Integer})
    @assert length(ops) == length(sites) "Length of operators ($(length(ops))) does not match length of site indices ($(length(sites)))"
    n_factors = length(ops)
    if n_factors == 1
        return add_term!(op, coeff, ops[1], sites[1])
    elseif n_factors == 2
        return add_term!(op, coeff, ops[1], sites[1], ops[2], sites[2])
    end

    c = ComplexF64(coeff)
    c_ops = [string(o) for o in ops]
    c_sites = Cint[Cint(s) for s in sites]
    ops_ptrs = [pointer(s) for s in c_ops]

    GC.@preserve c_ops c_sites ops_ptrs begin
        status = ccall(
            (:qkrylov_opsum_add_term_nbody, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cfloat, Cfloat, Cint, Ptr{Cstring}, Ptr{Cint}),
            op.ptr, Cfloat(real(c)), Cfloat(imag(c)), Cint(n_factors), ops_ptrs, c_sites
        )
    end
    status != QKRYLOV_SUCCESS && error("Failed to add $n_factors-body term to OpSum (status code $status)")
    return op
end
