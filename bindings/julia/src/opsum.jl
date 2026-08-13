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

# -----------------------------------------------------------------------------
# Operator Term Arithmetic & Generator Functions (Sz, Sp, Sm, Sx, Sy, n, c, cdag)
# -----------------------------------------------------------------------------

struct OpTerm
    coeff::ComplexF64
    factors::Vector{Tuple{String, Int}}
end

struct OpExpr
    terms::Vector{OpTerm}
end

# Site Operator Generators
Sz(site::Integer)   = OpTerm(1.0, [("Sz", Int(site))])
Sp(site::Integer)   = OpTerm(1.0, [("Sp", Int(site))])
Sm(site::Integer)   = OpTerm(1.0, [("Sm", Int(site))])
Sx(site::Integer)   = OpTerm(1.0, [("Sx", Int(site))])
Sy(site::Integer)   = OpTerm(1.0, [("Sy", Int(site))])
n(site::Integer)    = OpTerm(1.0, [("n",  Int(site))])
c(site::Integer)    = OpTerm(1.0, [("c",  Int(site))])
cdag(site::Integer) = OpTerm(1.0, [("cdag", Int(site))])

# Scaling (*)
Base.:*(a::Number, t::OpTerm) = OpTerm(ComplexF64(a) * t.coeff, t.factors)
Base.:*(t::OpTerm, a::Number) = OpTerm(t.coeff * ComplexF64(a), t.factors)

Base.:*(a::Number, expr::OpExpr) = OpExpr([a * t for t in expr.terms])
Base.:*(expr::OpExpr, a::Number) = OpExpr([t * a for t in expr.terms])

# Unary minus (-)
Base.:-(t::OpTerm) = OpTerm(-t.coeff, t.factors)
Base.:-(expr::OpExpr) = OpExpr([-t for t in expr.terms])

# Term multiplication (*)
Base.:*(t1::OpTerm, t2::OpTerm) = OpTerm(t1.coeff * t2.coeff, vcat(t1.factors, t2.factors))

# Addition (+)
Base.:+(t1::OpTerm, t2::OpTerm) = OpExpr([t1, t2])
Base.:+(expr::OpExpr, t::OpTerm) = OpExpr(vcat(expr.terms, [t]))
Base.:+(t::OpTerm, expr::OpExpr) = OpExpr(vcat([t], expr.terms))
Base.:+(e1::OpExpr, e2::OpExpr)   = OpExpr(vcat(e1.terms, e2.terms))

# Subtraction (-)
Base.:-(t1::OpTerm, t2::OpTerm) = t1 + (-t2)
Base.:-(expr::OpExpr, t::OpTerm) = expr + (-t)
Base.:-(t::OpTerm, expr::OpExpr) = t + (-expr)
Base.:-(e1::OpExpr, e2::OpExpr)   = e1 + (-e2)

# Adding terms/expressions into OpSum
function add_term!(op::OpSum, t::OpTerm)
    ops = String[f[1] for f in t.factors]
    sites = Int[f[2] for f in t.factors]
    add_term!(op, t.coeff, ops, sites)
    return op
end

function add_term!(op::OpSum, expr::OpExpr)
    for t in expr.terms
        add_term!(op, t)
    end
    return op
end

# OpSum arithmetic (os += expr, os -= expr)
Base.:+(op::OpSum, t::OpTerm) = add_term!(op, t)
Base.:+(op::OpSum, expr::OpExpr) = add_term!(op, expr)
Base.:-(op::OpSum, t::OpTerm) = add_term!(op, -t)
Base.:-(op::OpSum, expr::OpExpr) = add_term!(op, -expr)
