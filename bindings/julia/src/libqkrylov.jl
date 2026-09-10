# Shared library loader and C API status codes for qkrylov

const QKRYLOV_SUCCESS           =  0
const QKRYLOV_ERROR_INVALID_ARG = -1
const QKRYLOV_ERROR_EXCEPTION   = -2

struct LocalActionC
    valid::Cint
    new_state::UInt64
    matrix_element_re::Cdouble
    matrix_element_im::Cdouble
end

@enum BasisTypeC begin
    BASIS_SPIN_HALF = 0
    BASIS_SPIN_S    = 1
    BASIS_FERMION   = 2
    BASIS_HUBBARD   = 3
    BASIS_TJ        = 4
end

@enum SiteTypeC begin
    SITE_SPIN_HALF = 0
    SITE_SPIN_S    = 1
    SITE_FERMION   = 2
    SITE_HUBBARD   = 3
    SITE_TJ        = 4
end

"""
    get_last_error_message() -> String

Returns the last error message caught across the C ABI barrier on the current calling thread.
Returns an empty string `""` if no error occurred.
"""
function get_last_error_message()::String
    ptr = ccall((:qkrylov_get_last_error_message, libqkrylov), Cstring, ())
    return ptr == C_NULL ? "" : unsafe_string(ptr)
end

"""
    clear_last_error()

Clears the thread-local error message for the current calling thread.
"""
function clear_last_error()
    ccall((:qkrylov_clear_last_error, libqkrylov), Cvoid, ())
end

function _check_status(status::Integer, default_msg::AbstractString)
    if status != QKRYLOV_SUCCESS
        err = get_last_error_message()
        if !isempty(err)
            error("$default_msg (status code $status): $err")
        else
            error("$default_msg with status code $status")
        end
    end
end

function find_libqkrylov()
    # 1. Custom environment variable override (for local development or CI)
    if haskey(ENV, "QKRYLOV_LIB_PATH")
        p = ENV["QKRYLOV_LIB_PATH"]
        if isfile(p)
            return p
        else
            @warn "QKRYLOV_LIB_PATH is set to '$p' but file was not found. Searching fallback paths."
        end
    end

    # 2. Local repository build path (for development)
    root_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
    candidates = [
        joinpath(root_dir, "build", "libqkrylov.so"),
        joinpath(root_dir, "build", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "qkrylov.dll"),
        joinpath(root_dir, "build", "Release", "libqkrylov.so"),
        joinpath(root_dir, "build", "Release", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "Release", "qkrylov.dll"),
        joinpath(root_dir, "build", "Debug", "libqkrylov.so"),
        joinpath(root_dir, "build", "Debug", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "Debug", "qkrylov.dll")
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end


    # 3. Production prebuilt binary from qkrylov_jll (Primary fallback)
    try
        if isdefined(QuantumKrylov, :qkrylov_jll) && isdefined(qkrylov_jll, :libqkrylov)
            return qkrylov_jll.libqkrylov
        end
    catch
    end

    # 4. Fallback to system library resolution
    return "libqkrylov"
end

global libqkrylov::String = find_libqkrylov()

function __init__()
    global libqkrylov = find_libqkrylov()
end

