# Shared library loader and C API status codes for qkrylov

const QKRYLOV_SUCCESS           =  0
const QKRYLOV_ERROR_INVALID_ARG = -1
const QKRYLOV_ERROR_EXCEPTION   = -2

function find_libqkrylov()
    # 1. Custom environment variable override (for local development)
    if haskey(ENV, "QKRYLOV_LIB_PATH") && isfile(ENV["QKRYLOV_LIB_PATH"])
        return ENV["QKRYLOV_LIB_PATH"]
    end

    # 2. Production prebuilt binary from qkrylov_jll (Primary)
    try
        if isdefined(QKrylov, :qkrylov_jll) && isdefined(qkrylov_jll, :libqkrylov)
            return qkrylov_jll.libqkrylov
        end
    catch
    end

    # 3. Local repository build path (for development)
    root_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
    candidates = [
        joinpath(root_dir, "build", "libqkrylov.so"),
        joinpath(root_dir, "build", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "qkrylov.dll"),
        joinpath(root_dir, "build", "Release", "qkrylov.dll"),
        joinpath(root_dir, "build", "Debug", "qkrylov.dll")
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end

    # 4. Fallback to system library resolution
    return "libqkrylov"
end

const libqkrylov = find_libqkrylov()
