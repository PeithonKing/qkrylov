# Shared library loader and C API status codes for qkrylov

const QKRYLOV_SUCCESS           =  0
const QKRYLOV_ERROR_INVALID_ARG = -1
const QKRYLOV_ERROR_EXCEPTION   = -2

function find_libqkrylov()
    # 1. Custom environment variable override
    if haskey(ENV, "QKRYLOV_LIB_PATH") && isfile(ENV["QKRYLOV_LIB_PATH"])
        return ENV["QKRYLOV_LIB_PATH"]
    end

    # 2. Local repository build path (for development)
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

    # 3. Production JLL package resolution (if installed)
    try
        # Dynamically check if qkrylov_jll module is available
        jll_mod = Base.get_extension(QKrylov, :qkrylov_jll)
        if jll_mod !== nothing && isdefined(jll_mod, :libqkrylov)
            return jll_mod.libqkrylov
        end
    catch
    end

    # 4. Fallback to system library resolution
    return "libqkrylov"
end

const libqkrylov = find_libqkrylov()
