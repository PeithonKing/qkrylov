# Auto-generate per-file Starlight Markdown pages for QuantumKrylov.jl.
# Produces one .md page per source file (basis.md, solvers.md, etc.)
# plus an index.md table linking to all of them.
# Uses Julia's built-in Docs module to extract docstrings statically.

using Pkg

SCRIPT_DIR = @__DIR__
REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "../.."))
JULIA_SRC = joinpath(REPO_ROOT, "bindings", "julia")
OUT_DIR = joinpath(REPO_ROOT, "docs", "src", "content", "docs", "api", "auto", "julia")
mkpath(OUT_DIR)

# Source files we care about (in logical order)
SOURCE_FILES = [
    "basis",
    "site",
    "sector",
    "hamiltonian",
    "opsum",
    "solvers",
    "vector_ops",
    "device",
]

# Activate the Julia environment and load the module
Pkg.activate(JULIA_SRC, io=devnull)
include(joinpath(JULIA_SRC, "src", "QuantumKrylov.jl"))
using .QuantumKrylov

# Collect all documented symbols grouped by source file
meta = Docs.meta(QuantumKrylov)

# Map symbol -> source file based on @__module__ and file location hints
# We parse the file to extract exported names and match them to docstrings
function symbols_in_file(filename::String)
    filepath = joinpath(JULIA_SRC, "src", filename * ".jl")
    isfile(filepath) || return String[]
    code = read(filepath, String)
    names_found = String[]
    # Extract function/struct/abstract type names via simple regex
    for m in eachmatch(r"(?:^|\n)(?:function|struct|mutable struct|abstract type|const)\s+([A-Za-z_][A-Za-z0-9_!?]*)", code)
        push!(names_found, m.captures[1])
    end
    return unique(names_found)
end

function write_page(path::String, title::String, description::String, body::String)
    content = """---
title: "$(title)"
description: "$(description)"
---

$(body)
"""
    write(path, content)
end

function get_docstring(sym_name::String)::String
    for k in keys(meta)
        clean = replace(replace(string(k), "Main.QuantumKrylov." => ""), "QuantumKrylov." => "")
        if clean == sym_name
            multidoc = meta[k]
            parts = String[]
            for (sig, docstr) in multidoc.docs
                txt = hasproperty(docstr, :text) ? join(docstr.text, "\n") : string(docstr)
                push!(parts, txt)
            end
            return join(parts, "\n\n")
        end
    end
    return ""
end

index_rows = String[]

for src_file in SOURCE_FILES
    syms = symbols_in_file(src_file)
    body_io = IOBuffer()
    write(body_io, "# `QuantumKrylov.$src_file`\n\n")
    write(body_io, "Source: [`$src_file.jl`](https://github.com/sjp95/qkrylov/blob/main/bindings/julia/src/$src_file.jl)\n\n")

    documented = 0
    for sym in syms
        doc = get_docstring(sym)
        if !isempty(doc)
            write(body_io, "## `$sym`\n\n")
            write(body_io, doc)
            write(body_io, "\n\n")
            documented += 1
        end
    end

    if documented == 0
        write(body_io, "> No public symbols with docstrings found in this file yet.\n\n")
    end

    body = String(take!(body_io))
    out_path = joinpath(OUT_DIR, "$src_file.md")
    write_page(out_path, "QuantumKrylov.$src_file", "Julia API reference for QuantumKrylov.$src_file", body)
    println("[make_julia_docs] Generated $out_path ($documented symbols)")
    push!(index_rows, "| [`QuantumKrylov.$src_file`](./$src_file/) | $documented documented symbols |")
end

# Write index
index_body = """# Julia API Reference

Auto-generated from QuantumKrylov.jl source via Julia's built-in `Docs` module.

| Module | Contents |
|--------|----------|
$(join(index_rows, "\n"))
"""
index_path = joinpath(OUT_DIR, "index.md")
write_page(index_path, "Julia API Reference", "Auto-generated Julia API reference for QuantumKrylov.jl", index_body)
println("[make_julia_docs] Generated $index_path")
