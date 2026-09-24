import re

# Fix Basis virtual method
with open("include/qkrylov/basis/basis.hpp", "r") as f:
    basis = f.read()
if "bits_per_site() const" not in basis:
    basis = basis.replace("virtual ~Basis() = default;", "virtual ~Basis() = default;\n\n    virtual int bits_per_site() const = 0;")
    with open("include/qkrylov/basis/basis.hpp", "w") as f:
        f.write(basis)

# Fix spin_s_basis.hpp
with open("include/qkrylov/basis/spin_s_basis.hpp", "r") as f:
    ssb = f.read()

ssb = re.sub(r'\s*SpinSBasis\(int N,\s*double S,\s*const sector::Sz&\s*sz\)\s*:\s*SpinSBasis\(N,\s*S,\s*Sector\(sz\)\)\s*\{\}', '', ssb, flags=re.DOTALL)
ssb = re.sub(r'\s*SpinSBasis\(int N,\s*double S,\s*const sector::Unconstrained&\s*u\)\s*:\s*SpinSBasis\(N,\s*S,\s*Sector\(u\)\)\s*\{\}', '', ssb, flags=re.DOTALL)
ssb = re.sub(r'\s*SpinSBasis\(int N,\s*const sector::Sz&\s*sz\)\s*:\s*SpinSBasis\(N,\s*0\.5,\s*Sector\(sz\)\)\s*\{\}', '', ssb, flags=re.DOTALL)
ssb = re.sub(r'\s*SpinSBasis\(int N,\s*const sector::Unconstrained&\s*u\)\s*:\s*SpinSBasis\(N,\s*0\.5,\s*Sector\(u\)\)\s*\{\}', '', ssb, flags=re.DOTALL)

with open("include/qkrylov/basis/spin_s_basis.hpp", "w") as f:
    f.write(ssb)

