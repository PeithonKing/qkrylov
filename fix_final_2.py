with open("include/qkrylov/basis/basis.hpp", "r") as f:
    basis = f.read()

if "bits_per_site" not in basis:
    basis = basis.replace("virtual ~Basis() = default;", "virtual ~Basis() = default;\n    virtual int bits_per_site() const { return 0; }")
    with open("include/qkrylov/basis/basis.hpp", "w") as f:
        f.write(basis)

# Fix SpinSBasis cpp use_sz and sz2
with open("src/basis/spin_s_basis.cpp", "r") as f:
    ssb = f.read()

ssb = ssb.replace("sector_.use_sz", "(!sector_.sz.empty())")
ssb = ssb.replace("sector_.sz2", "(int(sector_.sz[0] * 2.0))")

with open("src/basis/spin_s_basis.cpp", "w") as f:
    f.write(ssb)
