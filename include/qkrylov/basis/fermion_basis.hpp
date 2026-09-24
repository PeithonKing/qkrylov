#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {

class FermionBasis : public Basis
{
public:

    int bits_per_site() const override { return 1; }

    FermionBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~FermionBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;

    int nsites() const noexcept
    {
        return N_;
    }

    const Sector& sector() const noexcept
    {
        return sector_;
    }

private:

    void build_full_basis();

    void build_n_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::FermionBasis;
}

namespace basis {
    using Fermion = qkrylov::FermionBasis;
    
}

} // namespace qkrylov
