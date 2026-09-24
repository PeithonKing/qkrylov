#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {

class TJBasis : public Basis
{
public:

    int bits_per_site() const override { return 2; }

    TJBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~TJBasis() override = default;

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

    void build_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::TJBasis;
}

namespace basis {
    using TJ = qkrylov::TJBasis;
    
}

} // namespace qkrylov
