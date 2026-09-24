#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {

class SpinHalfBasis : public Basis
{
public:

    int bits_per_site() const override { return 1; }

    SpinHalfBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~SpinHalfBasis() override = default;

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

    void build_sz_basis();

    static int compute_sz2(
        StateID state,
        int N
    );

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::SpinHalfBasis;
}

namespace basis {
    using SpinHalf = qkrylov::SpinHalfBasis;
    
}

} // namespace qkrylov