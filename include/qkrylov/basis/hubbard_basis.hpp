#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {

class HubbardBasis : public Basis
{
public:

    int bits_per_site() const override { return 2; }

    HubbardBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~HubbardBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;
    const std::vector<StateID>* state_list() const override { return &states_; }
    // bit 2i = (i, up), bit 2i+1 = (i, dn): N_up and N_dn are conserved separately
    std::vector<uint64_t> conserved_masks() const override { return {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL}; }

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

    void build_nup_ndn_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::HubbardBasis;
}

namespace basis {
    using Hubbard = qkrylov::HubbardBasis;
    
}

} // namespace qkrylov
