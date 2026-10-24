#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "combinatorics.hpp"
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
    bool is_identity() const override { return sector_.sz.empty(); }
    const std::vector<StateID>* state_list() const override;
    bool fixed_popcount_sector(int& nbits, int& k) const override;

    /// Build the explicit state list now (a single-sector basis is otherwise only a formula).
    /// Afterwards the Hamiltonian uses the list path instead of generating states itself.
    void materialize() const;

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

    int fixed_k_ = -1;                    // >= 0: one Sz sector with k up spins, kept as a formula
    mutable bool materialized_ = false;   // fixed_k_ sector: has states_ been built?
    mutable std::vector<StateID> states_; // multi-sector bases: always; fixed_k_: only once materialized
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::SpinHalfBasis;
}

namespace basis {
    using SpinHalf = qkrylov::SpinHalfBasis;
    
}

} // namespace qkrylov