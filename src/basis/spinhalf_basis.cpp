#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {

SpinHalfBasis::SpinHalfBasis(
    int N,
    const Sector& sector
)
    : N_(N),
      sector_(sector)
{
    if(N <= 0 || N > 63)
    {
        throw std::runtime_error(
            "SpinHalfBasis: N must satisfy 1 <= N <= 63"
        );
    }

    if(sector_.sz.empty())
        build_full_basis();
    else
        build_sz_basis();
}

Index SpinHalfBasis::size() const
{
    if (sector_.sz.empty()) {
        return Index(1) << N_;
    }
    if (fixed_k_ >= 0) {
        return static_cast<Index>(comb::binomial(N_, fixed_k_));
    }
    return states_.size();
}

StateID SpinHalfBasis::state(Index i) const
{
    if (sector_.sz.empty()) {
        const Index dim = Index(1) << N_;
        if (i >= dim) {
            throw std::out_of_range("SpinHalfBasis::state: index out of range");
        }
        return static_cast<StateID>(i);
    }
    if (fixed_k_ >= 0) {
        if (i >= size()) {
            throw std::out_of_range("SpinHalfBasis::state: index out of range");
        }
        return comb::unrank(i, N_, fixed_k_);
    }
    return states_.at(i);
}

Index SpinHalfBasis::index(StateID s) const
{
    if (sector_.sz.empty()) {
        const StateID dim = StateID(1) << N_;
        if (s < dim) {
            return static_cast<Index>(s);
        }
        throw std::runtime_error("State not present in basis");
    }
    if (fixed_k_ >= 0) {
        if (!contains(s)) {
            throw std::runtime_error("State not present in basis");
        }
        return static_cast<Index>(comb::rank(s));
    }

    auto it = std::lower_bound(states_.begin(), states_.end(), s);
    if (it == states_.end() || *it != s) {
        throw std::runtime_error("State not present in basis");
    }
    return static_cast<Index>(std::distance(states_.begin(), it));
}

bool SpinHalfBasis::contains(StateID s) const
{
    if (sector_.sz.empty()) {
        const StateID dim = StateID(1) << N_;
        return s < dim;
    }
    if (fixed_k_ >= 0) {
        return (s >> N_) == 0 && std::popcount(s) == fixed_k_;
    }
    return std::binary_search(states_.begin(), states_.end(), s);
}

const std::vector<StateID>* SpinHalfBasis::state_list() const
{
    if (sector_.sz.empty()) return nullptr;
    if (fixed_k_ >= 0) materialize();
    return &states_;
}

bool SpinHalfBasis::fixed_popcount_sector(int& nbits, int& k) const
{
    if (sector_.sz.empty() || fixed_k_ < 0 || materialized_) return false;
    nbits = N_;
    k = fixed_k_;
    return true;
}

void SpinHalfBasis::materialize() const
{
    if (fixed_k_ < 0 || materialized_) return;
    const int k = fixed_k_;
    states_.clear();
    states_.reserve(static_cast<size_t>(comb::binomial(N_, k)));
    if (k == 0) {
        states_.push_back(0);
    } else if (k == N_) {
        states_.push_back((StateID(1) << N_) - 1);
    } else {
        // Gosper's hack: next integer with the same popcount
        StateID s = (StateID(1) << k) - 1;
        const StateID limit = StateID(1) << N_;
        while (s < limit) {
            states_.push_back(s);
            const StateID c = s & -s;
            const StateID r = s + c;
            s = (((r ^ s) >> 2) / c) | r;
        }
    }
    materialized_ = true;
}

void SpinHalfBasis::build_full_basis()
{
    // Implicit indexing: states_ remains empty, zero RAM allocated.
}

int SpinHalfBasis::compute_sz2(
    StateID state,
    int N
)
{
    const int nup = std::popcount(state);
    const int ndown = N - nup;
    return nup - ndown;
}

void SpinHalfBasis::build_sz_basis()
{
    std::vector<int> target_nup;
    for (double val : sector_.sz) {
        int sz2 = static_cast<int>(val >= 0 ? val*2.0 + 0.5 : val*2.0 - 0.5);
        // 2*Sz = nup - ndown = 2*nup - N
        int twice_nup = sz2 + N_;
        if (twice_nup >= 0 && twice_nup % 2 == 0 && (twice_nup / 2) <= N_) {
            target_nup.push_back(twice_nup / 2);
        }
    }
    std::sort(target_nup.begin(), target_nup.end());
    target_nup.erase(std::unique(target_nup.begin(), target_nup.end()), target_nup.end());

    if (target_nup.size() == 1) {
        // One sector: keep it as a formula; states are generated on demand / by the Hamiltonian.
        fixed_k_ = target_nup[0];
        return;
    }

    // Several sectors (or none valid): explicit sorted list.
    for (int k : target_nup) {
        if (k == 0) {
            states_.push_back(0);
        } else if (k == N_) {
            states_.push_back((StateID(1) << N_) - 1);
        } else {
            StateID s = (StateID(1) << k) - 1;
            const StateID limit = StateID(1) << N_;
            while (s < limit) {
                states_.push_back(s);
                const StateID c = s & -s;
                const StateID r = s + c;
                s = (((r ^ s) >> 2) / c) | r;
            }
        }
    }
    std::sort(states_.begin(), states_.end());
    states_.shrink_to_fit();
}

} // namespace qkrylov
