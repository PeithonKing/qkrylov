#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/combinatorics.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>

/// \file state_index.hpp
/// \brief Device-side state -> index lookup for sorted basis state lists.
///
/// Three modes, chosen and verified at build time:
///  - Identity: the basis is {0, 1, ..., dim-1} (full spin-1/2 space). index = state, no tables.
///  - Lin:      H.Q. Lin's two-table scheme. A state splits into high/low bit halves and
///              index = hi_off[high] + lo_rank[low] -- two cache-resident reads, no search.
///              Valid when, for each high part, the matching low parts are exactly all lows of
///              one "class" (same popcounts under the basis' conserved masks). This is checked
///              exactly against the state list; otherwise we fall back to:
///  - Bisect:   a prefix table over the top bits gives a short window, then binary search.

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
struct StateIndex {
    using MemSpace = typename ExecSpace::memory_space;
    enum Mode : int { Identity = 0, Lin = 1, Bisect = 2 };

    int mode = Identity;
    int low_bits = 0;
    uint64_t low_mask = 0;
    uint64_t n_hi = 0;
    Index dim = 0;
    bool complete = false; ///< Lin: basis is every state with its popcount classes (no extra constraint)

    Kokkos::View<const StateID*, MemSpace> states;   ///< sorted states (Bisect only)
    Kokkos::View<const Index*, MemSpace> hi_off;     ///< first index per high part (n_hi + 1 entries)
    Kokkos::View<const int32_t*, MemSpace> hi_class; ///< Lin: class of each high group (-2 = empty)
    Kokkos::View<const uint32_t*, MemSpace> lo_rank; ///< Lin: rank of a low part inside its class
    Kokkos::View<const int32_t*, MemSpace> lo_class; ///< Lin: class of a low part (-1 = never used)

    /// Find the index of `s`; returns false if `s` is not in the basis.
    KOKKOS_INLINE_FUNCTION bool find(StateID s, Index& idx) const {
        if (mode == Identity) {
            idx = static_cast<Index>(s);
            return s < static_cast<StateID>(dim);
        }
        const uint64_t h = s >> low_bits;
        if (h >= n_hi) return false;
        const uint64_t l = s & low_mask;
        if (mode == Lin) {
            const int32_t c = lo_class(l);
            if (c < 0 || c != hi_class(h)) return false;
            idx = hi_off(h) + lo_rank(l);
            return true;
        }
        Index lo = hi_off(h), hi = hi_off(h + 1);
        const Index end = hi;
        while (lo < hi) {
            const Index m = lo + (hi - lo) / 2;
            if (states(m) < s) lo = m + 1; else hi = m;
        }
        if (lo < end && states(lo) == s) { idx = lo; return true; }
        return false;
    }
};

namespace state_index_detail {

inline int32_t state_class(uint64_t x, const std::vector<uint64_t>& masks) {
    int32_t key = 0;
    for (uint64_t m : masks) key = key * 65 + std::popcount(x & m);
    return key;
}

template <typename T, typename MemSpace>
Kokkos::View<const T*, MemSpace> upload(const std::vector<T>& v, const char* label) {
    Kokkos::View<T*, MemSpace> d(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), v.size());
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(v.data(), v.size());
    Kokkos::deep_copy(d, h);
    return d;
}

} // namespace state_index_detail

/// Index for the basis {0, 1, ..., dim-1}: index = state.
template <typename ExecSpace>
StateIndex<ExecSpace> make_identity_index(Index dim) {
    StateIndex<ExecSpace> idx;
    idx.mode = StateIndex<ExecSpace>::Identity;
    idx.dim = dim;
    return idx;
}

/// Build a verified StateIndex from a strictly ascending host state list.
/// `masks` are the bit masks whose popcounts the basis conserves (e.g. all bits for an Sz
/// sector, even/odd bits for Hubbard). `device_states` is the already-uploaded state list,
/// reused by the Bisect mode.
template <typename ExecSpace>
StateIndex<ExecSpace> build_state_index(
    const std::vector<StateID>& states,
    const std::vector<uint64_t>& masks,
    Kokkos::View<const StateID*, typename ExecSpace::memory_space> device_states)
{
    using MemSpace = typename ExecSpace::memory_space;
    StateIndex<ExecSpace> idx;
    const Index dim = states.size();
    idx.dim = dim;
    idx.mode = StateIndex<ExecSpace>::Bisect;
    if (dim == 0) return idx;
    for (Index i = 1; i < dim; ++i) {
        if (states[i - 1] >= states[i]) {
            throw std::invalid_argument("Basis states must be strictly ascending for index lookup");
        }
    }

    const int nbits = std::max(1, static_cast<int>(std::bit_width(states.back())));

    // ---- Try Lin's two-table scheme ------------------------------------------------------
    const int lb = nbits / 2;
    const int hb = nbits - lb;
    if (lb <= 24 && hb <= 24) {
        const uint64_t n_lo = 1ULL << lb, n_hi = 1ULL << hb, lmask = n_lo - 1;
        std::vector<Index> hi_off(n_hi + 1, 0);
        std::vector<int32_t> hi_class(n_hi, -2), lo_class(n_lo, -1);
        std::vector<uint32_t> lo_rank(n_lo, 0);
        std::vector<char> seen(n_lo, 0);

        bool ok = true;
        for (Index i = 0; i < dim; ++i) {
            const StateID s = states[i];
            hi_off[(s >> lb) + 1]++;
            seen[s & lmask] = 1;
        }
        for (uint64_t h = 0; h < n_hi; ++h) hi_off[h + 1] += hi_off[h];

        // Rank every used low part inside its class, in ascending order.
        std::vector<uint32_t> class_count;
        for (uint64_t l = 0; l < n_lo; ++l) {
            if (!seen[l]) continue;
            const int32_t c = state_index_detail::state_class(l, masks);
            if (static_cast<size_t>(c) >= class_count.size()) class_count.resize(c + 1, 0);
            lo_class[l] = c;
            lo_rank[l] = class_count[c]++;
        }
        // Verify: each group is exactly one complete class and every state maps to itself.
        for (uint64_t h = 0; h < n_hi && ok; ++h) {
            const Index a = hi_off[h], b = hi_off[h + 1];
            if (a == b) continue;
            const int32_t c = lo_class[states[a] & lmask];
            hi_class[h] = c;
            if (b - a != class_count[c]) ok = false;
        }
        for (Index i = 0; i < dim && ok; ++i) {
            const StateID s = states[i];
            const uint64_t h = s >> lb, l = s & lmask;
            if (lo_class[l] != hi_class[h] || hi_off[h] + lo_rank[l] != i) ok = false;
        }
        if (ok) {
            // Completeness: is the basis *every* nbits-wide state whose popcount tuple is
            // realised? Count them from per-half histograms and compare with dim. t-J
            // (no double occupancy) fails this, and then hops must keep the class check.
            using Tuple = std::array<int, 4>;
            const size_t nm = std::min<size_t>(masks.size(), 4);
            auto tuple_of = [&](uint64_t x) {
                Tuple t{}; for (size_t k = 0; k < nm; ++k) t[k] = std::popcount(x & masks[k]); return t;
            };
            std::map<Tuple, uint64_t> hi_hist, lo_hist;
            std::map<int32_t, Tuple> class_tuple;
            for (uint64_t h = 0; h < n_hi; ++h) hi_hist[tuple_of(h << lb)]++;
            for (uint64_t l = 0; l < n_lo; ++l) {
                lo_hist[tuple_of(l)]++;
                if (seen[l]) class_tuple.emplace(lo_class[l], tuple_of(l));
            }
            std::set<Tuple> realised;
            for (uint64_t h = 0; h < n_hi; ++h) {
                if (hi_class[h] < 0) continue;
                Tuple a = tuple_of(h << lb), b = class_tuple[hi_class[h]], t{};
                for (size_t k = 0; k < nm; ++k) t[k] = a[k] + b[k];
                realised.insert(t);
            }
            // `total` counts N-bit states, each at most once (a state has exactly one popcount tuple),
            // so total <= 2^N <= 2^64 because StateID is 64-bit. Plain uint64_t is therefore enough:
            // the only value that wraps is 2^64 (the unconstrained full space), which becomes 0 and
            // still differs from dim, i.e. the correct answer "not complete".
            static_assert(sizeof(StateID) == 8, "completeness count assumes at most 64 state bits");
            uint64_t total = 0;
            for (const auto& t : realised) {
                for (const auto& [a, na] : hi_hist) {
                    Tuple b{}; bool valid = true;
                    for (size_t k = 0; k < nm; ++k) { b[k] = t[k] - a[k]; if (b[k] < 0) valid = false; }
                    if (!valid) continue;
                    auto it = lo_hist.find(b);
                    if (it != lo_hist.end()) total += na * it->second;
                }
            }
            idx.complete = (total == static_cast<uint64_t>(dim));
            idx.mode = StateIndex<ExecSpace>::Lin;
            idx.low_bits = lb;
            idx.low_mask = lmask;
            idx.n_hi = n_hi;
            idx.hi_off = state_index_detail::upload<Index, MemSpace>(hi_off, "qkrylov::lin_hi_off");
            idx.hi_class = state_index_detail::upload<int32_t, MemSpace>(hi_class, "qkrylov::lin_hi_class");
            idx.lo_rank = state_index_detail::upload<uint32_t, MemSpace>(lo_rank, "qkrylov::lin_lo_rank");
            idx.lo_class = state_index_detail::upload<int32_t, MemSpace>(lo_class, "qkrylov::lin_lo_class");
            return idx;
        }
    }

    // ---- Fallback: prefix table + windowed binary search ----------------------------------
    // About 8 states per window on average, table capped at 2^24 entries.
    const int want_hb = std::clamp(static_cast<int>(std::bit_width(dim)) - 3, 1, std::min(nbits, 24));
    const int low = nbits - want_hb;
    const uint64_t n_hi = 1ULL << want_hb;
    std::vector<Index> hi_off(n_hi + 1, 0);
    for (Index i = 0; i < dim; ++i) hi_off[(states[i] >> low) + 1]++;
    for (uint64_t h = 0; h < n_hi; ++h) hi_off[h + 1] += hi_off[h];
    idx.mode = StateIndex<ExecSpace>::Bisect;
    idx.low_bits = low;
    idx.low_mask = low >= 64 ? ~0ULL : ((1ULL << low) - 1);
    idx.n_hi = n_hi;
    idx.hi_off = state_index_detail::upload<Index, MemSpace>(hi_off, "qkrylov::bisect_hi_off");
    idx.states = device_states;
    return idx;
}

/// Build the Lin lookup tables AND the sorted state list for the basis "all nbits-wide integers
/// with k bits set" directly from the combinatorial formulas -- no host state list, no sort, no
/// verification pass. The tables have only ~2^(nbits/2) entries; the O(dim) work is one parallel
/// kernel on ExecSpace that writes `states` (group by group). Returns false (outputs untouched)
/// when the halves would be too wide for the tables; the caller then uses the list-based path.
template <typename ExecSpace>
bool build_fixed_popcount_index(
    int nbits, int k,
    StateIndex<ExecSpace>& idx,
    Kokkos::View<StateID*, ExecSpace>& states)
{
    using MemSpace = typename ExecSpace::memory_space;
    if (nbits < 1 || nbits > 62 || k < 0 || k > nbits) return false;
    const int lb = nbits / 2;
    const int hb = nbits - lb;
    if (lb > 24 || hb > 24) return false;
    const uint64_t n_lo = 1ULL << lb, n_hi = 1ULL << hb, lmask = n_lo - 1;
    const Index dim = static_cast<Index>(comb::binomial(nbits, k));

    // Low-part classes: popcount c is used iff some high part supplies the other k - c bits.
    const int c_min = std::max(0, k - hb), c_max = std::min(lb, k);
    std::vector<int32_t> lo_class(n_lo, -1), class_off(lb + 2, 0);
    std::vector<uint32_t> lo_rank(n_lo, 0);
    std::vector<StateID> lo_sorted;
    for (int c = 0; c <= lb; ++c) {
        const bool used = (c >= c_min && c <= c_max);
        class_off[c + 1] = class_off[c] + static_cast<int32_t>(used ? comb::binomial(lb, c) : 0);
    }
    lo_sorted.assign(static_cast<size_t>(class_off[lb + 1]), 0);
    for (uint64_t l = 0; l < n_lo; ++l) {
        const int c = std::popcount(l);
        if (c < c_min || c > c_max) continue;
        const uint64_t r = comb::rank(l);
        lo_class[l] = c;
        lo_rank[l] = static_cast<uint32_t>(r);
        lo_sorted[static_cast<size_t>(class_off[c]) + r] = l;
    }

    // High parts: group h holds all lows of class k - popcount(h), in ascending order.
    std::vector<Index> hi_off(n_hi + 1, 0);
    std::vector<int32_t> hi_class(n_hi, -2);
    for (uint64_t h = 0; h < n_hi; ++h) {
        const int c = k - std::popcount(h);
        uint64_t cnt = 0;
        if (c >= 0 && c <= lb) { hi_class[h] = c; cnt = comb::binomial(lb, c); }
        hi_off[h + 1] = hi_off[h] + static_cast<Index>(cnt);
    }
    if (hi_off[n_hi] != dim) return false;   // would indicate a bug in the formulas

    auto h_states = Kokkos::View<StateID*, ExecSpace>(
        Kokkos::view_alloc(std::string("qkrylov::basis_states"), Kokkos::WithoutInitializing), dim);
    auto d_hi_off = state_index_detail::upload<Index, MemSpace>(hi_off, "qkrylov::fp_hi_off");
    auto d_hi_class = state_index_detail::upload<int32_t, MemSpace>(hi_class, "qkrylov::fp_hi_class");
    auto d_lo_sorted = state_index_detail::upload<StateID, MemSpace>(lo_sorted, "qkrylov::fp_lo_sorted");
    auto d_class_off = state_index_detail::upload<int32_t, MemSpace>(class_off, "qkrylov::fp_class_off");

    // One thread per high part writes its group of states.
    Kokkos::parallel_for("qkrylov::generate_states",
        Kokkos::RangePolicy<ExecSpace, Index>(0, n_hi),
        KOKKOS_LAMBDA(const Index h) {
            const int32_t c = d_hi_class(h);
            if (c < 0) return;
            const Index off = d_hi_off(h);
            const Index cnt = d_hi_off(h + 1) - off;
            const StateID high = static_cast<StateID>(h) << lb;
            const StateID* lows = &d_lo_sorted(static_cast<size_t>(d_class_off(c)));
            for (Index r = 0; r < cnt; ++r) h_states(off + r) = high | lows[r];
        });

    idx = StateIndex<ExecSpace>{};
    idx.mode = StateIndex<ExecSpace>::Lin;
    idx.low_bits = lb;
    idx.low_mask = lmask;
    idx.n_hi = n_hi;
    idx.dim = dim;
    idx.complete = true;
    idx.hi_off = d_hi_off;
    idx.hi_class = d_hi_class;
    idx.lo_rank = state_index_detail::upload<uint32_t, MemSpace>(lo_rank, "qkrylov::fp_lo_rank");
    idx.lo_class = state_index_detail::upload<int32_t, MemSpace>(lo_class, "qkrylov::fp_lo_class");
    states = h_states;
    return true;
}

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
