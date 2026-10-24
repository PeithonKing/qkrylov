#pragma once

#include "qkrylov/core/types.hpp"

#include <array>
#include <bit>
#include <stdexcept>

/// \file combinatorics.hpp
/// \brief Ranking and unranking of fixed-popcount bit strings (the combinatorial number system).
///
/// The N-bit integers with exactly k bits set, in ascending numeric order, are the k-subsets of
/// {0..N-1} in colexicographic order. So the position of a state inside a fixed-Sz spin-1/2
/// sector is a closed formula, and no state list is needed to answer size / state(i) / index(s).

namespace qkrylov {
namespace comb {

/// C(n, k) for 0 <= n, k <= 64 (0 when k > n). C(64, 32) = 1.83e18 fits in uint64_t.
inline const std::array<std::array<uint64_t, 65>, 65>& binomials() {
    static const auto table = [] {
        std::array<std::array<uint64_t, 65>, 65> c{};
        for (int n = 0; n <= 64; ++n) {
            c[n][0] = 1;
            for (int k = 1; k <= n; ++k) c[n][k] = c[n - 1][k - 1] + (k <= n - 1 ? c[n - 1][k] : 0);
        }
        return c;
    }();
    return table;
}

inline uint64_t binomial(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    return binomials()[n][k];
}

/// Rank of `s` among all integers with popcount(s) bits set, ascending numeric order.
inline uint64_t rank(StateID s) {
    uint64_t r = 0;
    int j = 1;
    while (s) {
        const int pos = std::countr_zero(s);
        r += binomial(pos, j);
        s &= s - 1;
        ++j;
    }
    return r;
}

/// The `r`-th (ascending) N-bit integer with exactly k bits set. Requires r < C(N, k).
inline StateID unrank(uint64_t r, int N, int k) {
    if (N < 0 || N > 63 || k < 0 || k > N || r >= binomial(N, k)) {
        throw std::out_of_range("comb::unrank: rank out of range");
    }
    StateID s = 0;
    for (int pos = N - 1; pos >= 0 && k > 0; --pos) {
        const uint64_t c = binomial(pos, k);
        if (c <= r) {
            s |= StateID(1) << pos;
            r -= c;
            --k;
        }
    }
    return s;
}

} // namespace comb
} // namespace qkrylov
