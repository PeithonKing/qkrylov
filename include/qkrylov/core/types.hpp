#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace qkrylov {

using Real    = double;
using Complex = std::complex<double>;

using StateID = uint64_t;
using Index   = std::size_t;

/// Host-side vector of complex numbers.
/// Used in result types and API boundaries where host access is needed.
using HostVector = std::vector<Complex>;

inline int popcount(uint64_t x) noexcept {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

}