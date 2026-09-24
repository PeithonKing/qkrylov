#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/// \file types.hpp
/// \brief Fundamental scalar types, bit manipulation utilities, and precision namespaces.

namespace qkrylov {

/// \brief 64-bit integer representing a quantum basis Fock configuration bitstring.
/// <TODO-LLM: Document maximum site capacity and bit packing layout for spin vs fermion systems>
using StateID = uint64_t;

/// \brief Standard size and indexing integer type.
using Index   = std::size_t;
using ComplexDouble = std::complex<double>;

/// \brief Count the number of set bits (population count) in a 64-bit word.
///
/// Uses hardware intrinsic `__builtin_popcountll` on GCC/Clang or `__popcnt64` on MSVC.
/// <TODO-LLM: Document performance characteristics and CPU instruction set requirements (POPCNT/SSE4.2)>
/// \param x 64-bit unsigned integer to evaluate.
/// \return Number of bits set to 1.
inline int popcount(uint64_t x) noexcept {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

#ifdef QKRYLOV_DOUBLE_PRECISION
#define QKRYLOV_PRECISION_NAMESPACE fp64
#else
#define QKRYLOV_PRECISION_NAMESPACE fp32
#endif

namespace QKRYLOV_PRECISION_NAMESPACE {

#ifdef QKRYLOV_DOUBLE_PRECISION
using Real    = double;
using Complex = std::complex<double>;
#else
using Real    = float;
using Complex = std::complex<float>;
#endif

using qkrylov::StateID;
using qkrylov::Index;
using qkrylov::popcount;

/// Host-side vector of complex numbers.
/// Used in result types and API boundaries where host access is needed.
using HostVector = std::vector<Complex>;

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov