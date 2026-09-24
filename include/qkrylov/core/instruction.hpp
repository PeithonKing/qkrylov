#pragma once
#include "qkrylov/core/types.hpp"
#include <cstdint>

namespace qkrylov {

/// \brief A precompiled Virtual Machine instruction representing a bitwise quantum operator.
struct Instruction {
    uint64_t check_mask;    // Bits that must be verified against expected_bits
    uint64_t expected_bits; // The required configuration for the check_mask bits
    uint64_t flip_mask;     // Bits to flip via XOR (off-diagonal action)
    uint64_t sign_mask;     // For fermions: flip sign if popcount(state & sign_mask) is odd
    ComplexDouble coeff;    // Amplitude multiplier
};

} // namespace qkrylov
