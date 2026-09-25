#pragma once
#include "qkrylov/core/types.hpp"
#include <cstdint>

namespace qkrylov {

/// \brief A precompiled Virtual Machine instruction representing a bitwise quantum operator.
///
/// For binary sites (SpinHalf, Fermion, Hubbard), set_mask == 0 always.
/// For base-d sites (SpinS with S > 1/2), set_mask holds d^j for site j.
/// The GPU kernel uses set_mask == 0 as a fast-path branch to avoid any overhead for binary sites.
struct Instruction {
    uint64_t check_mask;
    uint64_t expected_bits;
    union {
        struct {
            uint64_t flip_mask;
            uint64_t set_mask;
            uint64_t set_value;
        } bin;
        struct {
            uint16_t d;
            uint16_t num_sites;
            uint8_t  site[4];
            uint8_t  src_sigma[4];
            uint8_t  tgt_sigma[4];
            uint8_t  padding[8]; // pad to 24 bytes to match bin
        } based;
    };
    uint64_t sign_mask;
    ComplexDouble coeff;
};

} // namespace qkrylov
