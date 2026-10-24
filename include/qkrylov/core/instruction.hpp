#pragma once
#include "qkrylov/core/types.hpp"
#include <cstddef>
#include <cstdint>

namespace qkrylov {

/// \brief A precompiled Virtual Machine instruction representing a bitwise quantum operator.
///
/// Binary sites (SpinHalf, Fermion, Hubbard, t-J) only write `bin.flip_mask`.
/// Base-d sites (SpinS with S > 1/2) only write the `based` fields and set `based.d = 2S+1 >= 3`.
/// The kernels branch on `based.d == 0` to pick the binary fast path, so `based.d` must live in
/// bytes that binary instructions never touch (they stay zero because every Instruction is
/// value-initialised). The static_asserts below enforce that layout.
struct Instruction {
    uint64_t check_mask;
    uint64_t expected_bits;
    union {
        struct {
            uint64_t flip_mask;
            uint64_t reserved;    // always zero for binary instructions (holds based.d)
            uint64_t alt_xor;     // merged pair: also valid if masked state == expected_bits ^ alt_xor
        } bin;
        struct {
            uint64_t reserved;    // overlaps bin.flip_mask; never written for base-d instructions
            uint16_t d;           // local dimension; 0 => binary instruction
            uint16_t num_sites;
            uint8_t  site[4];
            uint8_t  src_sigma[4];
            uint8_t  tgt_sigma[4];
        } based;
    };
    uint64_t sign_mask;
    ComplexDouble coeff;
};

static_assert(sizeof(Instruction) == 64, "Instruction must stay one 64-byte cache line");
static_assert(offsetof(Instruction, based.d) >= offsetof(Instruction, bin.flip_mask) + sizeof(uint64_t),
              "based.d must not overlap bin.flip_mask (binary instructions would take the base-d path)");
static_assert(offsetof(Instruction, based.d) + sizeof(uint16_t) <= offsetof(Instruction, bin.alt_xor),
              "based.d must not overlap bin.alt_xor");
static_assert(offsetof(Instruction, based.tgt_sigma) + 4 <= offsetof(Instruction, sign_mask),
              "based fields must fit inside the union");

} // namespace qkrylov
