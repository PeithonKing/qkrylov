#pragma once

#include "qkrylov/core/instruction.hpp"
#include "qkrylov/core/types.hpp"
#include "qkrylov/operators/opsum.hpp"

#include <bit>
#include <vector>

/// \file bitwise_compiler.hpp
/// \brief Shared OpSum-term -> Bitmask-VM compiler for binary (one bit per mode) sites.
///
/// Semantics: in a product A * B * C, the rightmost factor acts first (standard operator
/// notation). Each factor is lowered to a few BitPrimitives acting on single bits of the
/// *current* state, so repeated operators on the same site (Sz*Sp, Cdag*C, ...) are exact.

namespace qkrylov {
namespace detail {

/// One elementary action on a single bit of the current state.
struct BitPrimitive {
    enum class Kind { Diag, Raise, Lower, Flip };
    Kind kind;
    uint64_t bit;              ///< single-bit mask
    ComplexDouble amp0;        ///< Diag: amplitude if bit=0. Flip: amplitude for 0->1. Raise: amplitude.
    ComplexDouble amp1;        ///< Diag: amplitude if bit=1. Flip: amplitude for 1->0. Lower: amplitude.
    uint64_t jw_mask = 0;      ///< fermionic sign string, evaluated on the current state
    uint64_t require_zero = 0; ///< extra bits that must be 0 in the current state (t-J projection)

    static BitPrimitive diag(uint64_t bit, ComplexDouble a0, ComplexDouble a1) { return {Kind::Diag, bit, a0, a1}; }
    static BitPrimitive raise(uint64_t bit, ComplexDouble a, uint64_t jw = 0, uint64_t rz = 0) { return {Kind::Raise, bit, a, 0.0, jw, rz}; }
    static BitPrimitive lower(uint64_t bit, ComplexDouble a, uint64_t jw = 0, uint64_t rz = 0) { return {Kind::Lower, bit, 0.0, a, jw, rz}; }
    static BitPrimitive flip(uint64_t bit, ComplexDouble a0to1, ComplexDouble a1to0) { return {Kind::Flip, bit, a0to1, a1to0}; }
};

namespace bitwise_compiler_impl {

inline bool known(const Instruction& inst, uint64_t bit) { return inst.check_mask & bit; }

/// Current value (0/1) of a constrained bit: source value XOR flips applied so far.
inline int current(const Instruction& inst, uint64_t bit) {
    return ((inst.expected_bits ^ inst.bin.flip_mask) & bit) ? 1 : 0;
}

/// Constrain the *source* state so that `bit` currently equals `value`.
inline Instruction with_current(Instruction inst, uint64_t bit, int value) {
    const bool src = (value != 0) != ((inst.bin.flip_mask & bit) != 0);
    inst.check_mask |= bit;
    if (src) inst.expected_bits |= bit; else inst.expected_bits &= ~bit;
    return inst;
}

/// Branch `inst` on the current value of `bit` (no-op if already known).
inline void branch(const Instruction& inst, uint64_t bit, std::vector<Instruction>& out) {
    if (known(inst, bit)) { out.push_back(inst); return; }
    out.push_back(with_current(inst, bit, 0));
    out.push_back(with_current(inst, bit, 1));
}

inline void apply_primitive(const BitPrimitive& p, const std::vector<Instruction>& in, std::vector<Instruction>& out) {
    using K = BitPrimitive::Kind;
    for (const auto& base : in) {
        // Constraints from the t-J projection: listed bits must currently be 0.
        Instruction inst = base;
        bool ok = true;
        for (uint64_t rz = p.require_zero; rz; rz &= rz - 1) {
            const uint64_t b = rz & (~rz + 1);
            if (known(inst, b)) { if (current(inst, b) != 0) { ok = false; break; } }
            else inst = with_current(inst, b, 0);
        }
        if (!ok) continue;

        // Flip with equal amplitudes needs no knowledge of the bit (e.g. Sx).
        if (p.kind == K::Flip && p.amp0 == p.amp1 && p.jw_mask == 0) {
            inst.coeff *= p.amp0;
            inst.bin.flip_mask ^= p.bit;
            out.push_back(inst);
            continue;
        }

        std::vector<Instruction> cases;
        branch(inst, p.bit, cases);
        for (auto c : cases) {
            const int v = current(c, p.bit);
            ComplexDouble amp;
            bool flips;
            switch (p.kind) {
                case K::Diag:  amp = v ? p.amp1 : p.amp0; flips = false; break;
                case K::Raise: if (v) continue; amp = p.amp0; flips = true; break;
                case K::Lower: if (!v) continue; amp = p.amp1; flips = true; break;
                case K::Flip:  amp = v ? p.amp1 : p.amp0; flips = true; break;
            }
            if (amp == ComplexDouble(0.0, 0.0)) continue;
            if (p.jw_mask) {
                // parity((source ^ flips) & m) = parity(source & m) ^ parity(flips & m)
                c.sign_mask ^= p.jw_mask;
                if (std::popcount(c.bin.flip_mask & p.jw_mask) & 1) amp = -amp;
            }
            c.coeff *= amp;
            if (flips) c.bin.flip_mask ^= p.bit;
            out.push_back(c);
        }
    }
}

} // namespace bitwise_compiler_impl

/// Compile one OpSum term. `lower_factor(factor, prims)` must append the BitPrimitives
/// for a single factor, in the order they act.
template <typename LowerFactor>
std::vector<Instruction> compile_bitwise(const OperatorTerm& term, LowerFactor&& lower_factor) {
    std::vector<Instruction> insts;
    Instruction init = {};
    init.coeff = term.coeff;
    insts.push_back(init);

    std::vector<BitPrimitive> prims;
    for (auto it = term.factors.rbegin(); it != term.factors.rend(); ++it) {
        prims.clear();
        lower_factor(*it, prims);
        for (const auto& p : prims) {
            std::vector<Instruction> next;
            bitwise_compiler_impl::apply_primitive(p, insts, next);
            insts = std::move(next);
            if (insts.empty()) return insts;
        }
    }
    return insts;
}

} // namespace detail
} // namespace qkrylov
