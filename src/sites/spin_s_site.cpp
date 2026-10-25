#include "qkrylov/core/types.hpp"
#include "qkrylov/core/instruction.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/operators/opsum.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace qkrylov {


SpinSSite::SpinSSite(double S)
    : S_(S),
      d_(static_cast<int>(std::round(2.0 * S + 1.0)))
{
    if (S <= 0.0) {
        throw std::runtime_error("SpinSSite: Spin S must be positive (e.g., 0.5, 1.0, 1.5, 2.0, 2.5).");
    }
}

LocalAction SpinSSite::apply(
    const std::string& op,
    int site,
    StateID state
) const
{
    LocalAction a;

    // Base-d state representation:
    // Local state sigma_j in {0, 1, ..., d-1} where m_z = sigma_j - S
    // For site j, state has factor d^j
    StateID d_pow = 1;
    for (int i = 0; i < site; ++i) {
        d_pow *= static_cast<StateID>(d_);
    }

    const int sigma = static_cast<int>((state / d_pow) % static_cast<StateID>(d_));
    const double mz = static_cast<double>(sigma) - S_;

    if (op == "Sz") {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = static_cast<double>(mz);
        return a;
    }

    if (op == "Sp") {
        if (sigma >= d_ - 1) {
            return a;
        }
        a.valid = true;
        a.new_state = state + d_pow;
        a.matrix_element = static_cast<double>(std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz + 1.0))));
        return a;
    }

    if (op == "Sm") {
        if (sigma <= 0) {
            return a;
        }
        a.valid = true;
        a.new_state = state - d_pow;
        a.matrix_element = static_cast<double>(std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz - 1.0))));
        return a;
    }

    if (op == "Sx") {
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sx as 0.5*Sp + 0.5*Sm in OpSum.");
    }

    if (op == "Sy") {
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sy as -0.5i*Sp + 0.5i*Sm in OpSum.");
    }

    throw std::runtime_error("Unknown spin operator for SpinSSite: " + op);
}

// ─── Bitmask VM compilation ───────────────────────────────────────────────────
// For base-d sites we cannot use binary XOR tricks. Instead each Instruction
// encodes a local transition on one site using the set_mask / set_value fields:
//
//   set_mask  = d^j          (which power of d is site j)
//   flip_mask = d^(j+1)      (next power, used as modulus to extract local sigma)
//   expected_bits = source_sigma  (required local state for validity)
//   set_value = target_sigma * d^j  (value to substitute in)
//   check_mask = set_mask    (same as set_mask – used by diagonal kernel)
//
// The GPU kernel checks: (state % flip_mask) / set_mask == expected_bits
// Then: new_state = state - expected_bits*set_mask + set_value
//
// For diagonal instructions: set_value == expected_bits * set_mask (same state).
// For off-diagonal: set_value != expected_bits * set_mask.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Instruction> SpinSSite::compile(const OperatorTerm& term) const {
    // Each intermediate Instruction represents a partial product across factors.
    // We store a list of (partial_instruction, coeff, current_state_offsets) tuples.
    // 
    // For SpinS we track per-site local sigma modifications as we walk the factors.
    // We start with one "identity" instruction and expand it through each factor.
    //
    // Representation for an intermediate partial result:
    //   - inst.coeff: accumulated amplitude
    //   - For each site touched: stored in a side-table mapping site -> (src_sigma, tgt_sigma, d^j)
    //     encoded in the Instruction fields as described above.
    //   - inst.flip_mask is reused per-site as d^(j+1) only for the LAST site in the product.
    //     Multiple sites require chain-composing.
    //
    // Because each factor acts on a different site (standard Hamiltonian assumption),
    // we compose factors sequentially, emitting one Instruction per valid (sigma_src -> sigma_tgt) path.

    // We represent each partial instruction as a vector of site-level transitions.
    struct SiteTransition {
        int site;
        int src_sigma;   // required input local state
        int tgt_sigma;   // output local state
        uint64_t d_pow;  // d^site
    };

    struct PartialInst {
        ComplexDouble coeff;
        std::vector<SiteTransition> transitions;
    };

    std::vector<PartialInst> partials;
    partials.push_back({term.coeff, {}});

    for (const auto& factor : term.factors) {
        uint64_t d_pow = 1;
        for (int i = 0; i < factor.site; ++i) d_pow *= static_cast<uint64_t>(d_);

        std::vector<PartialInst> next_partials;

        for (auto& partial : partials) {
            // Check if this site is already constrained
            int existing_tgt = -1;
            for (const auto& t : partial.transitions) {
                if (t.site == factor.site) {
                    existing_tgt = t.tgt_sigma;
                    break;
                }
            }

            // Enumerate all valid (src_sigma -> tgt_sigma) transitions for this operator
            int sigma_start = 0, sigma_end = d_;
            if (existing_tgt >= 0) {
                // Previous factor constrained this site, only one source is valid
                sigma_start = existing_tgt;
                sigma_end = existing_tgt + 1;
            }

            for (int src = sigma_start; src < sigma_end; ++src) {
                double mz = static_cast<double>(src) - S_;
                int tgt = -1;
                double me = 0.0;

                if (factor.op == "Sz") {
                    tgt = src;
                    me = mz;
                } else if (factor.op == "Sp") {
                    if (src < d_ - 1) {
                        tgt = src + 1;
                        me = std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz + 1.0)));
                    }
                } else if (factor.op == "Sm") {
                    if (src > 0) {
                        tgt = src - 1;
                        me = std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz - 1.0)));
                    }
                } else {
                    throw std::runtime_error("SpinSSite::compile: Unknown operator: " + factor.op);
                }

                if (tgt < 0 || me == 0.0) continue;

                // Update existing transition or add new one
                PartialInst new_partial = partial;
                new_partial.coeff *= me;

                bool updated = false;
                for (auto& t : new_partial.transitions) {
                    if (t.site == factor.site) {
                        t.tgt_sigma = tgt;
                        updated = true;
                        break;
                    }
                }
                if (!updated) {
                    new_partial.transitions.push_back({factor.site, src, tgt, d_pow});
                }
                next_partials.push_back(std::move(new_partial));
            }
        }

        partials = std::move(next_partials);
        if (partials.empty()) return {};
    }

    // Convert PartialInsts to Instructions
    // Encoding for base-d (SpinS) transitions:
    //   set_mask      = modulus = product of d^(site+1) for all constrained sites
    //   check_mask    = same as set_mask (used to detect base-d in kernel)  
    //   expected_bits = src residue = sum(src_sigma * d^j for constrained sites)
    //   set_value     = tgt residue = sum(tgt_sigma * d^j for constrained sites)
    //   flip_mask     = set_mask (= modulus, reused for the modulo operation)
    //
    // Kernel checks validity:  state % set_mask == expected_bits
    // Kernel computes alpha:   state_alpha = state - expected_bits + set_value
    //
    // For diagonal: set_value == expected_bits (src residue == tgt residue, state unchanged)
    // For off-diag: set_value != expected_bits

    std::vector<Instruction> result;
    for (const auto& partial : partials) {
        if (partial.transitions.empty()) continue;

        Instruction inst = {};
        inst.check_mask = 0;
        inst.expected_bits = 0;
        inst.sign_mask = 0;
        inst.coeff = partial.coeff;

        inst.based.d = d_;
        inst.based.num_sites = partial.transitions.size();
        for (size_t i = 0; i < partial.transitions.size(); ++i) {
            inst.based.site[i] = partial.transitions[i].site;
            inst.based.src_sigma[i] = partial.transitions[i].src_sigma;
            inst.based.tgt_sigma[i] = partial.transitions[i].tgt_sigma;
        }

        result.push_back(inst);
    }

    return result;
}


} // namespace qkrylov
