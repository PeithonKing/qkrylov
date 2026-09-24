#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/fermion_site.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov {



bool FermionSite::occupied(
    StateID state,
    int site
)
{
    return (state >> site) & 1ULL;
}

double FermionSite::phase(
    StateID state,
    int site
)
{
    // Jordan-Wigner phase: (-1)^(number of fermions to the left of 'site')
    // Here we assume sites are indexed 0, 1, 2, ...
    // So "to the left" means indices < site.
    StateID mask = (1ULL << site) - 1;
    int count = popcount(state & mask);
    return (count % 2 == 0) ? 1.0 : -1.0;
}

LocalAction FermionSite::apply(
    const std::string& op,
    int site,
    StateID state
) const
{
    LocalAction a;

    const bool occ =
        occupied(state, site);

    if(op == "N")
    {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = occ ? 1.0 : 0.0;
        return a;
    }

    if(op == "Id")
    {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = 1.0;
        return a;
    }

    if(op == "C")
    {
        if(!occ)
            return a;

        a.valid = true;
        a.new_state = state & ~(1ULL << site);
        a.matrix_element = phase(state, site);
        return a;
    }

    if(op == "Cdag")
    {
        if(occ)
            return a;

        a.valid = true;
        a.new_state = state | (1ULL << site);
        a.matrix_element = phase(state, site);
        return a;
    }

    throw std::runtime_error(
        "Unknown fermion operator: " + op
    );
}

std::vector<Instruction> FermionSite::compile(const OperatorTerm& term) const {
    std::vector<Instruction> insts;
    insts.push_back({0, 0, 0, 0, term.coeff});

    for (const auto& factor : term.factors) {
        std::vector<Instruction> next_insts;
        for (auto inst : insts) {
            uint64_t bit = 1ULL << factor.site;
            
            if (factor.op == "N") {
                if (inst.check_mask & bit) {
                    if (!(inst.expected_bits & bit)) continue;
                }
                inst.check_mask |= bit;
                inst.expected_bits |= bit;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Id") {
                next_insts.push_back(inst);
            }
            else if (factor.op == "C") {
                if (inst.check_mask & bit) {
                    if (!(inst.expected_bits & bit)) continue;
                }
                inst.check_mask |= bit;
                inst.expected_bits |= bit;
                inst.flip_mask ^= bit;
                inst.sign_mask ^= ((1ULL << factor.site) - 1);
                next_insts.push_back(inst);
            }
            else if (factor.op == "Cdag") {
                if (inst.check_mask & bit) {
                    if (inst.expected_bits & bit) continue;
                }
                inst.check_mask |= bit;
                inst.expected_bits &= ~bit;
                inst.flip_mask ^= bit;
                inst.sign_mask ^= ((1ULL << factor.site) - 1);
                next_insts.push_back(inst);
            }
            else {
                throw std::runtime_error("Unknown fermion operator: " + factor.op);
            }
        }
        insts = std::move(next_insts);
    }
    return insts;
}

}

