#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"

#include <stdexcept>

namespace qkrylov {



bool SpinHalfSite::spin_up(
    StateID state,
    int site
)
{
    return (state >> site) & 1ULL;
}

LocalAction SpinHalfSite::apply(
    const std::string& op,
    int site,
    StateID state
) const
{
    LocalAction a;

    const bool up =
        spin_up(state, site);

    //
    // Sz
    //
    if(op == "Sz")
    {
        a.valid = true;
        a.new_state = state;

        a.matrix_element =
            up ? 0.5 : -0.5;

        return a;
    }

    //
    // Sp
    //
    if(op == "Sp")
    {
        if(up)
            return a;

        a.valid = true;

        a.new_state =
            state | (1ULL << site);

        a.matrix_element = 1.0;

        return a;
    }

    //
    // Sm
    //
    if(op == "Sm")
    {
        if(!up)
            return a;

        a.valid = true;

        a.new_state =
            state & ~(1ULL << site);

        a.matrix_element = 1.0;

        return a;
    }

    //
    // Sx
    //
    if(op == "Sx")
    {
        a.valid = true;

        a.new_state =
            state ^ (1ULL << site);

        a.matrix_element = 0.5;

        return a;
    }

    //
    // Sy
    //
    if(op == "Sy")
    {
        a.valid = true;

        a.new_state =
            state ^ (1ULL << site);

        if(up)
        {
            a.matrix_element =
                ComplexDouble(0.0, -0.5);
        }
        else
        {
            a.matrix_element =
                ComplexDouble(0.0, 0.5);
        }

        return a;
    }

    throw std::runtime_error(
        "Unknown spin operator: " + op
    );
}


std::vector<Instruction> SpinHalfSite::compile(const OperatorTerm& term) const {
    std::vector<Instruction> insts;
    insts.push_back({0, 0, 0, 0, term.coeff});

    for (const auto& factor : term.factors) {
        std::vector<Instruction> next_insts;
        for (auto inst : insts) {
            uint64_t bit = 1ULL << factor.site;
            
            if (factor.op == "Sp") {
                if (inst.check_mask & bit) {
                    if (inst.expected_bits & bit) continue; // Requires UP, Sp is impossible
                }
                inst.check_mask |= bit;
                inst.expected_bits &= ~bit; // must be 0
                inst.flip_mask ^= bit;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sm") {
                if (inst.check_mask & bit) {
                    if (!(inst.expected_bits & bit)) continue; // Requires DN, Sm is impossible
                }
                inst.check_mask |= bit;
                inst.expected_bits |= bit; // must be 1
                inst.flip_mask ^= bit;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sz") {
                if (!(inst.check_mask & bit) || (inst.expected_bits & bit)) {
                    Instruction up = inst;
                    up.check_mask |= bit;
                    up.expected_bits |= bit;
                    up.coeff *= 0.5;
                    next_insts.push_back(up);
                }
                if (!(inst.check_mask & bit) || !(inst.expected_bits & bit)) {
                    Instruction dn = inst;
                    dn.check_mask |= bit;
                    dn.expected_bits &= ~bit;
                    dn.coeff *= -0.5;
                    next_insts.push_back(dn);
                }
            }
            else if (factor.op == "Sx") {
                inst.flip_mask ^= bit;
                inst.coeff *= 0.5;
                next_insts.push_back(inst);
            }
            else if (factor.op == "Sy") {
                if (!(inst.check_mask & bit) || (inst.expected_bits & bit)) {
                    Instruction up = inst;
                    up.check_mask |= bit;
                    up.expected_bits |= bit;
                    up.flip_mask ^= bit;
                    up.coeff *= ComplexDouble(0.0, -0.5);
                    next_insts.push_back(up);
                }
                if (!(inst.check_mask & bit) || !(inst.expected_bits & bit)) {
                    Instruction dn = inst;
                    dn.check_mask |= bit;
                    dn.expected_bits &= ~bit;
                    dn.flip_mask ^= bit;
                    dn.coeff *= ComplexDouble(0.0, 0.5);
                    next_insts.push_back(dn);
                }
            }
            else {
                throw std::runtime_error("Unknown spin operator: " + factor.op);
            }
        }
        insts = std::move(next_insts);
    }
    return insts;
}

} // namespace qkrylov
