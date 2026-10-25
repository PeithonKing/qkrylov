#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/bitwise_compiler.hpp"

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
    using detail::BitPrimitive;
    return detail::compile_bitwise(term, [](const OperatorFactor& f, std::vector<BitPrimitive>& out) {
        const uint64_t bit = 1ULL << f.site;
        const uint64_t jw  = bit - 1; // Jordan-Wigner string: all modes below this one
        if      (f.op == "N")    out.push_back(BitPrimitive::diag(bit, 0.0, 1.0));
        else if (f.op == "Id")   {}
        else if (f.op == "C")    out.push_back(BitPrimitive::lower(bit, 1.0, jw));
        else if (f.op == "Cdag") out.push_back(BitPrimitive::raise(bit, 1.0, jw));
        else throw std::runtime_error("Unknown fermion operator: " + f.op);
    });
}

}

