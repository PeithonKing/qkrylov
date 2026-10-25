#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/bitwise_compiler.hpp"

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

        // Sy = (Sp - Sm) / 2i:  Sy|up> = +i/2 |dn>,  Sy|dn> = -i/2 |up>
        a.matrix_element = up ? ComplexDouble(0.0, 0.5) : ComplexDouble(0.0, -0.5);

        return a;
    }

    throw std::runtime_error(
        "Unknown spin operator: " + op
    );
}


std::vector<Instruction> SpinHalfSite::compile(const OperatorTerm& term) const {
    using detail::BitPrimitive;
    return detail::compile_bitwise(term, [](const OperatorFactor& f, std::vector<BitPrimitive>& out) {
        const uint64_t bit = 1ULL << f.site;
        if      (f.op == "Sz") out.push_back(BitPrimitive::diag(bit, -0.5, 0.5));
        else if (f.op == "Sp") out.push_back(BitPrimitive::raise(bit, 1.0));
        else if (f.op == "Sm") out.push_back(BitPrimitive::lower(bit, 1.0));
        else if (f.op == "Sx") out.push_back(BitPrimitive::flip(bit, 0.5, 0.5));
        // Sy = (Sp - Sm) / 2i:  Sy|dn> = -i/2 |up>,  Sy|up> = +i/2 |dn>
        else if (f.op == "Sy") out.push_back(BitPrimitive::flip(bit, ComplexDouble(0.0, -0.5), ComplexDouble(0.0, 0.5)));
        else throw std::runtime_error("Unknown spin operator: " + f.op);
    });
}

} // namespace qkrylov
