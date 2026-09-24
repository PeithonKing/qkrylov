#pragma once

#include "qkrylov/core/types.hpp"

#include "site.hpp"

namespace qkrylov {



class FermionSite : public Site
{
public:

    int bits_per_site() const override { return 1; }


    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied(
        StateID state,
        int site
    );

    static double phase(
        StateID state,
        int site
    );

    std::vector<Instruction> compile(const OperatorTerm& term) const override;
};

}

