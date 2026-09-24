#pragma once

#include "qkrylov/core/types.hpp"

#include "site.hpp"

namespace qkrylov {



class HubbardSite : public Site
{
public:

    int bits_per_site() const override { return 2; }


    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied_up(
        StateID state,
        int site
    );

    static bool occupied_dn(
        StateID state,
        int site
    );

    static double phase_up(
        StateID state,
        int site
    );

    static double phase_dn(
        StateID state,
        int site
    );

    std::vector<Instruction> compile(const OperatorTerm& term) const override;
};

}

