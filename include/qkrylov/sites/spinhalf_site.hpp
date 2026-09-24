#pragma once
#include "qkrylov/core/types.hpp"
#include "site.hpp"

namespace qkrylov {

class SpinHalfSite : public Site
{
public:
    int bits_per_site() const override { return 1; }

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

    std::vector<Instruction> compile(const OperatorTerm& term) const override;

private:
    static bool spin_up(StateID state, int site);
};

} // namespace qkrylov
