#pragma once

#include "qkrylov/core/types.hpp"
#include <vector>

namespace qkrylov {

struct Sector
{
    std::vector<double> sz;
    std::vector<int> nup;
    std::vector<int> ndn;
    std::vector<int> n;
    std::vector<int> nb;

    Sector() = default;

    bool unconstrained() const {
        return sz.empty() && nup.empty() && ndn.empty() && n.empty() && nb.empty();
    }
};

} // namespace qkrylov
