#pragma once

#include "qkrylov/core/types.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



struct Sector
{
    //
    // Spin quantum number conservation
    //
    bool use_sz = false;

    // 2*Sz
    int sz2 = 0;

    //
    // Future Hubbard/t-J support
    //
    bool use_nup = false;
    bool use_ndn = false;

    int nup = 0;
    int ndn = 0;

    //
    // Fermion particle number conservation
    //
    bool use_n = false;
    int n = 0;

    //
    // Future boson support
    //
    bool use_nb = false;

    int nb = 0;
};



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov