#include <iostream>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main()
{
    basis::SpinHalf basis(10, basis::sector::Sz{0});

    std::cout
        << "Basis dimension (Sz=0) = "
        << basis.size()
        << std::endl;

    if (basis.size() != 252) {
        std::cerr << "Expected dimension 252 for 10 sites Sz=0, got " << basis.size() << std::endl;
        return 1;
    }

    basis::SpinHalf b_full(4, basis::sector::Unconstrained{});
    if (b_full.size() != 16) {
        std::cerr << "Expected dimension 16 for 4 sites unconstrained, got " << b_full.size() << std::endl;
        return 1;
    }

    for(Index i = 0; i < 10 && i < basis.size(); ++i)
    {
        std::cout
            << i
            << "  "
            << basis.state(i)
            << '\n';
    }

    return 0;
}