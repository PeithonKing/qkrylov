#include <iostream>
#include <memory>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"

#include "qkrylov/sites/spinhalf_site.hpp"

#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main()
{
    Sector sec;

    auto basis =
        std::make_shared<SpinHalfBasis>(
            2,
            sec
        );

    auto site =
        std::make_shared<SpinHalfSite>();

    OpSum os;

    {
        OperatorTerm t;

        t.coeff = 1.0;

        t.factors.push_back(
            {"Sz",0}
        );

        t.factors.push_back(
            {"Sz",1}
        );

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back(
            {"Sp",0}
        );

        t.factors.push_back(
            {"Sm",1}
        );

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back(
            {"Sm",0}
        );

        t.factors.push_back(
            {"Sp",1}
        );

        os.add_term(t);
    }

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(
        basis,
        site,
        os
    );

    auto res1 =
        lanczos_ground_state<Kokkos::DefaultExecutionSpace>(
            H,
            200,
            1e-12,
            false // Single-pass
        );

    auto res2 =
        lanczos_ground_state<Kokkos::DefaultExecutionSpace>(
            H,
            200,
            1e-12,
            true // Two-pass
        );

    std::cout
        << "Energy (single-pass) = "
        << res1.energy
        << "\n";

    std::cout
        << "Energy (two-pass)    = "
        << res2.energy
        << "\n";

    if (std::abs(res1.energy - res2.energy) > 1e-10) {
        std::cerr << "Mismatch between single-pass and two-pass energy!\n";
        return 1;
    }

    return 0;
}
