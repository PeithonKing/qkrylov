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
    auto basis =
        std::make_shared<basis::SpinHalf>(
            2,
            basis::sector::Unconstrained{}
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

    LanczosConfig config{200, 1e-12};

    auto res1 =
        solvers::lanczos<solvers::policy::SinglePass>(
            H,
            config
        );

    auto res2 =
        solvers::lanczos<solvers::policy::TwoPass>(
            H,
            config
        );

    // Verify structured binding unpack
    auto [e, v] = solvers::lanczos<solvers::policy::TwoPass>(H, config);
    if (std::abs(e - res2.energy) > 1e-12) {
        std::cerr << "Structured binding energy mismatch!\n";
        return 1;
    }
    if (v.size() != res2.eigenvector.size()) {
        std::cerr << "Structured binding eigenvector size mismatch!\n";
        return 1;
    }

    std::cout
        << "Energy (single-pass) = "
        << res1.energy
        << "\n";

    std::cout
        << "Energy (two-pass)    = "
        << res2.energy
        << "\n";

    std::cout
        << "Energy (structured)  = "
        << e
        << "\n";

    if (std::abs(res1.energy - res2.energy) > 1e-10) {
        std::cerr << "Mismatch between single-pass and two-pass energy!\n";
        return 1;
    }

    return 0;
}
