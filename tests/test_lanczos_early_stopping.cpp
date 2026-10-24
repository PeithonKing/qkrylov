// early_stopping = false must run exactly maxiter steps (same energy, no premature exit);
// early_stopping = true (default) must stop earlier once converged.
// real_output must deliver a real ground state in eigenvector_real (and nothing in eigenvector).
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

template <typename Policy>
static void check(const MatrixFreeHamiltonian<Kokkos::DefaultHostExecutionSpace>& H, const char* name) {
    LanczosConfig cfg;
    cfg.maxiter = 40;
    cfg.tol = (sizeof(Real) == 4) ? Real(1e-4) : Real(1e-10);

    auto early = solvers::lanczos<Policy>(H, cfg);
    assert(early.converged && early.iterations < cfg.maxiter);

    cfg.early_stopping = false;
    auto full = solvers::lanczos<Policy>(H, cfg);
    assert(full.iterations == cfg.maxiter);
    assert(!full.converged);
    assert(full.alphas.size() == static_cast<size_t>(cfg.maxiter));

    const Real tol = (sizeof(Real) == 4) ? Real(1e-3) : Real(1e-8);
    assert(std::abs(full.energy - early.energy) < tol);
    std::cout << name << ": early stop after " << early.iterations << ", full run " << full.iterations
              << ", E0=" << full.energy << "\n";
}

template <typename Policy>
static void check_real_output(const MatrixFreeHamiltonian<Kokkos::DefaultHostExecutionSpace>& H, const char* name) {
    LanczosConfig cfg;
    cfg.maxiter = 60;
    cfg.tol = (sizeof(Real) == 4) ? Real(1e-5) : Real(1e-11);
    auto plain = solvers::lanczos<Policy>(H, cfg);
    cfg.real_output = true;
    auto real = solvers::lanczos<Policy>(H, cfg);
    assert(H.has_real_coefficients());
    assert(real.eigenvector.empty() && real.eigenvectors.empty());
    assert(real.eigenvector_real.size() == static_cast<size_t>(H.dimension()));
    assert(plain.eigenvector_real.empty() && plain.eigenvector.size() == real.eigenvector_real.size());
    double ov = 0.0, n1 = 0.0, n2 = 0.0;
    for (size_t i = 0; i < real.eigenvector_real.size(); ++i) {
        const double a = plain.eigenvector[i].real(), b = real.eigenvector_real[i];
        assert(std::abs(plain.eigenvector[i].imag()) < 1e-12);
        ov += a * b; n1 += a * a; n2 += b * b;
    }
    assert(std::abs(std::abs(ov) / std::sqrt(n1 * n2) - 1.0) < 1e-6);
    std::cout << name << ": real_output ok (overlap " << std::abs(ov) / std::sqrt(n1 * n2) << ")\n";
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const int L = 12;
        Sector sec; sec.sz = {0};
        auto basis = std::make_shared<SpinHalfBasis>(L, sec);
        auto site = std::make_shared<SpinHalfSite>();
        OpSum ops;
        for (int i = 0; i < L; ++i) {
            const int j = (i + 1) % L;
            ops.add_term(OperatorTerm(1.0, {{"Sz", i}, {"Sz", j}}));
            ops.add_term(OperatorTerm(0.5, {{"Sp", i}, {"Sm", j}}));
            ops.add_term(OperatorTerm(0.5, {{"Sm", i}, {"Sp", j}}));
        }
        MatrixFreeHamiltonian<Kokkos::DefaultHostExecutionSpace> H(basis, site, ops);
        check<solvers::policy::OnePass>(H, "OnePass");
        check<solvers::policy::OnePass_full>(H, "OnePass_full");
        check<solvers::policy::TwoPass>(H, "TwoPass");
        check_real_output<solvers::policy::OnePass_full>(H, "OnePass_full");
        check_real_output<solvers::policy::TwoPass>(H, "TwoPass");
    }
    Kokkos::finalize();
    std::cout << "test_lanczos_early_stopping passed\n";
    return 0;
}
