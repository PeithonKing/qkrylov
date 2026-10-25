// Davidson against an independent solver (Lanczos with full reorthogonalisation), for a real and a complex
// Hamiltonian, several numbers of states, with residual checks on the returned eigenvectors and the
// small-dimension corner cases of the subspace bookkeeping.
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/operators/local_op.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

using Exec = Kokkos::DefaultExecutionSpace;

namespace {

constexpr bool kSingle = sizeof(Real) == 4;
const double kEigTol = kSingle ? 2e-3 : 1e-8;         // agreement of eigenvalues
const Real kDavidsonTol = kSingle ? Real(1e-4) : Real(1e-8);
const double kResidualTol = kSingle ? 5e-3 : 5e-7;     // ||H v - lambda v|| of the returned vectors

// Open chain with distinct bond strengths: no degeneracies, no reflection symmetry. With `complex_terms` a
// Dzyaloshinskii-Moriya term D (SxSy - SySx) makes the matrix complex Hermitian (it is (iD/2)(S+S- - S-S+)).
OpSum chain(int L, bool complex_terms) {
    OpSum ops;
    for (int i = 0; i + 1 < L; ++i) {
        const double J = 1.0 + 0.13 * i;
        ops.add_term(J * Sz(i) * Sz(i + 1));
        ops.add_term(0.5 * J * Sp(i) * Sm(i + 1));
        ops.add_term(0.5 * J * Sm(i) * Sp(i + 1));
        if (complex_terms) {
            OperatorTerm a;
            a.coeff = ComplexDouble(0.3, 0.0);
            a.factors.push_back({"Sx", i});
            a.factors.push_back({"Sy", i + 1});
            ops.add_term(a);
            OperatorTerm b;
            b.coeff = ComplexDouble(-0.3, 0.0);
            b.factors.push_back({"Sy", i});
            b.factors.push_back({"Sx", i + 1});
            ops.add_term(b);
        }
    }
    return ops;
}

double residual_norm(const MatrixFreeHamiltonian<Exec>& H, const HostVector& v, Real lambda) {
    HostVector hv(v.size());
    H.apply(v.data(), hv.data());
    double s = 0.0, nrm = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        s += std::norm(hv[i] - lambda * v[i]);
        nrm += std::norm(v[i]);
    }
    assert(std::abs(nrm - 1.0) < (kSingle ? 1e-3 : 1e-8));   // returned vectors are normalised
    return std::sqrt(s);
}

void check_against_lanczos(int L, bool complex_terms, int n_eig, int max_subspace) {
    Sector sector;
    sector.sz = {0.0};
    auto basis = std::make_shared<SpinHalfBasis>(L, sector);
    auto site = std::make_shared<SpinHalfSite>();
    MatrixFreeHamiltonian<Exec> H(basis, site, chain(L, complex_terms));
    assert(H.has_real_coefficients() == !complex_terms);   // exercises the real and the complex code path

    LanczosConfig ref_cfg;
    ref_cfg.maxiter = static_cast<int>(std::min<Index>(H.dimension(), 150));
    ref_cfg.n_eig = n_eig;
    ref_cfg.early_stopping = false;
    const auto ref = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, ref_cfg);
    assert(static_cast<int>(ref.eigenvalues.size()) >= n_eig);

    const auto res = davidson_lowest<Exec>(H, n_eig, max_subspace, kDavidsonTol);
    assert(res.converged);
    assert(static_cast<int>(res.eigenvalues.size()) == n_eig);
    assert(static_cast<int>(res.eigenvectors.size()) == n_eig);
    for (int k = 0; k < n_eig; ++k) {
        assert(std::abs(res.eigenvalues[k] - ref.eigenvalues[k]) < kEigTol);
        assert(residual_norm(H, res.eigenvectors[k], res.eigenvalues[k]) < kResidualTol);
    }
    // same input, same answer
    const auto again = davidson_lowest<Exec>(H, n_eig, max_subspace, kDavidsonTol);
    assert(again.iterations == res.iterations);
    for (int k = 0; k < n_eig; ++k) assert(again.eigenvalues[k] == res.eigenvalues[k]);

    // the iteration limit is honoured: one outer iteration is not enough, and says so
    assert(res.iterations > 1);
    const auto capped = davidson_lowest<Exec>(H, n_eig, max_subspace, kDavidsonTol, 1);
    assert(!capped.converged && capped.iterations == 1);
    bool rejected = false;
    try { davidson_lowest<Exec>(H, n_eig, max_subspace, kDavidsonTol, 0); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    std::cout << (complex_terms ? "complex" : "real   ") << " H, L=" << L << ", n_eig=" << n_eig
              << ": iterations=" << res.iterations << ", E0=" << res.eigenvalues[0] << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        for (int n_eig : {1, 2, 4}) check_against_lanczos(10, false, n_eig, 20);
        for (int n_eig : {1, 3}) check_against_lanczos(10, true, n_eig, 20);

        // A subspace limit that forces many restarts, and one close to the minimum allowed (2 * n_eig).
        check_against_lanczos(10, false, 3, 8);
        check_against_lanczos(10, true, 2, 6);

        // Dimension (6) smaller than the requested subspace and than 2 * n_eig.
        check_against_lanczos(4, false, 1, 20);
        check_against_lanczos(4, false, 4, 20);
        check_against_lanczos(4, true, 3, 20);
    }
    Kokkos::finalize();
    std::cout << "test_davidson_exact passed\n";
    return 0;
}
