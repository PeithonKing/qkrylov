// Exact-reference checks of the stochastic and dynamics solvers on a 70-dimensional problem whose complete
// spectrum is known: FTLM thermodynamics and observables (streamed, cached, both observable strategies),
// pure-state real-time evolution, the finite-temperature correlator and the correction vector.
//
// The reference is the full eigensystem from Lanczos with full reorthogonalisation (dimension 70, no degeneracies:
// open chain with distinct bonds). FTLM is a stochastic estimator, so it is compared with the exact traces within the
// statistical error of the (fixed-seed, hence reproducible) sample.
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/operators/local_op.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

using Exec = Kokkos::DefaultExecutionSpace;
using H_t = MatrixFreeHamiltonian<Exec>;

namespace {

constexpr bool kSingle = sizeof(Real) == 4;
const double kNum = kSingle ? 2e-3 : 1e-8;     // deterministic quantities (time evolution, correction vector)

OpSum chain(int L, bool complex_terms) {
    OpSum ops;
    for (int i = 0; i + 1 < L; ++i) {
        const double J = 1.0 + 0.13 * i;
        ops.add_term(J * Sz(i) * Sz(i + 1));
        ops.add_term(0.5 * J * Sp(i) * Sm(i + 1));
        ops.add_term(0.5 * J * Sm(i) * Sp(i + 1));
        if (complex_terms) {   // Dzyaloshinskii-Moriya: complex Hermitian
            OperatorTerm a; a.coeff = ComplexDouble(0.3, 0.0);  a.factors.push_back({"Sx", i}); a.factors.push_back({"Sy", i + 1}); ops.add_term(a);
            OperatorTerm b; b.coeff = ComplexDouble(-0.3, 0.0); b.factors.push_back({"Sy", i}); b.factors.push_back({"Sx", i + 1}); ops.add_term(b);
        }
    }
    return ops;
}

struct Exact {
    int D = 0;
    std::vector<double> E;
    std::vector<HostVector> psi;      // eigenvectors
};

Exact full_spectrum(const H_t& H) {
    LanczosConfig cfg;
    cfg.maxiter = static_cast<int>(H.dimension());
    cfg.n_eig = static_cast<int>(H.dimension());
    cfg.early_stopping = false;
    const auto r = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);
    Exact ex;
    ex.D = static_cast<int>(r.eigenvalues.size());
    assert(ex.D == static_cast<int>(H.dimension()));
    for (int n = 0; n < ex.D; ++n) ex.E.push_back(r.eigenvalues[n]);
    ex.psi = r.eigenvectors;
    return ex;
}

using CD = std::complex<double>;   // the references are computed in double whatever the library precision
CD to_cd(const Complex& c) { return CD(c.real(), c.imag()); }

CD braket(const HostVector& a, const HostVector& b) {
    CD s(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i) s += std::conj(to_cd(a[i])) * to_cd(b[i]);
    return s;
}

// matrix elements <m|O|n> in the eigenbasis
std::vector<std::vector<CD>> eigen_matrix(const H_t& O, const Exact& ex) {
    std::vector<std::vector<CD>> M(ex.D, std::vector<CD>(ex.D));
    HostVector tmp(ex.psi[0].size());
    for (int n = 0; n < ex.D; ++n) {
        O.apply(ex.psi[n].data(), tmp.data());
        for (int m = 0; m < ex.D; ++m) M[m][n] = braket(ex.psi[m], tmp);
    }
    return M;
}

struct Thermo { double U, Cv, O; };
Thermo exact_thermo(const Exact& ex, const std::vector<std::vector<CD>>& O, double beta) {
    double Z = 0, E1 = 0, E2 = 0, Ob = 0;
    const double E0 = ex.E[0];
    for (int n = 0; n < ex.D; ++n) {
        const double w = std::exp(-beta * (ex.E[n] - E0));
        Z += w; E1 += w * ex.E[n]; E2 += w * ex.E[n] * ex.E[n]; Ob += w * O[n][n].real();
    }
    const double U = E1 / Z;
    return {U, beta * beta * (E2 / Z - U * U), Ob / Z};
}

void check_ftlm(const H_t& H, const H_t& O, const Exact& ex, const std::vector<std::vector<CD>>& Omat) {
    const int R = 1500;
    const int M = static_cast<int>(H.dimension());   // full Krylov space: only the random-vector error remains
    const std::vector<Real> few = {Real(0.5), Real(2.0), Real(6.0)};

    // few temperatures (< M): observables evaluated directly; many temperatures (>= M): by projection.
    std::vector<Real> many;
    for (int i = 0; i < M + 5; ++i) many.push_back(Real(0.1 + 0.08 * i));
    many[3] = few[0]; many[7] = few[1]; many[11] = few[2];

    const auto a = ftlm_sweep<Exec>(H, few, {O}, R, M, 17);
    const auto b = ftlm_sweep<Exec>(H, many, {O}, R, M, 17);
    const auto c = ftlm_evaluate_sweep(ftlm_sample<Exec>(H, {O}, R, M, 17), few, H.dimension());
    const auto again = ftlm_sweep<Exec>(H, few, {O}, R, M, 17);
    const auto other = ftlm_sweep<Exec>(H, few, {O}, R, M, 18);

    const double tolU = 0.04, tolCv = 0.12, tolO = 0.01;   // a few standard errors of R = 1500 random vectors
    double worst = 0;
    for (size_t i = 0; i < few.size(); ++i) {
        const auto ex_t = exact_thermo(ex, Omat, few[i]);
        const double dU = std::abs(double(a.internal_energies[i]) - ex_t.U);
        const double dC = std::abs(double(a.specific_heats[i]) - ex_t.Cv);
        const double dO = std::abs(double(a.observable_expectations[0][i].real()) - ex_t.O);
        std::cout << "  beta=" << few[i] << "  U " << a.internal_energies[i] << " (exact " << ex_t.U << ")  Cv " << a.specific_heats[i]
                  << " (" << ex_t.Cv << ")  <O> " << a.observable_expectations[0][i].real() << " (" << ex_t.O << ")" << std::endl;
        assert(dU < tolU && dC < tolCv && dO < tolO);
        worst = std::max({worst, dU / tolU, dC / tolCv, dO / tolO});

        // streamed (direct observables), streamed (projection), cached: the same random vectors, the same numbers
        const size_t j = (i == 0) ? 3 : (i == 1 ? 7 : 11);
        assert(std::abs(a.internal_energies[i] - b.internal_energies[j]) < 1e-3 * (kSingle ? 10 : 1e-6));
        assert(std::abs(to_cd(a.observable_expectations[0][i]) - to_cd(b.observable_expectations[0][j])) < (kSingle ? 1e-3 : 1e-9));
        assert(std::abs(to_cd(a.observable_expectations[0][i]) - to_cd(c.observable_expectations[0][i])) < (kSingle ? 1e-3 : 1e-9));
        assert(std::abs(a.internal_energies[i] - c.internal_energies[i]) < (kSingle ? 1e-3 : 1e-9));
        // reproducible; a different seed gives a different (but equally valid) estimate
        assert(a.internal_energies[i] == again.internal_energies[i]);
        assert(a.internal_energies[i] != other.internal_energies[i]);
    }
    std::cout << "FTLM: worst error = " << worst << " of the tolerance" << std::endl;
}

void check_time_evolve(const H_t& H, const H_t& O, const Exact& ex, const std::vector<std::vector<CD>>& Omat) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> nd(0.0, 1.0);
    HostVector psi0(H.dimension());
    double nrm = 0;
    for (auto& c : psi0) { c = Complex(nd(rng), nd(rng)); nrm += std::norm(c); }
    for (auto& c : psi0) c /= std::sqrt(nrm);

    std::vector<CD> a(ex.D);   // <n|psi0>
    for (int n = 0; n < ex.D; ++n) a[n] = braket(ex.psi[n], psi0);

    const int M = static_cast<int>(H.dimension());
    // 4 times (< M: direct observable evaluation) and 90 times (> M: projection)
    for (int nt : {4, 90}) {
        std::vector<Real> times;
        for (int i = 0; i < nt; ++i) times.push_back(Real(0.37 * i));
        const auto res = time_evolve<Exec>(H, psi0, times, {O}, M);
        double worst = 0;
        for (int i = 0; i < nt; ++i) {
            const double t = times[i];
            CD surv(0.0, 0.0), obs(0.0, 0.0);
            std::vector<CD> ct(ex.D);
            for (int n = 0; n < ex.D; ++n) ct[n] = a[n] * std::exp(CD(0.0, -ex.E[n] * t));
            for (int n = 0; n < ex.D; ++n) surv += std::norm(a[n]) * std::exp(CD(0.0, -ex.E[n] * t));
            for (int m = 0; m < ex.D; ++m)
                for (int n = 0; n < ex.D; ++n) obs += std::conj(ct[m]) * Omat[m][n] * ct[n];
            worst = std::max({worst, std::abs(to_cd(res.survival_probabilities[i]) - surv), std::abs(to_cd(res.observable_expectations[0][i]) - obs)});
        }
        std::cout << "  time_evolve, " << nt << " times: worst deviation " << worst << std::endl;
        assert(worst < kNum * 50);
    }
}

void check_ftlm_dynamics(const H_t& H, const H_t& A, const Exact& ex, const std::vector<std::vector<CD>>& Amat) {
    const double beta = 1.0;
    const std::vector<Real> times = {Real(0.0), Real(0.5), Real(1.5), Real(3.0)};
    const auto res = ftlm_dynamics<Exec>(H, Real(beta), A, A, times, 1500, static_cast<int>(H.dimension()), 5);
    double Z = 0;
    const double E0 = ex.E[0];
    for (int n = 0; n < ex.D; ++n) Z += std::exp(-beta * (ex.E[n] - E0));
    double worst = 0;
    for (size_t ti = 0; ti < times.size(); ++ti) {
        // C(t) = (1/Z) sum_{n,m} exp(-beta E_n) exp(i (E_n - E_m) t) A_nm A_mn
        CD C(0.0, 0.0);
        for (int n = 0; n < ex.D; ++n)
            for (int m = 0; m < ex.D; ++m)
                C += std::exp(-beta * (ex.E[n] - E0)) * std::exp(CD(0.0, (ex.E[n] - ex.E[m]) * double(times[ti]))) * Amat[n][m] * Amat[m][n];
        C /= Z;
        const double dev = std::abs(to_cd(res.correlations[ti]) - C);
        std::cout << "  C(t=" << times[ti] << ") = " << res.correlations[ti] << " (exact " << C << ")" << std::endl;
        worst = std::max(worst, dev);
    }
    assert(std::abs(res.correlations[0].real() - 0.25) < (kSingle ? 1e-4 : 1e-9));   // <Sz^2> = 1/4 for every random vector
    assert(worst < 0.01);
}

void check_correction_vector(const H_t& H, const H_t& O, const Exact& ex) {
    const Real E0 = static_cast<Real>(ex.E[0]), omega = Real(0.8), eta = Real(0.4);
    // right-hand side: Sz_0 acting on a fixed vector
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nd(0.0, 1.0);
    HostVector rhs(H.dimension());
    for (auto& c : rhs) c = Complex(nd(rng), 0.0);
    std::vector<CD> c(ex.D);
    for (int n = 0; n < ex.D; ++n) c[n] = braket(ex.psi[n], rhs);
    const double s = E0 + omega;
    double exact = 0;
    for (int n = 0; n < ex.D; ++n) exact += eta * std::norm(c[n]) / ((ex.E[n] - s) * (ex.E[n] - s) + double(eta) * eta);
    exact /= 3.141592653589793;

    const auto real_rhs = correction_vector_spectral<Exec>(H, rhs, E0, omega, eta, 600, Real(kSingle ? 1e-5 : 1e-11));
    assert(real_rhs.converged);
    std::cout << "  correction vector: S = " << real_rhs.spectral_function << " (exact " << exact << ")" << std::endl;
    assert(std::abs(double(real_rhs.spectral_function) - exact) < (kSingle ? 1e-3 : 1e-8) * std::max(1.0, exact));

    // the same problem with a (tiny) imaginary part forces the complex code path; the answers must agree
    HostVector rhs_c = rhs;
    rhs_c[0] += Complex(0.0, 1e-30);
    const auto cplx = correction_vector_spectral<Exec>(H, rhs_c, E0, omega, eta, 600, Real(kSingle ? 1e-5 : 1e-11));
    assert(cplx.converged);
    assert(std::abs(double(cplx.spectral_function) - double(real_rhs.spectral_function)) < (kSingle ? 1e-3 : 1e-9) * std::max(1.0, exact));
    assert(cplx.iterations == real_rhs.iterations || kSingle);
    (void)O;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        Sector sector;
        sector.sz = {0.0};
        auto basis = std::make_shared<SpinHalfBasis>(8, sector);   // dimension 70
        auto site = std::make_shared<SpinHalfSite>();
        OpSum szsz; szsz.add_term(1.0 * Sz(0) * Sz(1));
        OpSum sz0;  sz0.add_term(1.0 * Sz(0));

        for (bool complex_terms : {false, true}) {
            std::cout << (complex_terms ? "complex" : "real") << " Hamiltonian" << std::endl;
            H_t H(basis, site, chain(8, complex_terms));
            H_t O(basis, site, szsz);
            H_t A(basis, site, sz0);
            const auto ex = full_spectrum(H);
            const auto Omat = eigen_matrix(O, ex);
            const auto Amat = eigen_matrix(A, ex);
            check_time_evolve(H, O, ex, Omat);
            check_correction_vector(H, O, ex);
            if (!complex_terms) {
                check_ftlm(H, O, ex, Omat);
                check_ftlm_dynamics(H, A, ex, Amat);
            }
        }
    }
    Kokkos::finalize();
    std::cout << "test_solvers_exact passed" << std::endl;
    return 0;
}
