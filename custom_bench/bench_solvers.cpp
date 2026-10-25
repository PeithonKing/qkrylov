// Timing and sanity values for the non-Lanczos solvers (Davidson, FTLM, real-time dynamics, correction vector).
// Same public API before and after the solver rewrite, so one source measures both.
//
//   bench_solvers_fp64 [L=18] [repeat=2] [device]   periodic Heisenberg chain, Sz = 0, double precision
//                                                   device: "cpu" (default, host threads) or "default" (the build's
//                                                   default execution space, i.e. the GPU in a CUDA/HIP/SYCL build)
//
// Each case is run `repeat` times and the fastest time is printed, together with values that must not change
// (eigenvalues, thermodynamics, correlators) so a speed-up cannot hide a wrong answer.
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/davidson.hpp>
#include <qkrylov/solvers/ftlm.hpp>
#include <qkrylov/solvers/dynamics.hpp>
#include <qkrylov/solvers/correction_vector.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;


static double time_ms(int repeat, const std::function<void()>& fn) {
    double best = 1e300;
    for (int r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

template <class Exec>
static void run(int L, int repeat) {
    {
        Sector sector;
        sector.sz = {0.0};
        auto basis = std::make_shared<SpinHalfBasis>(L, sector);
        auto site = std::make_shared<SpinHalfSite>();

        OpSum ops;
        for (int i = 0; i < L; ++i) {
            const int j = (i + 1) % L;
            ops.add_term(1.0 * Sz(i) * Sz(j));
            ops.add_term(0.5 * Sp(i) * Sm(j));
            ops.add_term(0.5 * Sm(i) * Sp(j));
        }
        MatrixFreeHamiltonian<Exec> H(basis, site, ops);
        const Index dim = H.dimension();

        OpSum o1; o1.add_term(1.0 * Sz(0) * Sz(1));
        MatrixFreeHamiltonian<Exec> O(basis, site, o1);
        OpSum o2; o2.add_term(1.0 * Sz(0));
        MatrixFreeHamiltonian<Exec> Sz0(basis, site, o2);

        std::printf("L = %d, dimension = %llu, repeat = %d\n", L, (unsigned long long)dim, repeat);
        std::printf("%-34s %12s   %s\n", "case", "time (ms)", "values (must not change)");

        // ---- Davidson --------------------------------------------------------------------------------
        for (int n_eig : {1, 3}) {
            DavidsonResult res;
            const double ms = time_ms(repeat, [&] { res = davidson_lowest(H, n_eig, 20, Real(1e-8)); });
            std::printf("%-34s %12.1f   iters=%d conv=%d  E=", ("davidson n_eig=" + std::to_string(n_eig)).c_str(), ms,
                        res.iterations, int(res.converged));
            for (Real e : res.eigenvalues) std::printf("%.9f ", double(e));
            std::printf("\n");
        }

        // ---- FTLM: thermodynamics only, then with one observable --------------------------------------
        const std::vector<Real> betas = {0.5, 1.0, 2.0, 5.0};
        {
            FTLMSweepResult res;
            const double ms = time_ms(repeat, [&] { res = ftlm_sweep<Exec>(H, betas, {}, 4, 40, 7); });
            std::printf("%-34s %12.1f   E(beta=1)=%.6f  Cv(beta=1)=%.6f  S(beta=5)=%.6f\n", "ftlm_sweep thermo (R=4, M=40)", ms,
                        double(res.internal_energies[1]), double(res.specific_heats[1]), double(res.entropies[3]));
        }
        {
            FTLMSweepResult res;
            const double ms = time_ms(repeat, [&] { res = ftlm_sweep<Exec>(H, betas, {O}, 4, 40, 7); });
            std::printf("%-34s %12.1f   <SzSz>(beta=1)=%.6f  <SzSz>(beta=5)=%.6f\n", "ftlm_sweep + 1 observable", ms,
                        double(res.observable_expectations[0][1].real()), double(res.observable_expectations[0][3].real()));
        }
        {
            std::vector<FTLMKrylovSample> samples;
            const double ms = time_ms(repeat, [&] { samples = ftlm_sample<Exec>(H, {O}, 4, 40, 7); });
            auto res = ftlm_evaluate_sweep(samples, betas, dim);
            std::printf("%-34s %12.1f   E(beta=1)=%.6f  <SzSz>(beta=1)=%.6f\n", "ftlm_sample cached (+1 observable)", ms,
                        double(res.internal_energies[1]), double(res.observable_expectations[0][1].real()));
        }

        // ---- real-time dynamics ------------------------------------------------------------------------
        const std::vector<Real> times = {0.0, 0.5, 1.0, 2.0};
        {
            FTLMDynamicsResult res;
            const double ms = time_ms(repeat, [&] { res = ftlm_dynamics<Exec>(H, Real(1.0), Sz0, Sz0, times, 2, 30, 11); });
            std::printf("%-34s %12.1f   C(0)=%.6f  C(1)=%.6f  C(2)=%.6f\n", "ftlm_dynamics (R=2, M=30)", ms,
                        double(res.correlations[0].real()), double(res.correlations[2].real()), double(res.correlations[3].real()));
        }
        {
            std::mt19937_64 rng(5);
            std::normal_distribution<double> nd(0.0, 1.0);
            HostVector psi0(dim);
            for (auto& c : psi0) c = Complex(nd(rng), nd(rng));
            RealTimeResult res;
            const double ms = time_ms(repeat, [&] { res = time_evolve<Exec>(H, psi0, times, {O}, 40); });
            std::printf("%-34s %12.1f   survival(1)=%.6f%+.6fi  <SzSz>(2)=%.6f\n", "time_evolve (M=40, +1 observable)", ms,
                        double(res.survival_probabilities[2].real()), double(res.survival_probabilities[2].imag()),
                        double(res.observable_expectations[0][3].real()));
        }

        // ---- correction vector -------------------------------------------------------------------------
        {
            std::mt19937_64 rng(9);
            std::normal_distribution<double> nd(0.0, 1.0);
            HostVector rhs(dim);
            for (auto& c : rhs) c = Complex(nd(rng), 0.0);
            CorrectionVectorResult res;
            const double ms = time_ms(repeat, [&] { res = correction_vector_spectral<Exec>(H, rhs, Real(-0.44) * L, Real(1.0), Real(0.3), 400, Real(1e-8)); });
            std::printf("%-34s %12.1f   iters=%d conv=%d  S(omega)=%.8f\n", "correction_vector (real rhs)", ms, res.iterations,
                        int(res.converged), double(res.spectral_function));
        }
    }
}

int main(int argc, char** argv) {
    const int L = argc > 1 ? std::atoi(argv[1]) : 18;
    const int repeat = argc > 2 ? std::atoi(argv[2]) : 2;
    const std::string device = argc > 3 ? argv[3] : "cpu";
    Kokkos::initialize(argc, argv);
    if (device == "default") run<Kokkos::DefaultExecutionSpace>(L, repeat);
    else run<Kokkos::DefaultHostExecutionSpace>(L, repeat);
    Kokkos::finalize();
    return 0;
}
