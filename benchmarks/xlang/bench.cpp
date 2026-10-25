// One cell of the cross-language benchmark (C++ side). See bench.py for the contract.
//   xlang_fp64 L MODE ITERS [cpu|cuda]   (precision is chosen by the executable: xlang_fp32 / xlang_fp64)
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>

#include <sys/resource.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;
using Clock = std::chrono::steady_clock;

static double g_rss_base = 0.0;
static double rss_mb() {
    rusage u; getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / 1024.0;
}
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template <typename Space>
static void run(int L, const std::string& mode, int iters, bool report, double& rss_base_out) {
    const auto t0 = Clock::now();
    Sector sec; sec.sz.push_back(0.0);
    auto basis = std::make_shared<SpinHalfBasis>(L, sec);
    auto site = std::make_shared<SpinHalfSite>();
    OpSum ops;
    for (int i = 0; i < L; ++i) {
        const int j = (i + 1) % L;
        ops.add_term(1.0 * Sz(i) * Sz(j));
        ops.add_term(0.5 * Sp(i) * Sm(j));
        ops.add_term(0.5 * Sm(i) * Sp(j));
    }
    MatrixFreeHamiltonian<Space> H(basis, site, ops);
    const auto t1 = Clock::now();

    LanczosConfig cfg;
    cfg.maxiter = iters; cfg.tol = Real(1e-12); cfg.early_stopping = false; cfg.real_output = true;
    LanczosResult res = (mode == "state") ? solvers::lanczos<solvers::policy::TwoPass>(H, cfg)
                                          : solvers::lanczos<solvers::policy::OnePass>(H, cfg);
    const auto t2 = Clock::now();
    rss_base_out = rss_mb();
    if (!report) return;

    double checksum = 0.0;
    for (size_t k = 0; k < res.eigenvector_real.size(); ++k) checksum += double(res.eigenvector_real[k]) * std::sin(double(k));
    for (size_t k = 0; k < res.eigenvector.size(); ++k) checksum += double(res.eigenvector[k].real()) * std::sin(double(k));

    std::printf("lang=cpp prec=%d mode=%s L=%d dim=%llu E0=%.12e iters=%d t_setup_ms=%.2f t_solve_ms=%.2f "
                "rss_base_mb=%.0f rss_peak_mb=%.0f checksum=%.10e\n",
                int(sizeof(Real) * 8), mode.c_str(), L, (unsigned long long)H.dimension(), double(res.energy),
                res.iterations, ms(t0, t1), ms(t1, t2), g_rss_base, rss_mb(), checksum);
}

template <typename Space>
static void warm_up_and_run(int L, const std::string& mode, int iters) {
    double dummy = 0.0;
    run<Space>(8, mode, 8, false, dummy);      // warm-up (Kokkos / CUDA context): outside the timings
    g_rss_base = rss_mb();
    run<Space>(L, mode, iters, true, dummy);
}

int main(int argc, char** argv) {
    const int L = std::atoi(argv[1]);
    const std::string mode = argv[2];
    const int iters = std::atoi(argv[3]);
    Kokkos::initialize(argc, argv);
    const std::string device = argc > 4 ? argv[4] : "cpu";
#if defined(KOKKOS_ENABLE_CUDA)
    if (device == "cuda") warm_up_and_run<Kokkos::Cuda>(L, mode, iters); else
#endif
    warm_up_and_run<Kokkos::DefaultHostExecutionSpace>(L, mode, iters);
    Kokkos::finalize();
    return 0;
}
