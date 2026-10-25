#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <qkrylov/core/kokkos_types.hpp>

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <cstdlib>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

struct StepTimes {
    double basis = 0.0;
    double ham = 0.0;
    double solver_setup = 0.0;
    double execution = 0.0;
    double total = 0.0;
};

template <typename ExecSpace>
StepTimes single_run(int L, int passes) {
    StepTimes t;
    
    // Step 1: Basis
    auto t0 = std::chrono::high_resolution_clock::now();
    double sz_val = (L % 2 == 0) ? 0.0 : 0.5;
    Sector s;
    s.sz.push_back(sz_val);
    auto basis = std::make_shared<SpinHalfBasis>(L, s);
    auto t1 = std::chrono::high_resolution_clock::now();
    t.basis = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Step 2: Hamiltonian Setup
    t0 = std::chrono::high_resolution_clock::now();
    auto site = std::make_shared<SpinHalfSite>();
    OpSum ops;
    for (int i = 0; i < L; ++i) {
        int j = (i + 1) % L;
        ops.add_term(1.0 * Sz(i) * Sz(j));
        ops.add_term(0.5 * Sp(i) * Sm(j));
        ops.add_term(0.5 * Sm(i) * Sp(j));
    }
    MatrixFreeHamiltonian<ExecSpace> H(basis, site, ops);
    t1 = std::chrono::high_resolution_clock::now();
    t.ham = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Step 3: Solver Setup
    t0 = std::chrono::high_resolution_clock::now();
    LanczosConfig cfg;
    cfg.maxiter = 200;
    cfg.tol = 1e-8;
    t1 = std::chrono::high_resolution_clock::now();
    t.solver_setup = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Step 4: Execution
    t0 = std::chrono::high_resolution_clock::now();
    if (passes == 0) {            // energy only: 3 vectors, no eigenvector
        auto res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);
    } else if (passes == 1) {     // stores every Krylov vector
        auto res = solvers::lanczos<solvers::policy::OnePass_full>(H, cfg);
    } else {                      // 3 vectors in RAM + replay for the eigenvector
        auto res = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
    }
    t1 = std::chrono::high_resolution_clock::now();
    t.execution = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    t.total = t.basis + t.ham + t.solver_setup + t.execution;
    return t;
}

template <typename ExecSpace>
void benchmark_loop(int L, int passes, int repeat) {
    // Warmup run (matches remote_benchmark.py line 62)
    single_run<ExecSpace>(L, passes);
    
    std::vector<double> basis_list, ham_list, solver_list, exec_list, total_list;
    for (int r = 0; r < repeat; ++r) {
        StepTimes t = single_run<ExecSpace>(L, passes);
        basis_list.push_back(t.basis);
        ham_list.push_back(t.ham);
        solver_list.push_back(t.solver_setup);
        exec_list.push_back(t.execution);
        total_list.push_back(t.total);
    }
    
    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    
    std::cout << mean(basis_list) << "|"
              << mean(ham_list) << "|"
              << mean(solver_list) << "|"
              << mean(exec_list) << "|"
              << mean(total_list) << "\n";
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        int L = 20;
        int passes = 1;
        std::string dev_str = "cpu";
        int repeat = 5;
        
        if (argc > 1) L = std::atoi(argv[1]);
        if (argc > 2) passes = std::atoi(argv[2]);
        if (argc > 3) dev_str = argv[3];
        if (argc > 4) repeat = std::atoi(argv[4]);
        
#if defined(KOKKOS_ENABLE_CUDA)
        if (dev_str == "gpu") {
            benchmark_loop<Kokkos::Cuda>(L, passes, repeat);
        } else {
            benchmark_loop<Kokkos::DefaultHostExecutionSpace>(L, passes, repeat);
        }
#else
        benchmark_loop<Kokkos::DefaultHostExecutionSpace>(L, passes, repeat);
#endif
    }
    Kokkos::finalize();
    return 0;
}
