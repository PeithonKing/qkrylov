// Ground state of the periodic Heisenberg chain, Sz = 0 sector (C++).
// The same problem is solved in examples/quickstart.py (Python) and examples/quickstart.jl (Julia).
//
//     cmake -B build -DQKRYLOV_BUILD_EXAMPLES=ON && cmake --build build --target quickstart
//     ./build/quickstart [L]
//
// Precision comes from the namespace (qkrylov::fp64 here, qkrylov::fp32 for single precision) and the device
// from the Hamiltonian's execution space (Kokkos::DefaultHostExecutionSpace = CPU, Kokkos::Cuda = GPU).
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>

#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main(int argc, char** argv) {
    const int L = argc > 1 ? std::atoi(argv[1]) : 20;
    Kokkos::initialize(argc, argv);
    {
        Sector sector;
        sector.sz = {0.0};
        auto basis = std::make_shared<SpinHalfBasis>(L, sector);   // describes the sector only: no states are built yet
        auto site = std::make_shared<SpinHalfSite>();

        OpSum ops;
        for (int i = 0; i < L; ++i) {
            const int j = (i + 1) % L;
            ops.add_term(1.0 * Sz(i) * Sz(j));
            ops.add_term(0.5 * Sp(i) * Sm(j));
            ops.add_term(0.5 * Sm(i) * Sp(j));
        }

        MatrixFreeHamiltonian<Kokkos::DefaultHostExecutionSpace> H(basis, site, ops);   // Kokkos::Cuda for the GPU
        LanczosConfig cfg;
        cfg.real_output = true;                                    // real H -> real eigenvector
        auto res = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);

        double norm2 = 0.0;
        for (double c : res.eigenvector_real) norm2 += c * c;
        std::printf("L = %d, device = cpu, dimension = %llu\n", L, (unsigned long long)basis->size());
        std::printf("ground-state energy  E0 = %.10f   (E0/L = %.8f)\n", double(res.energy), double(res.energy) / L);
        std::printf("state number 5 as bits: %s   |psi| = %.6f\n",
                    std::bitset<64>(basis->state(5)).to_string().substr(64 - L).c_str(), std::sqrt(norm2));
    }
    Kokkos::finalize();
    return 0;
}
