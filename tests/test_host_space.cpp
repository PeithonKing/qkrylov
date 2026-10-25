// A Hamiltonian on the CPU execution space must work in every solver even when the build's default
// execution space is a GPU (then Kokkos::DefaultHostExecutionSpace != Kokkos::DefaultExecutionSpace).
// In a CPU-only build both spaces coincide and this is a plain consistency test. Regression test for
// a Davidson kernel that was launched on the default space regardless of where its vectors lived.
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;
using HostSpace = Kokkos::DefaultHostExecutionSpace;
using DefSpace  = Kokkos::DefaultExecutionSpace;

template <typename Space>
static MatrixFreeHamiltonian<Space> make_H(int L) {
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
    return MatrixFreeHamiltonian<Space>(basis, site, ops);
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const int L = 8;
        const Real tol = (sizeof(Real) == 4) ? Real(2e-3) : Real(1e-8);
        auto Hh = make_H<HostSpace>(L);
        auto Hd = make_H<DefSpace>(L);

        LanczosConfig cfg; cfg.maxiter = 100; cfg.tol = (sizeof(Real) == 4) ? Real(1e-5) : Real(1e-11);
        const Real e_ref = solvers::lanczos<solvers::policy::OnePass>(Hd, cfg).energy;
        for (Real e : {solvers::lanczos<solvers::policy::OnePass>(Hh, cfg).energy,
                       solvers::lanczos<solvers::policy::OnePass_full>(Hh, cfg).energy,
                       solvers::lanczos<solvers::policy::TwoPass>(Hh, cfg).energy}) {
            assert(std::abs(e - e_ref) < tol);
        }

        // Davidson (the solver that used to launch its kernel on the default space)
        auto dav = davidson_lowest<HostSpace>(Hh, 1);
        assert(std::abs(dav.eigenvalues[0] - e_ref) < Real(1e-3));

        // real-time dynamics: <H>(t) is conserved
        HostVector psi0(Hh.dimension(), Complex(0.0, 0.0));
        psi0[0] = Complex(1.0, 0.0);
        std::vector<Real> times = {Real(0.0), Real(0.4), Real(0.8)};
        auto rt = time_evolve<HostSpace>(Hh, psi0, times, {Hh}, 20);
        const Real e0 = rt.observable_expectations[0][0].real();
        for (size_t k = 1; k < times.size(); ++k) {
            assert(std::abs(rt.observable_expectations[0][k].real() - e0) < Real(1e-2));
        }

        // apply agrees between the two spaces
        HostVector x(Hh.dimension()), yh(Hh.dimension()), yd(Hd.dimension());
        for (size_t i = 0; i < x.size(); ++i) x[i] = Complex(std::sin(0.3 * i), std::cos(0.7 * i));
        Hh.apply(x.data(), yh.data());
        Hd.apply(x.data(), yd.data());
        for (size_t i = 0; i < x.size(); ++i) assert(std::abs(yh[i] - yd[i]) < tol);

        std::cout << "host-space solvers OK (default space is "
                  << (std::is_same_v<HostSpace, DefSpace> ? "the host" : "a device") << ")\n";
    }
    Kokkos::finalize();
    return 0;
}
