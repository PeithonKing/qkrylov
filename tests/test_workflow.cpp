#undef NDEBUG
#include <cassert>
#include <iostream>
#include <memory>
#include <cmath>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/operators/local_op.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;

void test_heisenberg_workflow() {
    std::cout << "Running test_heisenberg_workflow..." << std::endl;

    // 4-site chain with Sz=0 sector (dimension = C(4,2) = 6)
    Sector sec;
    sec.use_sz = true;
    sec.sz2 = 0;  // 2*Sz = 0
    auto basis = std::make_shared<SpinHalfBasis>(4, sec);
    assert(basis->size() == 6 && "Sz=0 sector of 4-site chain should have dim 6");
    assert(basis->nsites() == 4 && "Basis should have 4 sites");

    auto site = std::make_shared<SpinHalfSite>();

    // Use the new algebraic expression template syntax
    OpSum os;
    for (int i = 0; i < 3; ++i) {
        os += 1.0 * Sz(i) * Sz(i+1) + 0.5 * Sp(i) * Sm(i+1) + 0.5 * Sm(i) * Sp(i+1);
    }
    assert(os.size() == 9 && "Should have 9 terms for 3 bonds");

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os);
    assert(H.dimension() == 6 && "Hamiltonian dimension should match Sz=0 sector size");

    auto res = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H, 200, 1e-12);

    // Exact ground state energy for 4-site Heisenberg chain (OBC) = 1 - sqrt(2) ≈ -0.6160254038
    // but restricted to Sz=0 sector the ground state is still -1.6160254038
    double exact_energy = -1.6160254038;
    std::cout << "Computed Energy: " << res.energy << " Exact: " << exact_energy << std::endl;
    assert(std::abs(res.energy - exact_energy) < 1e-5 && "Ground state energy should match exact 4-site Heisenberg value");
    assert(res.eigenvector.size() == 6 && "Eigenvector dimension should match Sz=0 sector size");

    std::cout << "test_heisenberg_workflow PASSED!" << std::endl;
}

int main() {
    test_heisenberg_workflow();
    std::cout << "All workflow tests passed successfully." << std::endl;
    return 0;
}
