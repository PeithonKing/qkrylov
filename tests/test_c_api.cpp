#include "qkrylov/c_api.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <complex>

int main() {
    std::cout << "Testing C API..." << std::endl;

    // 1. Test Sector API
    qkrylov_sector_h sec = qkrylov_sector_create();
    assert(sec != NULL);
    int res_sz = qkrylov_sector_set_sz(sec, 0);
    assert(res_sz == QKRYLOV_SUCCESS);

    // 2. Test Basis API (4-site SpinHalfBasis with total Sz = 0)
    int N = 4;
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sec);
    assert(basis != NULL);
    uint64_t dim = qkrylov_basis_dimension(basis);
    std::cout << "Basis dimension for N=4, Sz=0: " << dim << std::endl;
    assert(dim == 6); // 4 choose 2 = 6 states in Sz=0 sector
    assert(qkrylov_basis_nsites(basis) == 4);

    // 3. Test Site API
    qkrylov_site_h site = qkrylov_spinhalf_site_create();
    assert(site != NULL);

    // 4. Test OpSum API (4-site Heisenberg chain)
    qkrylov_opsum_h opsum = qkrylov_opsum_create();
    assert(opsum != NULL);

    for (int i = 0; i < N - 1; ++i) {
        // Sz_i Sz_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 1.0, 0.0, "Sz", i, "Sz", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sp_i Sm_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sp", i, "Sm", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sm_i Sp_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sm", i, "Sp", i+1) == QKRYLOV_SUCCESS);
    }

    // 5. Test Hamiltonian API
    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);
    assert(H != NULL);
    assert(qkrylov_hamiltonian_dimension(H) == dim);

    // 6. Test Matrix-Vector Apply
    std::vector<double> x_real(dim, 1.0);
    std::vector<double> x_imag(dim, 0.0);
    std::vector<double> y_real(dim, 0.0);
    std::vector<double> y_imag(dim, 0.0);

    int apply_res = qkrylov_hamiltonian_apply(H, x_real.data(), x_imag.data(), y_real.data(), y_imag.data());
    assert(apply_res == QKRYLOV_SUCCESS);

    // Test Zero-Copy Direct Complex Apply
    std::vector<std::complex<double>> x_cx(dim, std::complex<double>(1.0, 0.0));
    std::vector<std::complex<double>> y_cx(dim, std::complex<double>(0.0, 0.0));
    int apply_cx_res = qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const double*>(x_cx.data()), reinterpret_cast<double*>(y_cx.data()));
    assert(apply_cx_res == QKRYLOV_SUCCESS);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(y_cx[i].real() - y_real[i]) < 1e-12);
        assert(std::abs(y_cx[i].imag() - y_imag[i]) < 1e-12);
    }

    // 7. Test Lanczos Ground State Solver via C API
    qkrylov_lanczos_result_c_t lanczos_res;
    int solver_res = qkrylov_lanczos_ground_state(H, 200, 1e-12, &lanczos_res);
    assert(solver_res == QKRYLOV_SUCCESS);

    std::cout << "C API Lanczos Ground State Energy: " << lanczos_res.energy << std::endl;
    // Expected energy for 4-site Heisenberg chain is approx -1.6160254038
    assert(std::abs(lanczos_res.energy - (-1.6160254038)) < 1e-6);

    // Cleanup handles
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sec);

    std::cout << "C API tests passed successfully!" << std::endl;
    return 0;
}
