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

    // Test Basis state lookups
    uint64_t s0 = qkrylov_basis_state(basis, 0);
    assert(qkrylov_basis_contains(basis, s0) == 1);
    assert(qkrylov_basis_index(basis, s0) == 0);

    // Test Sector set_n and set_nb functions
    assert(qkrylov_sector_set_n(sec, 2) == QKRYLOV_SUCCESS);
    assert(qkrylov_sector_set_nb(sec, 1) == QKRYLOV_SUCCESS);

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

    // Test 3-Body N-Body Term API on separate OpSum handle
    qkrylov_opsum_h opsum_nbody = qkrylov_opsum_create();
    assert(opsum_nbody != NULL);
    const char* ops3[3] = {"Sz", "Sz", "Sz"};
    int sites3[3] = {0, 1, 2};
    assert(qkrylov_opsum_add_term_nbody(opsum_nbody, 0.1f, 0.0f, 3, ops3, sites3) == QKRYLOV_SUCCESS);
    qkrylov_opsum_destroy(opsum_nbody);

    // 5. Test Device & Hamiltonian API
    int is_gpu = qkrylov_is_gpu_build();
    int gpus = qkrylov_gpu_count();
    const char* gpu_backend = qkrylov_find_gpu();
    std::cout << "Device check: is_gpu=" << is_gpu << ", gpu_count=" << gpus
              << ", backend=" << (gpu_backend ? gpu_backend : "none") << std::endl;
    assert(qkrylov_initialize_device("cpu") == QKRYLOV_SUCCESS);

    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);
    assert(H != NULL);
    assert(qkrylov_hamiltonian_dimension(H) == dim);

    qkrylov_hamiltonian_h H_dev = qkrylov_hamiltonian_create_device(basis, site, opsum, "cpu");
    assert(H_dev != NULL);
    assert(qkrylov_hamiltonian_dimension(H_dev) == dim);
    qkrylov_hamiltonian_destroy(H_dev);

    // 6. Test Matrix-Vector Apply
    std::vector<float> x_real(dim, 1.0);
    std::vector<float> x_imag(dim, 0.0);
    std::vector<float> y_real(dim, 0.0);
    std::vector<float> y_imag(dim, 0.0);

    int apply_res = qkrylov_hamiltonian_apply(H, x_real.data(), x_imag.data(), y_real.data(), y_imag.data());
    assert(apply_res == QKRYLOV_SUCCESS);

    // Test Zero-Copy Direct Complex Apply
    std::vector<std::complex<float>> x_cx(dim, std::complex<float>(1.0, 0.0));
    std::vector<std::complex<float>> y_cx(dim, std::complex<float>(0.0, 0.0));
    int apply_cx_res = qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const float*>(x_cx.data()), reinterpret_cast<float*>(y_cx.data()));
    assert(apply_cx_res == QKRYLOV_SUCCESS);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(y_cx[i].real() - y_real[i]) < 1e-12);
        assert(std::abs(y_cx[i].imag() - y_imag[i]) < 1e-12);
    }

    // Test Matrix-Free Diagonal Extraction
    std::vector<float> diag(dim);
    int diag_res = qkrylov_hamiltonian_diagonal(H, diag.data());
    assert(diag_res == QKRYLOV_SUCCESS);

    // 7. Test Lanczos Ground State Solver via C API (Energy & Eigenvector)
    qkrylov_lanczos_result_c_t lanczos_res;
    std::vector<std::complex<float>> psi_cx(dim);
    int solver_res = qkrylov_lanczos_ground_state_complex(H, 200, 1e-12, &lanczos_res, reinterpret_cast<float*>(psi_cx.data()));
    assert(solver_res == QKRYLOV_SUCCESS);

    std::cout << "C API Lanczos Ground State Energy: " << lanczos_res.energy << std::endl;
    assert(std::abs(lanczos_res.energy - (-1.6160254038)) < 1e-5);

    // Verify eigenvector normalization: ||psi||^2 == 1.0
    float norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += std::norm(psi_cx[i]);
    }
    assert(std::abs(norm_sq - 1.0f) < 1e-4);

    // 8. Test Davidson Solver via C API (Lowest 2 Eigenpairs)
    int n_eig = 2;
    std::vector<float> dav_evals(n_eig);
    std::vector<std::complex<float>> dav_evecs(n_eig * dim);
    qkrylov_davidson_result_c_t dav_info;
    int dav_status = qkrylov_davidson_lowest_complex(H, n_eig, 20, 1e-6f, dav_evals.data(), reinterpret_cast<float*>(dav_evecs.data()), &dav_info);
    assert(dav_status == QKRYLOV_SUCCESS);
    assert(dav_info.converged == 1);

    std::cout << "C API Davidson Lowest Eigenvalues: E0=" << dav_evals[0] << ", E1=" << dav_evals[1] << std::endl;
    assert(std::abs(dav_evals[0] - lanczos_res.energy) < 1e-4);
    assert(dav_evals[0] <= dav_evals[1]);

    // Verify H * psi_k == E_k * psi_k for each computed eigenpair
    for (int k = 0; k < n_eig; ++k) {
        const std::complex<float>* vk = dav_evecs.data() + (k * dim);
        std::vector<std::complex<float>> Hvk(dim);
        assert(qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const float*>(vk), reinterpret_cast<float*>(Hvk.data())) == QKRYLOV_SUCCESS);
        for (size_t i = 0; i < dim; ++i) {
            std::complex<float> expected = dav_evals[k] * vk[i];
            assert(std::abs(Hvk[i] - expected) < 1e-3);
        }
    }
    // 9. Test Dynamics & Spectral Function C API
    int n_iter = 20;
    std::vector<float> alphas(n_iter);
    std::vector<float> betas(n_iter);
    float norm_phi0 = 0.0f;
    int num_coeffs = 0;

    int dyn_status = qkrylov_continued_fraction_coeffs_complex(H, reinterpret_cast<const float*>(psi_cx.data()), n_iter, alphas.data(), betas.data(), &norm_phi0, &num_coeffs);
    assert(dyn_status == QKRYLOV_SUCCESS);
    assert(num_coeffs > 0);
    assert(std::abs(norm_phi0 - 1.0f) < 1e-3);

    float spec_val = qkrylov_evaluate_spectral_function(alphas.data(), betas.data(), num_coeffs, norm_phi0, 0.5f, lanczos_res.energy, 0.1f);
    std::cout << "C API Spectral function at w=0.5: " << spec_val << std::endl;
    assert(spec_val > 0.0f);

    // 10. Test FTLM C API
    qkrylov_ftlm_result_c_t ftlm_res;
    int ftlm_status = qkrylov_ftlm(H, 1.0f, 10, 20, &ftlm_res);
    assert(ftlm_status == QKRYLOV_SUCCESS);
    std::cout << "C API FTLM Result: Z=" << ftlm_res.partition_function << ", E=" << ftlm_res.internal_energy << ", Cv=" << ftlm_res.specific_heat << std::endl;
    assert(ftlm_res.partition_function > 0.0f);

    // Cleanup handles
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sec);

    // 11. Test Spin-S Site C API
    qkrylov_site_h spin1_site = qkrylov_site_create_spin_s(1.0);
    assert(spin1_site != NULL);
    qkrylov_site_h spinhalf_s_site = qkrylov_site_create_spin_s(0.5);
    assert(spinhalf_s_site != NULL);
    qkrylov_site_h spin32_site = qkrylov_site_create_spin_s(1.5);
    assert(spin32_site != NULL);

    // Invalid spin site parameters (S <= 0)
    assert(qkrylov_site_create_spin_s(0.0) == NULL);
    assert(qkrylov_site_create_spin_s(-1.0) == NULL);

    qkrylov_site_destroy(spinhalf_s_site);
    qkrylov_site_destroy(spin32_site);

    // 12. Test Spin-S Basis C API (Unconstrained & Sz-conserved)
    // Full basis: N = 2, S = 1.0 -> dimension = 3^2 = 9
    qkrylov_basis_h spin1_basis_full = qkrylov_basis_create_spin_s(2, 1.0, NULL);
    assert(spin1_basis_full != NULL);
    assert(qkrylov_basis_dimension(spin1_basis_full) == 9);
    assert(qkrylov_basis_nsites(spin1_basis_full) == 2);
    qkrylov_basis_destroy(spin1_basis_full);

    // Invalid basis parameters
    assert(qkrylov_basis_create_spin_s(0, 1.0, NULL) == NULL);
    assert(qkrylov_basis_create_spin_s(-2, 1.0, NULL) == NULL);
    assert(qkrylov_basis_create_spin_s(2, 0.0, NULL) == NULL);
    assert(qkrylov_basis_create_spin_s(2, -0.5, NULL) == NULL);

    // Sz-conserved basis: N = 4, S = 1.0, Sz = 0
    qkrylov_sector_h sec_spin1 = qkrylov_sector_create();
    assert(sec_spin1 != NULL);
    assert(qkrylov_sector_set_sz(sec_spin1, 0) == QKRYLOV_SUCCESS);

    int N_s1 = 4;
    qkrylov_basis_h spin1_basis = qkrylov_basis_create_spin_s(N_s1, 1.0, sec_spin1);
    assert(spin1_basis != NULL);
    uint64_t dim_s1 = qkrylov_basis_dimension(spin1_basis);
    std::cout << "Spin-1 Basis dimension for N=4, Sz=0: " << dim_s1 << std::endl;
    assert(dim_s1 == 19);
    assert(qkrylov_basis_nsites(spin1_basis) == 4);

    // Test state lookup in Spin-S basis
    uint64_t s1_state0 = qkrylov_basis_state(spin1_basis, 0);
    assert(qkrylov_basis_contains(spin1_basis, s1_state0) == 1);
    assert(qkrylov_basis_index(spin1_basis, s1_state0) == 0);

    // 13. Test Spin-1 Hamiltonian Setup
    qkrylov_opsum_h ops_s1 = qkrylov_opsum_create();
    assert(ops_s1 != NULL);
    for (int i = 0; i < N_s1 - 1; ++i) {
        assert(qkrylov_opsum_add_term_2body(ops_s1, 1.0f, 0.0f, "Sz", i, "Sz", i+1) == QKRYLOV_SUCCESS);
        assert(qkrylov_opsum_add_term_2body(ops_s1, 0.5f, 0.0f, "Sp", i, "Sm", i+1) == QKRYLOV_SUCCESS);
        assert(qkrylov_opsum_add_term_2body(ops_s1, 0.5f, 0.0f, "Sm", i, "Sp", i+1) == QKRYLOV_SUCCESS);
    }

    qkrylov_hamiltonian_h H_s1 = qkrylov_hamiltonian_create(spin1_basis, spin1_site, ops_s1);
    assert(H_s1 != NULL);
    assert(qkrylov_hamiltonian_dimension(H_s1) == dim_s1);

    // Compute ground state for Spin-1 chain
    qkrylov_lanczos_result_c_t lanczos_s1_res;
    std::vector<std::complex<float>> psi0_s1(dim_s1);
    int gs_status = qkrylov_lanczos_ground_state_complex(
        H_s1, 200, 1e-5f, &lanczos_s1_res, reinterpret_cast<float*>(psi0_s1.data())
    );
    assert(gs_status == QKRYLOV_SUCCESS);
    std::cout << "Spin-1 Lanczos: energy=" << lanczos_s1_res.energy
              << ", converged=" << lanczos_s1_res.converged
              << ", iters=" << lanczos_s1_res.iterations << std::endl;
    assert(lanczos_s1_res.converged == 1);
    std::cout << "Spin-1 N=4 Ground State Energy: " << lanczos_s1_res.energy << std::endl;

    // Apply local operator O = Sz_0 to create excitation state |Op_psi0>
    qkrylov_opsum_h ops_sz0 = qkrylov_opsum_create();
    assert(ops_sz0 != NULL);
    assert(qkrylov_opsum_add_term_1body(ops_sz0, 1.0f, 0.0f, "Sz", 0) == QKRYLOV_SUCCESS);
    qkrylov_hamiltonian_h H_sz0 = qkrylov_hamiltonian_create(spin1_basis, spin1_site, ops_sz0);
    assert(H_sz0 != NULL);

    std::vector<std::complex<float>> op_psi0(dim_s1);
    int apply_sz0_status = qkrylov_hamiltonian_apply_complex(
        H_sz0,
        reinterpret_cast<const float*>(psi0_s1.data()),
        reinterpret_cast<float*>(op_psi0.data())
    );
    assert(apply_sz0_status == QKRYLOV_SUCCESS);

    // 14. Test Correction Vector Spectroscopy Solver C API
    qkrylov_correction_vector_result_c_t cv_result;
    std::vector<std::complex<float>> cv_vec_out(dim_s1);
    float omega = 1.5f;
    float eta = 0.1f;
    int cv_status = qkrylov_solver_correction_vector(
        H_s1,
        reinterpret_cast<const float*>(op_psi0.data()),
        lanczos_s1_res.energy,
        omega,
        eta,
        500,
        1e-6f,
        &cv_result,
        reinterpret_cast<float*>(cv_vec_out.data())
    );
    assert(cv_status == QKRYLOV_SUCCESS);
    assert(cv_result.converged == 1);
    assert(cv_result.iterations > 0);
    assert(cv_result.spectral_function >= 0.0f);
    std::cout << "C API Correction Vector Result: converged=" << cv_result.converged
              << ", iters=" << cv_result.iterations
              << ", S(omega=" << omega << ")=" << cv_result.spectral_function << std::endl;

    // Check correction vector norm
    float cv_norm = 0.0f;
    for (size_t i = 0; i < dim_s1; ++i) {
        cv_norm += std::norm(cv_vec_out[i]);
    }
    assert(cv_norm > 0.0f);

    // Test with NULL output vector (only compute spectral function)
    qkrylov_correction_vector_result_c_t cv_result_no_vec;
    int cv_status2 = qkrylov_solver_correction_vector(
        H_s1,
        reinterpret_cast<const float*>(op_psi0.data()),
        lanczos_s1_res.energy,
        omega,
        eta,
        500,
        1e-6f,
        &cv_result_no_vec,
        NULL
    );
    assert(cv_status2 == QKRYLOV_SUCCESS);
    assert(cv_result_no_vec.converged == 1);
    assert(std::abs(cv_result_no_vec.spectral_function - cv_result.spectral_function) < 1e-4f);

    // Test error handling for correction vector solver
    assert(qkrylov_solver_correction_vector(NULL, reinterpret_cast<const float*>(op_psi0.data()), 0.0f, omega, eta, 100, 1e-6f, &cv_result, NULL) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_solver_correction_vector(H_s1, NULL, 0.0f, omega, eta, 100, 1e-6f, &cv_result, NULL) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_solver_correction_vector(H_s1, reinterpret_cast<const float*>(op_psi0.data()), 0.0f, omega, eta, 100, 1e-6f, NULL, NULL) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_solver_correction_vector(H_s1, reinterpret_cast<const float*>(op_psi0.data()), 0.0f, omega, eta, 0, 1e-6f, &cv_result, NULL) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_solver_correction_vector(H_s1, reinterpret_cast<const float*>(op_psi0.data()), 0.0f, omega, eta, -10, 1e-6f, &cv_result, NULL) == QKRYLOV_ERROR_INVALID_ARG);

    // Cleanup Spin-1 handles
    qkrylov_hamiltonian_destroy(H_sz0);
    qkrylov_opsum_destroy(ops_sz0);
    qkrylov_hamiltonian_destroy(H_s1);
    qkrylov_opsum_destroy(ops_s1);
    qkrylov_site_destroy(spin1_site);
    qkrylov_basis_destroy(spin1_basis);
    qkrylov_sector_destroy(sec_spin1);

    std::cout << "C API tests passed successfully!" << std::endl;
    return 0;
}
