#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "krylov_detail.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_steps
)
{
    const Index dim = H.dimension();
    if (dim == 0 || phi0.empty()) return {};

    Real norm_sq = Real(0.0);
    for (Index i = 0; i < phi0.size(); ++i) {
        norm_sq += std::norm(phi0[i]);
    }
    const Real norm_phi = std::sqrt(norm_sq);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (norm_phi < mach_eps) return { {}, {}, Real(0.0) };

    LanczosConfig cfg;
    cfg.maxiter = n_steps;
    cfg.min_iterations = n_steps;
    cfg.tol = Real(0.0);
    cfg.initial_vector = phi0;
    cfg.compute_eigenvectors = false;

    auto l_res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);

    DynamicsResult res;
    res.alphas = std::move(l_res.alphas);
    res.betas = std::move(l_res.betas);
    res.norm_phi0 = norm_phi;
    return res;
}

Real evaluate_spectral_function(
    const Real* alphas,
    const Real* betas,
    size_t n,
    Real norm_phi0,
    Real omega,
    Real E0,
    Real eta
)
{
    if (n == 0) return 0.0;

    std::complex<Real> z(omega + E0, eta);

    // Backward recursion for continued fraction
    // C_n = 1 / (z - a_n)
    // C_{i} = 1 / (z - a_i - b_i^2 * C_{i+1})

    std::complex<Real> f = 0.0;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        if (i == static_cast<int>(n) - 1) {
            f = Real(1.0) / (z - alphas[i]);
        } else {
            f = Real(1.0) / (z - alphas[i] - betas[i] * betas[i] * f);
        }
    }

    constexpr Real PI = static_cast<Real>(3.141592653589793238462643383279502884L);
    return -static_cast<Real>(1.0) / PI * std::imag(norm_phi0 * norm_phi0 * f);
}

template <typename ExecSpace>
RealTimeResult time_evolve(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& psi0,
    const std::vector<Real>& time_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_steps
)
{
    using namespace krylov_detail;
    using MemSpace = typename ExecSpace::memory_space;
    const Index dim = H.dimension();
    if (dim == 0 || psi0.empty() || time_grid.empty()) return {};

    Real norm_sq = Real(0.0);
    for (Index i = 0; i < psi0.size(); ++i) {
        norm_sq += std::norm(psi0[i]);
    }
    const Real norm_psi = std::sqrt(norm_sq);
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (norm_psi < mach_eps) return {};

    const int max_steps = clamp_to_dim(n_steps, dim);

    Vec<ExecSpace> v_prev = make_vector<ExecSpace>("te::v_prev", dim);
    Vec<ExecSpace> v_curr = make_vector<ExecSpace>("te::v_curr", dim);
    Vec<ExecSpace> w = make_vector<ExecSpace>("te::w", dim);

    copy_host_to_device(psi0, v_curr);
    scale_into(v_curr, v_curr, Real(1.0) / norm_psi);

    // The Krylov vectors are kept as the columns of one block: full reorthogonalisation (to keep the evolution
    // unitary) and the observable projections each cost one pass over the block, not one per vector.
    Blk<ExecSpace> basis = make_block<ExecSpace>("te::basis", dim, max_steps);
    linalg::CoefBuffer<KComplex, MemSpace> coef(max_steps);

    std::vector<Real> alphas;
    std::vector<Real> betas;
    alphas.reserve(max_steps);
    betas.reserve(max_steps);

    for (int j = 0; j < max_steps; ++j) {
        {
            Vec<ExecSpace> c = column<ExecSpace>(basis, j);
            Kokkos::deep_copy(c, v_curr);
        }

        const Real alpha = H.apply_dot(v_curr, w).real();
        alphas.push_back(alpha);
        lanczos_update_norm2(w, v_curr, v_prev, alpha, j > 0 ? betas.back() : Real(0.0), j > 0);

        // Full reorthogonalisation against every basis vector so far (twice: "twice is enough")
        linalg::block_orthogonalize<KComplex, MemSpace>(basis, j + 1, w, coef);

        const Real beta_val = norm(w);
        if (beta_val < mach_eps) break;
        if (j + 1 == max_steps) break;
        betas.push_back(beta_val);

        scale_into(v_prev, w, Real(1.0) / beta_val);
        std::swap(v_prev, v_curr);
    }

    const int M = static_cast<int>(alphas.size());
    auto eig = linalg::tridiag_eigensystem_full(alphas, betas, M);

    const size_t num_times = time_grid.size();
    const size_t num_obs = observables.size();

    // Observables: project onto the Krylov basis once (M applications), or, when there are fewer times than
    // Krylov vectors, evaluate <psi(t)|O|psi(t)> directly at each time (one application per time).
    const bool direct = num_obs > 0 && static_cast<int>(num_times) < M;
    std::vector<std::vector<Complex>> projected_obs(num_obs);
    if (!direct) {
        for (size_t oi = 0; oi < num_obs; ++oi) {
            projected_obs[oi] = project_between<ExecSpace>(observables[oi], basis, M, basis, M, w);
        }
    }
    Vec<ExecSpace> psi_t = direct ? make_vector<ExecSpace>("te::psi_t", dim) : Vec<ExecSpace>();

    RealTimeResult res;
    res.time_grid = time_grid;
    res.survival_probabilities.resize(num_times);
    res.observable_expectations.resize(num_obs, std::vector<Complex>(num_times, Complex(0.0, 0.0)));

    for (size_t ti = 0; ti < num_times; ++ti) {
        Real t = time_grid[ti];

        std::vector<Complex> c(M, Complex(0.0, 0.0));
        Complex overlap(0.0, 0.0);

        for (int m = 0; m < M; ++m) {
            Real theta = -eig.eigenvalues[m] * t;
            Complex phase(std::cos(theta), std::sin(theta));
            Real v0 = eig.eigenvectors[m][0];
            overlap += (v0 * v0) * phase;

            Complex factor = v0 * phase;
            for (int j = 0; j < M; ++j) {
                c[j] += factor * eig.eigenvectors[m][j];
            }
        }

        res.survival_probabilities[ti] = overlap;

        if (direct) {
            linalg::block_combine<KComplex, MemSpace>(basis, M, coef.set(c, M), psi_t);
            for (size_t oi = 0; oi < num_obs; ++oi) {
                observables[oi].apply(psi_t, w);
                const KComplex val = dot(psi_t, w);
                res.observable_expectations[oi][ti] = Complex(val.real(), val.imag());
            }
        } else {
            for (size_t oi = 0; oi < num_obs; ++oi) {
                const auto& mat = projected_obs[oi];
                Complex val(0.0, 0.0);
                for (int j = 0; j < M; ++j) {
                    for (int k = 0; k < M; ++k) {
                        val += std::conj(c[j]) * mat[j * M + k] * c[k];
                    }
                }
                res.observable_expectations[oi][ti] = val;
            }
        }
    }

    return res;
}

template <typename ExecSpace>
FTLMDynamicsResult ftlm_dynamics(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    const MatrixFreeHamiltonian<ExecSpace>& A,
    const MatrixFreeHamiltonian<ExecSpace>& B,
    const std::vector<Real>& time_grid,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    using namespace krylov_detail;
    const Index dim = H.dimension();
    if (dim == 0 || n_random <= 0 || n_steps <= 0 || time_grid.empty()) return {};

    const size_t R = static_cast<size_t>(n_random);
    const size_t num_times = time_grid.size();
    const int max_steps = clamp_to_dim(n_steps, dim);
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    Vec<ExecSpace> v_prev = make_vector<ExecSpace>("ftlmdyn::v_prev", dim);
    Vec<ExecSpace> v_curr = make_vector<ExecSpace>("ftlmdyn::v_curr", dim);
    Vec<ExecSpace> w = make_vector<ExecSpace>("ftlmdyn::w", dim);

    // Pass 1: Find global ground-state energy shift E_min (spectra only, no stored vectors)
    Real E_min = std::numeric_limits<Real>::infinity();
    for (size_t r = 0; r < R; ++r) {
        random_unit_vector<ExecSpace>(v_curr, seed, static_cast<uint64_t>(r));
        auto tri = lanczos_recursion<ExecSpace>(H, v_prev, v_curr, w, max_steps, [](int, const Vec<ExecSpace>&) {});
        const int M = static_cast<int>(tri.alphas.size());
        auto eig = linalg::tridiag_eigensystem_full(tri.alphas, tri.betas, M);
        if (!eig.eigenvalues.empty() && eig.eigenvalues[0] < E_min) {
            E_min = eig.eigenvalues[0];
        }
    }

    if (std::isinf(E_min)) return {};

    // Pass 2: Streamed real-time correlator evaluation
    Real Z_sum = Real(0.0);
    std::vector<Complex> C_sum(num_times, Complex(0.0, 0.0));
    std::vector<Real> sample_Z(R, Real(0.0));
    std::vector<std::vector<Complex>> sample_C(R, std::vector<Complex>(num_times, Complex(0.0, 0.0)));

    // Two Krylov bases (thermal state and B|phi>) as blocks, plus a few work vectors, allocated once.
    Blk<ExecSpace> basis = make_block<ExecSpace>("ftlmdyn::basis", dim, max_steps);
    Blk<ExecSpace> chi_basis = make_block<ExecSpace>("ftlmdyn::chi_basis", dim, max_steps);
    Vec<ExecSpace> phi_vec = make_vector<ExecSpace>("ftlmdyn::phi", dim);
    Vec<ExecSpace> chi_0 = make_vector<ExecSpace>("ftlmdyn::chi0", dim);
    Vec<ExecSpace> u_prev = make_vector<ExecSpace>("ftlmdyn::u_prev", dim);
    Vec<ExecSpace> u_curr = make_vector<ExecSpace>("ftlmdyn::u_curr", dim);
    Vec<ExecSpace> u_w = make_vector<ExecSpace>("ftlmdyn::u_w", dim);

    for (size_t r = 0; r < R; ++r) {
        random_unit_vector<ExecSpace>(v_curr, seed, static_cast<uint64_t>(r));
        auto tri = lanczos_recursion<ExecSpace>(H, v_prev, v_curr, w, max_steps,
            [&](int j, const Vec<ExecSpace>& v) {
                Vec<ExecSpace> c = column<ExecSpace>(basis, j);
                Kokkos::deep_copy(c, v);
            });

        const int M = static_cast<int>(tri.alphas.size());
        auto eig = linalg::tridiag_eigensystem_full(tri.alphas, tri.betas, M);

        // Thermal state |phi_r(beta/2)> components and sample partition weight
        Real z_r = Real(0.0);
        std::vector<Real> c_phi(M, Real(0.0));
        for (int m = 0; m < M; ++m) {
            Real exp_arg = -beta * (eig.eigenvalues[m] - E_min);
            Real exp_val = (exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(exp_arg);
            Real v0 = eig.eigenvectors[m][0];
            z_r += v0 * v0 * exp_val;

            Real half_exp_arg = -beta * (eig.eigenvalues[m] - E_min) * Real(0.5);
            Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
            Real factor = v0 * half_exp_val;
            for (int j = 0; j < M; ++j) {
                c_phi[j] += factor * eig.eigenvectors[m][j];
            }
        }
        sample_Z[r] = z_r;
        Z_sum += z_r;

        // Form thermal state |phi> in full Hilbert space (one pass over the basis)
        combine_real<ExecSpace>(basis, c_phi, phi_vec);

        // Apply B to |phi>: |chi_0> = B |phi>
        B.apply(phi_vec, chi_0);
        const Real chi_norm = norm(chi_0);

        if (chi_norm < mach_eps) {
            for (size_t ti = 0; ti < num_times; ++ti) {
                sample_C[r][ti] = Complex(0.0, 0.0);
            }
            continue;
        }

        // Run Lanczos for |chi_0>
        scale_into(u_curr, chi_0, Real(1.0) / chi_norm);
        auto tri_chi = lanczos_recursion<ExecSpace>(H, u_prev, u_curr, u_w, max_steps,
            [&](int j, const Vec<ExecSpace>& v) {
                Vec<ExecSpace> c = column<ExecSpace>(chi_basis, j);
                Kokkos::deep_copy(c, v);
            });

        const int M_chi = static_cast<int>(tri_chi.alphas.size());
        auto eig_chi = linalg::tridiag_eigensystem_full(tri_chi.alphas, tri_chi.betas, M_chi);

        // Project operator A between the two Krylov bases: A_jk = <v_j | A | w_k>
        std::vector<Complex> mat_A_cross = project_between<ExecSpace>(A, basis, M, chi_basis, M_chi, w);

        // Propagate in real time across time_grid
        for (size_t ti = 0; ti < num_times; ++ti) {
            Real t = time_grid[ti];

            // Real-time evolved |phi(t)> coefficients in basis_vectors
            std::vector<Complex> phi_t(M, Complex(0.0, 0.0));
            for (int m = 0; m < M; ++m) {
                Real theta = -eig.eigenvalues[m] * t;
                Complex phase(std::cos(theta), std::sin(theta));

                Real half_exp_arg = -beta * (eig.eigenvalues[m] - E_min) * Real(0.5);
                Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                Complex factor_phi = (eig.eigenvectors[m][0] * half_exp_val) * phase;

                for (int j = 0; j < M; ++j) {
                    phi_t[j] += factor_phi * eig.eigenvectors[m][j];
                }
            }

            // Real-time evolved |chi(t)> coefficients in chi_basis
            std::vector<Complex> chi_t(M_chi, Complex(0.0, 0.0));
            for (int m = 0; m < M_chi; ++m) {
                Real theta = -eig_chi.eigenvalues[m] * t;
                Complex phase(std::cos(theta), std::sin(theta));
                Complex factor_chi = eig_chi.eigenvectors[m][0] * phase;

                for (int k = 0; k < M_chi; ++k) {
                    chi_t[k] += factor_chi * eig_chi.eigenvectors[m][k];
                }
            }

            // Contract: <phi(t) | A | chi(t)>
            Complex corr(0.0, 0.0);
            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M_chi; ++k) {
                    corr += std::conj(phi_t[j]) * mat_A_cross[j * M_chi + k] * chi_t[k];
                }
            }
            corr *= chi_norm;

            sample_C[r][ti] = corr;
            C_sum[ti] += corr;
        }
    }

    FTLMDynamicsResult res;
    res.beta = beta;
    res.time_grid = time_grid;
    res.correlations.resize(num_times, Complex(0.0, 0.0));
    res.correlation_errors.resize(num_times, Real(0.0));

    Real Z_bar = Z_sum / Real(R);
    if (Z_sum > Real(0.0) && Z_bar > Real(0.0)) {
        for (size_t ti = 0; ti < num_times; ++ti) {
            Complex mean_c = C_sum[ti] / Z_sum;
            res.correlations[ti] = mean_c;

            if (R > 1) {
                Real var_linearized = Real(0.0);
                for (size_t r = 0; r < R; ++r) {
                    Complex delta_r = sample_C[r][ti] - mean_c * sample_Z[r];
                    var_linearized += std::norm(delta_r);
                }
                Real denom = Real(R) * Real(R - 1) * (Z_bar * Z_bar);
                if (denom > Real(0.0)) {
                    res.correlation_errors[ti] = std::sqrt(var_linearized / denom);
                }
            }
        }
    }

    return res;
}

// Explicit instantiations
#define INSTANTIATE_DYNAMICS(EXEC_SPACE) \
template DynamicsResult continued_fraction_coeffs<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, const HostVector&, int); \
template RealTimeResult time_evolve<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, const HostVector&, const std::vector<Real>&, const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, int); \
template FTLMDynamicsResult ftlm_dynamics<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, Real, const MatrixFreeHamiltonian<EXEC_SPACE>&, const MatrixFreeHamiltonian<EXEC_SPACE>&, const std::vector<Real>&, int, int, uint64_t);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_DYNAMICS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_DYNAMICS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_DYNAMICS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_DYNAMICS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_DYNAMICS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_DYNAMICS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_DYNAMICS

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
