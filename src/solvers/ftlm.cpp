#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/solvers/policy.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "krylov_detail.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace, typename Policy>
std::vector<FTLMKrylovSample> ftlm_sample(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    using namespace krylov_detail;
    const Index dim = H.dimension();
    if (dim == 0 || n_random <= 0 || n_steps <= 0) return {};

    const int max_steps = clamp_to_dim(n_steps, dim);
    const bool need_basis = !observables.empty();

    std::vector<FTLMKrylovSample> samples(n_random);

    Vec<ExecSpace> v_prev = make_vector<ExecSpace>("ftlm::v_prev", dim);
    Vec<ExecSpace> v_curr = make_vector<ExecSpace>("ftlm::v_curr", dim);
    Vec<ExecSpace> w = make_vector<ExecSpace>("ftlm::w", dim);
    // The Krylov vectors are only kept when observables must be projected onto them.
    Blk<ExecSpace> basis;
    if (need_basis) basis = make_block<ExecSpace>("ftlm::basis", dim, max_steps);

    for (int r = 0; r < n_random; ++r) {
        random_unit_vector<ExecSpace>(v_curr, seed, static_cast<uint64_t>(r));

        auto tri = lanczos_recursion<ExecSpace>(H, v_prev, v_curr, w, max_steps,
            [&](int j, const Vec<ExecSpace>& v) {
                if (need_basis) {
                    Vec<ExecSpace> c = column<ExecSpace>(basis, j);
                    Kokkos::deep_copy(c, v);
                }
            });

        const int M = static_cast<int>(tri.alphas.size());
        auto eig = linalg::tridiag_eigensystem_full(tri.alphas, tri.betas, M);

        samples[r].dimension = dim;
        samples[r].norm = Real(1.0);
        samples[r].eigenvalues = eig.eigenvalues;
        samples[r].first_components.resize(M);
        for (int m = 0; m < M; ++m) {
            samples[r].first_components[m] = eig.eigenvectors[m][0];
        }
        samples[r].eigenvectors = std::move(eig.eigenvectors);

        // Project observables onto the Krylov subspace: O_jk = <v_j | O | v_k>
        if (need_basis) {
            samples[r].projected_operators.resize(observables.size());
            for (size_t oi = 0; oi < observables.size(); ++oi) {
                ProjectedOperator proj;
                proj.matrix = project_between<ExecSpace>(observables[oi], basis, M, basis, M, w);
                samples[r].projected_operators[oi] = std::move(proj);
            }
        }
    }

    return samples;
}

FTLMSweepResult ftlm_evaluate_sweep(
    const std::vector<FTLMKrylovSample>& samples,
    const std::vector<Real>& beta_grid,
    Index dimension
)
{
    if (samples.empty() || beta_grid.empty()) return {};

    const size_t R = samples.size();
    const size_t num_betas = beta_grid.size();

    Index dim = dimension;
    if (dim == 0) {
        for (const auto& s : samples) {
            if (s.dimension > 0) {
                dim = s.dimension;
                break;
            }
        }
    }
    if (dim == 0) dim = 1;

    Real E_min = std::numeric_limits<Real>::infinity();
    size_t num_obs = 0;
    for (const auto& s : samples) {
        if (!s.eigenvalues.empty() && s.eigenvalues[0] < E_min) {
            E_min = s.eigenvalues[0];
        }
        if (s.projected_operators.size() > num_obs) {
            num_obs = s.projected_operators.size();
        }
    }

    if (std::isinf(E_min)) return {};

    FTLMSweepResult res;
    res.dimension = dim;
    res.beta_grid = beta_grid;
    res.partition_functions.resize(num_betas, Real(0.0));
    res.free_energies.resize(num_betas, Real(0.0));
    res.internal_energies.resize(num_betas, Real(0.0));
    res.specific_heats.resize(num_betas, Real(0.0));
    res.entropies.resize(num_betas, Real(0.0));
    res.effective_samples.resize(num_betas, Real(0.0));
    res.observable_expectations.resize(num_obs, std::vector<Complex>(num_betas, Complex(0.0, 0.0)));
    res.observable_errors.resize(num_obs, std::vector<Real>(num_betas, Real(0.0)));

    const Real max_exp = (sizeof(Real) > 4) ? Real(700.0) : Real(85.0);

    for (size_t bi = 0; bi < num_betas; ++bi) {
        const Real beta = beta_grid[bi];

        Real Z_sum = Real(0.0);
        Real E_sum = Real(0.0);
        Real E2_sum = Real(0.0);
        Real sum_Z_sq = Real(0.0);

        std::vector<Complex> A_sum(num_obs, Complex(0.0, 0.0));
        std::vector<Real> sample_Z(R, Real(0.0));
        std::vector<std::vector<Complex>> sample_A(num_obs, std::vector<Complex>(R, Complex(0.0, 0.0)));

        for (size_t r = 0; r < R; ++r) {
            const auto& s = samples[r];
            const int M = static_cast<int>(s.eigenvalues.size());
            if (M == 0) continue;

            Real z_r = Real(0.0);
            Real e_r = Real(0.0);
            Real e2_r = Real(0.0);
            std::vector<Complex> c(M, Complex(0.0, 0.0));

            for (int m = 0; m < M; ++m) {
                Real exp_arg = -beta * (s.eigenvalues[m] - E_min);
                Real exp_val = (exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(exp_arg);
                Real v0 = s.first_components[m];
                Real weight = v0 * v0 * exp_val;

                z_r += weight;
                e_r += s.eigenvalues[m] * weight;
                e2_r += s.eigenvalues[m] * s.eigenvalues[m] * weight;

                Real half_exp_arg = -beta * (s.eigenvalues[m] - E_min) * Real(0.5);
                Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                Complex factor(v0 * half_exp_val, Real(0.0));
                for (int j = 0; j < M; ++j) {
                    c[j] += factor * Complex(s.eigenvectors[m][j], Real(0.0));
                }
            }

            sample_Z[r] = z_r;
            Z_sum += z_r;
            E_sum += e_r;
            E2_sum += e2_r;
            sum_Z_sq += z_r * z_r;

            // Sesquilinear observable contraction: c^\dagger * O * c (yielding Complex)
            for (size_t oi = 0; oi < num_obs; ++oi) {
                if (oi >= s.projected_operators.size()) continue;
                const auto& mat = s.projected_operators[oi].matrix;
                Complex obs_val(0.0, 0.0);
                for (int j = 0; j < M; ++j) {
                    Complex conj_c_j = std::conj(c[j]);
                    for (int k = 0; k < M; ++k) {
                        obs_val += conj_c_j * mat[j * M + k] * c[k];
                    }
                }
                sample_A[oi][r] = obs_val;
                A_sum[oi] += obs_val;
            }
        }

        // Effective sample size diagnostic
        res.effective_samples[bi] = (sum_Z_sq > Real(0.0)) ? ((Z_sum * Z_sum) / sum_Z_sq) : Real(0.0);

        Real Z_bar = Z_sum / Real(R);
        if (Z_sum > Real(0.0) && Z_bar > Real(0.0)) {
            // Partition function scaled by Hilbert space dimension D
            Real log_Z = std::log(static_cast<Real>(dim)) + std::log(Z_bar) - beta * E_min;
            res.partition_functions[bi] = (log_Z < max_exp) ? std::exp(log_Z) : std::numeric_limits<Real>::infinity();

            Real mean_E = E_sum / Z_sum;
            res.internal_energies[bi] = mean_E;
            res.free_energies[bi] = (beta > Real(0.0)) ? (E_min - (std::log(static_cast<Real>(dim)) + std::log(Z_bar)) / beta) : Real(0.0);

            Real var_E = E2_sum / Z_sum - mean_E * mean_E;
            if (var_E < Real(0.0)) var_E = Real(0.0);
            res.specific_heats[bi] = (beta * beta) * var_E;

            // Observables with Linearized Ratio Variance
            for (size_t oi = 0; oi < num_obs; ++oi) {
                Complex mean_obs = A_sum[oi] / Z_sum;
                res.observable_expectations[oi][bi] = mean_obs;

                if (R > 1) {
                    Real var_linearized = Real(0.0);
                    for (size_t r = 0; r < R; ++r) {
                        Complex delta_r = sample_A[oi][r] - mean_obs * sample_Z[r];
                        var_linearized += std::norm(delta_r);
                    }
                    Real denom = Real(R) * Real(R - 1) * (Z_bar * Z_bar);
                    if (denom > Real(0.0)) {
                        res.observable_errors[oi][bi] = std::sqrt(var_linearized / denom);
                    } else {
                        res.observable_errors[oi][bi] = Real(0.0);
                    }
                } else {
                    res.observable_errors[oi][bi] = Real(0.0);
                }
            }
        }
    }

    // Thermodynamic integration for monotonic entropy when starting near zero,
    // or fallback to S = beta * (<E> - F) when beta_grid is unanchored at high-T.
    std::vector<size_t> order(num_betas);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        return beta_grid[i] < beta_grid[j];
    });

    const Real ln_D = std::log(static_cast<Real>(dim));
    const Real min_beta = beta_grid[order[0]];
    const bool starts_near_zero = (min_beta <= Real(1e-4));

    if (starts_near_zero) {
        Real integral = Real(0.0);
        Real prev_beta = Real(0.0);
        Real prev_integrand = Real(0.0);

        for (size_t idx = 0; idx < num_betas; ++idx) {
            size_t bi = order[idx];
            Real curr_beta = beta_grid[bi];
            Real curr_integrand = curr_beta * res.specific_heats[bi];

            if (curr_beta > prev_beta) {
                Real dbeta = curr_beta - prev_beta;
                integral += Real(0.5) * dbeta * (prev_integrand + curr_integrand);
            }
            prev_beta = curr_beta;
            prev_integrand = curr_integrand;

            Real s_val = ln_D - integral;
            if (s_val < Real(0.0)) s_val = Real(0.0);
            res.entropies[bi] = s_val;
        }
    } else {
        for (size_t bi = 0; bi < num_betas; ++bi) {
            Real beta = beta_grid[bi];
            if (beta > Real(0.0)) {
                Real s_val = beta * (res.internal_energies[bi] - res.free_energies[bi]);
                if (s_val < Real(0.0)) s_val = Real(0.0);
                res.entropies[bi] = s_val;
            } else {
                res.entropies[bi] = ln_D;
            }
        }
    }

    return res;
}

template <typename ExecSpace, typename Policy>
FTLMSweepResult ftlm_sweep_streamed(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    using namespace krylov_detail;
    const Index dim = H.dimension();
    if (dim == 0 || n_random <= 0 || n_steps <= 0 || beta_grid.empty()) return {};

    const size_t R = static_cast<size_t>(n_random);
    const size_t num_betas = beta_grid.size();
    const size_t num_obs = observables.size();
    const int max_steps = clamp_to_dim(n_steps, dim);
    const Real max_exp = (sizeof(Real) > 4) ? Real(700.0) : Real(85.0);

    // =========================================================================
    // PASS 1: Ground Shift Discovery & Spectrum Precomputation (H only, O(D) RAM)
    // =========================================================================
    struct SampleSpectrum {
        std::vector<Real> eigenvalues;
        std::vector<Real> first_components;
        std::vector<std::vector<Real>> eigenvectors;
        std::vector<Real> alphas;
        std::vector<Real> betas;
    };
    std::vector<SampleSpectrum> spectra(R);
    Real E_min = std::numeric_limits<Real>::infinity();

    Vec<ExecSpace> v_prev = make_vector<ExecSpace>("ftlm::v_prev", dim);
    Vec<ExecSpace> v_curr = make_vector<ExecSpace>("ftlm::v_curr", dim);
    Vec<ExecSpace> w = make_vector<ExecSpace>("ftlm::w", dim);

    for (size_t r = 0; r < R; ++r) {
        random_unit_vector<ExecSpace>(v_curr, seed, static_cast<uint64_t>(r));
        auto tri = lanczos_recursion<ExecSpace>(H, v_prev, v_curr, w, max_steps,
                                                [](int, const Vec<ExecSpace>&) {});

        const int M = static_cast<int>(tri.alphas.size());
        auto eig = linalg::tridiag_eigensystem_full(tri.alphas, tri.betas, M);

        if (!eig.eigenvalues.empty() && eig.eigenvalues[0] < E_min) {
            E_min = eig.eigenvalues[0];
        }

        spectra[r].eigenvalues = eig.eigenvalues;
        spectra[r].first_components.resize(M);
        for (int m = 0; m < M; ++m) {
            spectra[r].first_components[m] = eig.eigenvectors[m][0];
        }
        spectra[r].eigenvectors = std::move(eig.eigenvectors);
        spectra[r].alphas = std::move(tri.alphas);
        spectra[r].betas = std::move(tri.betas);
    }

    if (std::isinf(E_min)) return {};

    // =========================================================================
    // PASS 2: Boltzmann accumulation per sample. The Krylov vectors are regenerated and observables projected only
    // when there are observables; thermodynamics alone needs just the spectra above.
    // =========================================================================
    std::vector<Real> Z_sum(num_betas, Real(0.0));
    std::vector<Real> E_sum(num_betas, Real(0.0));
    std::vector<Real> E2_sum(num_betas, Real(0.0));
    std::vector<Real> sum_Z_sq(num_betas, Real(0.0));
    std::vector<std::vector<Complex>> A_sum(num_obs, std::vector<Complex>(num_betas, Complex(0.0, 0.0)));

    // Scalar histories for linearized ratio variance and R_eff: size R x (1 + K) x num_betas (~1-2 MB)
    std::vector<std::vector<Real>> sample_Z(num_betas, std::vector<Real>(R, Real(0.0)));
    std::vector<std::vector<std::vector<Complex>>> sample_A(num_obs, std::vector<std::vector<Complex>>(num_betas, std::vector<Complex>(R, Complex(0.0, 0.0))));

    Blk<ExecSpace> basis;
    int M_max = 0;
    for (const auto& sp : spectra) M_max = std::max<int>(M_max, static_cast<int>(sp.eigenvalues.size()));
    if (num_obs > 0) basis = make_block<ExecSpace>("ftlm::basis", dim, M_max);

    // Two exact ways to get <phi_beta|O|phi_beta> with |phi_beta> = sum_j c_j v_j:
    //   projection: O_jk = <v_j|O|v_k> once per sample (M applications of O and M block dot products), then
    //               c^† O c for every temperature -- cheap per temperature, expensive up front;
    //   direct:     form |phi_beta> (one pass over the basis), apply O once, one dot -- per temperature.
    // Direct wins whenever there are fewer temperatures than Krylov steps.
    const bool direct = num_obs > 0 && static_cast<int>(num_betas) < M_max;
    Vec<ExecSpace> phi = direct ? make_vector<ExecSpace>("ftlm::phi", dim) : Vec<ExecSpace>();

    for (size_t r = 0; r < R; ++r) {
        const int M = static_cast<int>(spectra[r].eigenvalues.size());

        // Project observables for sample r only
        std::vector<std::vector<Complex>> projected_mats(num_obs);
        if (num_obs > 0) {
            random_unit_vector<ExecSpace>(v_curr, seed, static_cast<uint64_t>(r));
            lanczos_replay<ExecSpace>(H, v_prev, v_curr, w, spectra[r].alphas, spectra[r].betas,
                [&](int j, const Vec<ExecSpace>& v) {
                    Vec<ExecSpace> c = column<ExecSpace>(basis, j);
                    Kokkos::deep_copy(c, v);
                });
            if (!direct) {
                for (size_t oi = 0; oi < num_obs; ++oi) {
                    projected_mats[oi] = project_between<ExecSpace>(observables[oi], basis, M, basis, M, w);
                }
            }
        }

        // Accumulate Boltzmann sums across all temperatures for sample r
        const auto& s = spectra[r];
        for (size_t bi = 0; bi < num_betas; ++bi) {
            Real beta = beta_grid[bi];

            Real z_r = Real(0.0);
            Real e_r = Real(0.0);
            Real e2_r = Real(0.0);
            std::vector<Complex> c(M, Complex(0.0, 0.0));

            for (int m = 0; m < M; ++m) {
                Real exp_arg = -beta * (s.eigenvalues[m] - E_min);
                Real exp_val = (exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(exp_arg);
                Real v0 = s.first_components[m];
                Real weight = v0 * v0 * exp_val;

                z_r += weight;
                e_r += s.eigenvalues[m] * weight;
                e2_r += s.eigenvalues[m] * s.eigenvalues[m] * weight;

                if (num_obs > 0) {
                    Real half_exp_arg = -beta * (s.eigenvalues[m] - E_min) * Real(0.5);
                    Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                    Complex factor(v0 * half_exp_val, Real(0.0));
                    for (int j = 0; j < M; ++j) {
                        c[j] += factor * Complex(s.eigenvectors[m][j], Real(0.0));
                    }
                }
            }

            sample_Z[bi][r] = z_r;
            Z_sum[bi] += z_r;
            E_sum[bi] += e_r;
            E2_sum[bi] += e2_r;
            sum_Z_sq[bi] += z_r * z_r;

            if (direct) {
                // |phi_beta> = sum_j c_j v_j, then <phi|O|phi> with a single application of each observable
                linalg::CoefBuffer<KComplex, typename ExecSpace::memory_space> cbuf(M);
                linalg::block_combine<KComplex, typename ExecSpace::memory_space>(basis, M, cbuf.set(c, M), phi);
                for (size_t oi = 0; oi < num_obs; ++oi) {
                    observables[oi].apply(phi, w);
                    const KComplex val = dot(phi, w);
                    const Complex obs_val(val.real(), val.imag());
                    sample_A[oi][bi][r] = obs_val;
                    A_sum[oi][bi] += obs_val;
                }
            } else {
                for (size_t oi = 0; oi < num_obs; ++oi) {
                    Complex obs_val(0.0, 0.0);
                    for (int j = 0; j < M; ++j) {
                        Complex conj_c_j = std::conj(c[j]);
                        for (int k = 0; k < M; ++k) {
                            obs_val += conj_c_j * projected_mats[oi][j * M + k] * c[k];
                        }
                    }
                    sample_A[oi][bi][r] = obs_val;
                    A_sum[oi][bi] += obs_val;
                }
            }
        }
    }

    // =========================================================================
    // POST-PROCESSING: Statistics, Linearized Error Bars, and Entropy
    // =========================================================================
    FTLMSweepResult res;
    res.dimension = dim;
    res.beta_grid = beta_grid;
    res.partition_functions.resize(num_betas, Real(0.0));
    res.free_energies.resize(num_betas, Real(0.0));
    res.internal_energies.resize(num_betas, Real(0.0));
    res.specific_heats.resize(num_betas, Real(0.0));
    res.entropies.resize(num_betas, Real(0.0));
    res.effective_samples.resize(num_betas, Real(0.0));
    res.observable_expectations.resize(num_obs, std::vector<Complex>(num_betas, Complex(0.0, 0.0)));
    res.observable_errors.resize(num_obs, std::vector<Real>(num_betas, Real(0.0)));

    for (size_t bi = 0; bi < num_betas; ++bi) {
        Real beta = beta_grid[bi];

        res.effective_samples[bi] = (sum_Z_sq[bi] > Real(0.0)) ? ((Z_sum[bi] * Z_sum[bi]) / sum_Z_sq[bi]) : Real(0.0);

        Real Z_bar = Z_sum[bi] / Real(R);
        if (Z_sum[bi] > Real(0.0) && Z_bar > Real(0.0)) {
            Real log_Z = std::log(static_cast<Real>(dim)) + std::log(Z_bar) - beta * E_min;
            res.partition_functions[bi] = (log_Z < max_exp) ? std::exp(log_Z) : std::numeric_limits<Real>::infinity();

            Real mean_E = E_sum[bi] / Z_sum[bi];
            res.internal_energies[bi] = mean_E;
            res.free_energies[bi] = (beta > Real(0.0)) ? (E_min - (std::log(static_cast<Real>(dim)) + std::log(Z_bar)) / beta) : Real(0.0);

            Real var_E = E2_sum[bi] / Z_sum[bi] - mean_E * mean_E;
            if (var_E < Real(0.0)) var_E = Real(0.0);
            res.specific_heats[bi] = (beta * beta) * var_E;

            for (size_t oi = 0; oi < num_obs; ++oi) {
                Complex mean_obs = A_sum[oi][bi] / Z_sum[bi];
                res.observable_expectations[oi][bi] = mean_obs;

                if (R > 1) {
                    Real var_linearized = Real(0.0);
                    for (size_t r = 0; r < R; ++r) {
                        Complex delta_r = sample_A[oi][bi][r] - mean_obs * sample_Z[bi][r];
                        var_linearized += std::norm(delta_r);
                    }
                    Real denom = Real(R) * Real(R - 1) * (Z_bar * Z_bar);
                    if (denom > Real(0.0)) {
                        res.observable_errors[oi][bi] = std::sqrt(var_linearized / denom);
                    } else {
                        res.observable_errors[oi][bi] = Real(0.0);
                    }
                } else {
                    res.observable_errors[oi][bi] = Real(0.0);
                }
            }
        }
    }

    // Thermodynamic integration for monotonic entropy when starting near zero,
    // or fallback to S = beta * (<E> - F) when beta_grid is unanchored at high-T.
    std::vector<size_t> order(num_betas);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        return beta_grid[i] < beta_grid[j];
    });

    const Real ln_D = std::log(static_cast<Real>(dim));
    const Real min_beta = beta_grid[order[0]];
    const bool starts_near_zero = (min_beta <= Real(1e-4));

    if (starts_near_zero) {
        Real integral = Real(0.0);
        Real prev_beta = Real(0.0);
        Real prev_integrand = Real(0.0);

        for (size_t idx = 0; idx < num_betas; ++idx) {
            size_t bi = order[idx];
            Real curr_beta = beta_grid[bi];
            Real curr_integrand = curr_beta * res.specific_heats[bi];

            if (curr_beta > prev_beta) {
                Real dbeta = curr_beta - prev_beta;
                integral += Real(0.5) * dbeta * (prev_integrand + curr_integrand);
            }
            prev_beta = curr_beta;
            prev_integrand = curr_integrand;

            Real s_val = ln_D - integral;
            if (s_val < Real(0.0)) s_val = Real(0.0);
            res.entropies[bi] = s_val;
        }
    } else {
        for (size_t bi = 0; bi < num_betas; ++bi) {
            Real beta = beta_grid[bi];
            if (beta > Real(0.0)) {
                Real s_val = beta * (res.internal_energies[bi] - res.free_energies[bi]);
                if (s_val < Real(0.0)) s_val = Real(0.0);
                res.entropies[bi] = s_val;
            } else {
                res.entropies[bi] = ln_D;
            }
        }
    }

    return res;
}

template <typename ExecSpace, typename Policy>
FTLMSweepResult ftlm_sweep(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed,
    FTLMWorkflow workflow
)
{
    if (workflow == FTLMWorkflow::Streamed) {
        return ftlm_sweep_streamed<ExecSpace, Policy>(H, beta_grid, observables, n_random, n_steps, seed);
    } else {
        auto samples = ftlm_sample<ExecSpace, Policy>(H, observables, n_random, n_steps, seed);
        return ftlm_evaluate_sweep(samples, beta_grid, H.dimension());
    }
}

template <typename ExecSpace, typename Policy>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    auto sweep = ftlm_sweep<ExecSpace, Policy>(H, {beta}, observables, n_random, n_steps, seed, FTLMWorkflow::Streamed);
    FTLMResult res;
    res.dimension = H.dimension();
    res.beta = beta;
    if (!sweep.partition_functions.empty()) {
        res.partition_function = sweep.partition_functions[0];
        res.free_energy = sweep.free_energies[0];
        res.internal_energy = sweep.internal_energies[0];
        res.specific_heat = sweep.specific_heats[0];
        res.entropy = sweep.entropies[0];
        res.effective_samples = sweep.effective_samples[0];
        res.observable_expectations.resize(sweep.observable_expectations.size());
        res.observable_errors.resize(sweep.observable_errors.size());
        for (size_t oi = 0; oi < sweep.observable_expectations.size(); ++oi) {
            res.observable_expectations[oi] = sweep.observable_expectations[oi][0];
            res.observable_errors[oi] = sweep.observable_errors[oi][0];
        }
    }
    return res;
}

// Explicit instantiations
#define INSTANTIATE_FTLM(EXEC_SPACE) \
template std::vector<FTLMKrylovSample> ftlm_sample<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, \
    int, int, uint64_t); \
template FTLMSweepResult ftlm_sweep_streamed<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    const std::vector<Real>&, \
    const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, \
    int, int, uint64_t); \
template FTLMSweepResult ftlm_sweep<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    const std::vector<Real>&, \
    const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, \
    int, int, uint64_t, FTLMWorkflow); \
template FTLMResult ftlm<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    Real, const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, int, int, uint64_t);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_FTLM(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_FTLM(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_FTLM(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_FTLM(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_FTLM(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_FTLM(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_FTLM

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
