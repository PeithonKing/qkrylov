#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"

#include <random>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <type_traits>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace solvers {

namespace kern {
using namespace QKRYLOV_PRECISION_NAMESPACE;

template <class V> using S = typename V::non_const_value_type;
template <class V> constexpr bool is_real_v = std::is_same_v<S<V>, Real>;

template <class V> double dot_acc_real(const V& x, const V& y) {
    using ExecSpace = typename V::execution_space;
    double r = 0.0;
    Kokkos::parallel_reduce("qkrylov::dot_real", Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i, double& sum) { sum += static_cast<double>(x(i)) * y(i); }, r);
    return r;
}
template <class V> S<V> dot(const V& x, const V& y) {
    if constexpr (is_real_v<V>) return static_cast<Real>(dot_acc_real(x, y));
    else return QKRYLOV_PRECISION_NAMESPACE::dot(x, y);
}
template <class V> Real norm(const V& x) {
    if constexpr (is_real_v<V>) return std::sqrt(kern::dot(x, x));
    else return std::sqrt(kern::dot(x, x).real());
}
template <class V> void axpy(S<V> a, const V& x, V& y) {
    using ExecSpace = typename V::execution_space;
    Kokkos::parallel_for("qkrylov::axpy_g", Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) { y(i) += a * x(i); });
}
template <class V> void normalize(V& x) {
    const Real n = kern::norm(x);
    if (n == Real(0.0)) return;
    scale_into(x, x, Real(1.0) / n);
}
template <class HT, class V> Real apply_dot(const HT& H, const V& x, V& y) {
    if constexpr (is_real_v<V>) return H.apply_dot_real(x, y);
    else return H.apply_dot(x, y).real();
}
template <class HT, class V> void apply(const HT& H, const V& x, V& y) {
    if constexpr (is_real_v<V>) H.apply_real(x, y);
    else H.apply(x, y);
}
template <class V> void to_host(const V& d, HostVector& h) {
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    h.resize(d.extent(0));
    for (Index i = 0; i < d.extent(0); ++i) {
        if constexpr (is_real_v<V>) h[i] = Complex(m(i), Real(0.0));
        else h[i] = Complex(m(i).real(), m(i).imag());
    }
}
/// Ground-state export. Real vector + real_output: one device->host copy straight into
/// result.eigenvector_real; otherwise the complex host vector (and its eigenvectors alias).
template <class V> void export_ground_state(const V& d, const LanczosConfig& cfg, LanczosResult& res) {
    if constexpr (is_real_v<V>) {
        if (cfg.eigenvector_real_out) {
            Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(cfg.eigenvector_real_out, d.extent(0));
            Kokkos::deep_copy(h, d);
            return;
        }
        if (cfg.real_output) {
            res.eigenvector_real.resize(d.extent(0));
            Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(res.eigenvector_real.data(), d.extent(0));
            Kokkos::deep_copy(h, d);
            return;
        }
    }
    if constexpr (!is_real_v<V>) {
        if (cfg.eigenvector_complex_out) {
            Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(
                reinterpret_cast<KComplex*>(cfg.eigenvector_complex_out), d.extent(0));
            Kokkos::deep_copy(h, d);
            return;
        }
    }
    to_host(d, res.eigenvector);
    res.eigenvectors = {res.eigenvector};
}
} // namespace kern

template <typename Policy, typename ExecSpace, typename Scalar>
QKRYLOV_PRECISION_NAMESPACE::LanczosResult lanczos_run(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    using Traits = policy::policy_traits<Policy>;
    using Vec = Kokkos::View<Scalar*, typename ExecSpace::memory_space>;
    static_assert(policy::is_policy_v<Policy>, "Unknown or invalid solver policy");

    const Index dim = H.dimension();
    if (dim == 0) return {};

    // 1. Policy Scope Enforcement
    if constexpr (!Traits::supports_multistate) {
        if (config.n_eig > 1) {
            throw std::invalid_argument(
                "Selected Lanczos policy only supports ground-state calculations (n_eig = 1). "
                "Use policy::OnePass_DKGS for an arbitrary number of low energy states."
            );
        }
    }

    const int target_n_eig = Traits::supports_multistate
        ? std::max(1, clamp_to_dim(config.n_eig, dim))
        : 1;

    // 2. Working vectors. Allocated without zero-fill: every entry is written before use.
    auto alloc = [dim](const char* label) {
        return Vec(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), dim);
    };
    Vec v_prev = alloc("v_prev");
    Vec v_curr = alloc("v_curr");
    Vec w = alloc("w");

    // Stored-basis policies keep each Krylov vector by handle (no copies); each new
    // vector gets a fresh allocation instead of recycling v_prev.
    std::vector<Vec> basis_vectors;
    if constexpr (Traits::stores_basis) {
        basis_vectors.reserve(config.maxiter);
    }

    // 3. State Initialization
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    const uint64_t seed = config.seed.value_or(123456789ULL);

    auto load_start_vector = [&](Vec& v) {
        if (!config.initial_vector.empty()) {
            auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, v);
            for (Index i = 0; i < dim; ++i) {
                if constexpr (kern::is_real_v<Vec>) host(i) = config.initial_vector[i].real();
                else host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
            }
            Kokkos::deep_copy(v, host);
        } else {
            fill_random(v, seed);
        }
        kern::normalize(v);
    };

    if (!config.initial_vector.empty() && static_cast<Index>(config.initial_vector.size()) != dim) {
        throw std::invalid_argument("initial_vector size (" + std::to_string(config.initial_vector.size()) +
                                   ") does not match Hamiltonian dimension (" + std::to_string(dim) + ")");
    }
    load_start_vector(v_curr);
    if (!config.initial_vector.empty() && kern::norm(v_curr) < mach_eps) {
        throw std::invalid_argument("initial_vector has zero norm");
    }

    std::vector<Real> alphas;
    std::vector<Real> betas;
    alphas.reserve(config.maxiter);
    betas.reserve(config.maxiter);

    std::vector<Real> prev_energies;
    bool converged = false;
    int actual_iters = 0;

    // Convergence tolerance floored at what the working precision can resolve: asking
    // float32 for |dE| < 1e-8 at |E| ~ 10 (one ulp ~ 1e-6) only converges by chance.
    auto effective_tol = [&](Real energy) {
        return std::max(static_cast<Real>(config.tol),
                        Real(8.0) * std::numeric_limits<Real>::epsilon() * std::abs(energy));
    };

    const Real effective_breakdown = (config.breakdown_tol > Real(0.0))
        ? config.breakdown_tol
        : mach_eps;

    // 4. Phase 1: Search Loop
    //    Per iteration: w = H v and alpha = <v|w> (one kernel), then the three-term update
    //    fused with ||w||^2 (one pass), then v_next = w / beta (one pass).
    const int max_steps = clamp_to_dim(config.maxiter, dim);
    for (int iter = 0; iter < max_steps; ++iter) {
        actual_iters = iter + 1;

        if constexpr (Traits::stores_basis) {
            basis_vectors.push_back(v_curr);
        }

        const Real alpha = kern::apply_dot(H, v_curr, w);
        alphas.push_back(alpha);
        const Real beta_prev = iter > 0 ? betas.back() : Real(0.0);

        Real beta;
        if constexpr (Traits::uses_dgks) {
            lanczos_update_norm2(w, v_curr, v_prev, alpha, beta_prev, iter > 0);
            // DGKS Full Reorthogonalization (Twice-is-enough)
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& bv : basis_vectors) {
                    const Scalar proj = kern::dot(bv, w);
                    kern::axpy(-proj, bv, w);
                }
            }
            beta = kern::norm(w);
        } else {
            beta = std::sqrt(lanczos_update_norm2(w, v_curr, v_prev, alpha, beta_prev, iter > 0));
        }

        if (beta < effective_breakdown) {
            converged = true;
            break;
        }
        if (static_cast<Index>(iter + 1) == dim) {
            converged = true;
            break;
        }
        if (iter + 1 == max_steps) {
            break;
        }
        betas.push_back(beta);

        if constexpr (Traits::stores_basis) {
            Vec v_next = alloc("basis_v");
            scale_into(v_next, w, Real(1.0) / beta);
            v_prev = v_curr;
            v_curr = v_next;
        } else {
            scale_into(v_prev, w, Real(1.0) / beta); // recycle the oldest buffer
            std::swap(v_prev, v_curr);
        }

        // Convergence Check
        const int m = static_cast<int>(alphas.size());
        if (config.early_stopping && m >= target_n_eig && (iter + 1) >= config.min_iterations && (iter + 1) % config.check_interval == 0) {
            if constexpr (Traits::supports_multistate) {
                auto tridiag = linalg::tridiag_eigensystem_full(alphas, betas, m);
                if (!prev_energies.empty()) {
                    bool all_converged = true;
                    for (int k = 0; k < target_n_eig; ++k) {
                        Real diff = std::abs(tridiag.eigenvalues[k] - prev_energies[k]);
                        Real ritz_res = beta * std::abs(tridiag.eigenvectors[k][m - 1]);
                        const Real tol_k = effective_tol(tridiag.eigenvalues[k]);
                        if (diff > tol_k && ritz_res > tol_k) {
                            all_converged = false;
                            break;
                        }
                    }
                    if (all_converged) {
                        converged = true;
                        break;
                    }
                }
                prev_energies.assign(tridiag.eigenvalues.begin(), tridiag.eigenvalues.begin() + target_n_eig);
            } else {
                auto tridiag = linalg::tridiag_ground_state_full(alphas, betas, m);
                if (!prev_energies.empty()) {
                    if (std::abs(tridiag.energy - prev_energies[0]) < effective_tol(tridiag.energy)) {
                        converged = true;
                        break;
                    }
                }
                prev_energies = {tridiag.energy};
            }
        }
    }

    // 5. Final Tridiagonal Eigensystem Solution
    const int final_m = static_cast<int>(alphas.size());
    auto final_tridiag = linalg::tridiag_eigensystem_full(alphas, betas, final_m);

    LanczosResult result;
    result.iterations = actual_iters;
    result.converged = converged;

    const int num_out = std::min<int>(target_n_eig, static_cast<int>(final_tridiag.eigenvalues.size()));
    result.eigenvalues.assign(final_tridiag.eigenvalues.begin(), final_tridiag.eigenvalues.begin() + num_out);
    result.energy = result.eigenvalues.empty() ? Real(0.0) : result.eigenvalues[0];
    result.alphas = alphas;
    result.betas = betas;

    // 6. Phase 2: Vector Reconstruction
    if constexpr (Traits::computes_vector) {
        if (!config.compute_eigenvectors) {
            return result;
        }

        if constexpr (Traits::is_two_pass) {
            // TwoPass: replay the recursion from the start vector, accumulating the k = 0 Ritz vector.
            Vec ritz("ritz", dim);
            load_start_vector(v_curr);

            for (int j = 0; j < final_m; ++j) {
                Real y_j = final_tridiag.eigenvectors[0][j];
                kern::axpy(Scalar(y_j), v_curr, ritz);

                if (j + 1 == final_m) break; // SpMV elision optimization

                kern::apply(H, v_curr, w);
                lanczos_update_norm2(w, v_curr, v_prev, alphas[j], j > 0 ? betas[j - 1] : Real(0.0), j > 0);
                scale_into(v_prev, w, Real(1.0) / betas[j]);
                std::swap(v_prev, v_curr);
            }

            kern::normalize(ritz);
            kern::export_ground_state(ritz, config, result);
        } else if constexpr (Traits::stores_basis) {
            if constexpr (Traits::supports_multistate) {
                // OnePass_DKGS: Arbitrary number of low energy states
                for (int k = 0; k < num_out; ++k) {
                    Vec ritz("ritz", dim);
                    for (int j = 0; j < final_m; ++j) {
                        Real y_j = final_tridiag.eigenvectors[k][j];
                        kern::axpy(Scalar(y_j), basis_vectors[j], ritz);
                    }
                    kern::normalize(ritz);
                    HostVector host_ritz;
                    kern::to_host(ritz, host_ritz);
                    result.eigenvectors.push_back(std::move(host_ritz));
                }
                if (!result.eigenvectors.empty()) {
                    result.eigenvector = result.eigenvectors[0];
                }
            } else {
                // OnePass_full: Ground state only (k = 0)
                Vec ritz("ritz", dim);
                for (int j = 0; j < final_m; ++j) {
                    Real y_j = final_tridiag.eigenvectors[0][j];
                    kern::axpy(Scalar(y_j), basis_vectors[j], ritz);
                }
                kern::normalize(ritz);
                kern::export_ground_state(ritz, config, result);
            }
        }
    }

    return result;
}


/// Real Hamiltonians (all coefficients real) with a real start vector run the whole
/// recursion in real arithmetic: half the memory traffic, no complex multiplies.
template <typename Policy, typename ExecSpace>
QKRYLOV_PRECISION_NAMESPACE::LanczosResult lanczos(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    bool real_path = H.has_real_coefficients();
    for (const auto& c : config.initial_vector) {
        if (c.imag() != Real(0.0)) { real_path = false; break; }
    }
    if (real_path) return lanczos_run<Policy, ExecSpace, Real>(H, config);
    return lanczos_run<Policy, ExecSpace, KComplex>(H, config);
}

} // namespace solvers

// Explicit instantiations for all 4 policies across enabled Kokkos execution spaces
#define INSTANTIATE_LANCZOS(Space) \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass_DKGS, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass_full, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::TwoPass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_LANCZOS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_LANCZOS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_LANCZOS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_LANCZOS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_LANCZOS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_LANCZOS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_LANCZOS

} // namespace qkrylov
