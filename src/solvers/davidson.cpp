#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/linalg/block_ops.hpp"
#include "krylov_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{
struct SmallEigensystem {
    std::vector<Real> eigenvalues;
    std::vector<std::vector<Complex>> eigenvectors;
};

// Jacobi rotation-based eigensolver for small Hermitian matrices.
//
// Each rotation U = [[c, -conj(s)], [s, c]] on the pair (p, q) uses s = sin(phi) conj(a_pq) / |a_pq| and
// tan(2 phi) = 2 |a_pq| / (a_pp - a_qq), which annihilates a_pq of A' = U^† A U. The other rows keep A Hermitian:
// A'_pi = conj(A'_ip). (An earlier version used s = sin(phi) a_pq / |a_pq| and wrote A'_pi = A'_ip, which is only
// right for real symmetric matrices -- complex Hamiltonians diagonalised wrongly once the subspace had a dozen vectors.)
SmallEigensystem solve_small_hermitian(const std::vector<std::vector<Complex>>& H) {
    int n = H.size();
    std::vector<std::vector<Complex>> A = H;
    std::vector<std::vector<Complex>> V(n, std::vector<Complex>(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;

    // Stop when every off-diagonal element is at rounding level relative to the matrix scale.
    Real scale = Real(0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) scale = std::max(scale, static_cast<Real>(std::abs(A[i][j])));
    const Real jacobi_tol = std::numeric_limits<Real>::epsilon() * Real(10.0) * std::max(scale, std::numeric_limits<Real>::min());
    const int max_rotations = std::max(1000, 200 * n * n);

    for (int iter = 0; iter < max_rotations; ++iter) {
        Real max_off = 0.0;
        int p = 0, q = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A[i][j]) > max_off) {
                    max_off = std::abs(A[i][j]);
                    p = i; q = j;
                }
            }
        }

        if (max_off < jacobi_tol) break;

        const Real app = std::real(A[p][p]);
        const Real aqq = std::real(A[q][q]);
        const Complex apq = A[p][q];
        const Real r = std::abs(apq);

        const Real phi = 0.5 * std::atan2(2.0 * r, app - aqq);
        const Real c = std::cos(phi);
        const Complex s = std::sin(phi) * std::conj(apq) / r;

        for (int i = 0; i < n; ++i) {
            const Complex vip = V[i][p];
            const Complex viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -std::conj(s) * vip + c * viq;

            if (i != p && i != q) {
                const Complex aip = A[i][p];
                const Complex aiq = A[i][q];
                A[i][p] = c * aip + s * aiq;
                A[p][i] = std::conj(A[i][p]);
                A[i][q] = -std::conj(s) * aip + c * aiq;
                A[q][i] = std::conj(A[i][q]);
            }
        }
        const Real cross = Real(2.0) * c * std::real(s * apq);
        const Real s2 = std::norm(s);
        A[p][p] = Complex(c * c * app + s2 * aqq + cross, Real(0.0));
        A[q][q] = Complex(s2 * app + c * c * aqq - cross, Real(0.0));
        A[p][q] = A[q][p] = Complex(0.0, 0.0);
    }

    SmallEigensystem res;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return std::real(A[i][i]) < std::real(A[j][j]);
    });

    for (int i : idx) {
        res.eigenvalues.push_back(std::real(A[i][i]));
        std::vector<Complex> v(n);
        for (int j = 0; j < n; ++j) v[j] = V[j][i];
        res.eigenvectors.push_back(v);
    }
    return res;
}
}  // namespace

// Device pieces of the solver as named functors (never device lambdas): NVCC restricts extended lambdas, and the
// shared kernels it relies on are instantiated in several translation units.
namespace davidson_detail {

using namespace krylov_detail;

KOKKOS_INLINE_FUNCTION double real_part(Real v) { return static_cast<double>(v); }
KOKKOS_INLINE_FUNCTION double real_part(KComplex v) { return static_cast<double>(v.real()); }

KOKKOS_INLINE_FUNCTION Real guard_denominator(Real d, Real cutoff) {
    if (Kokkos::abs(d) < cutoff) d = (d >= Real(0.0)) ? cutoff : -cutoff;
    return d;
}
KOKKOS_INLINE_FUNCTION KComplex guard_denominator(KComplex d, Real cutoff) {
    if (Kokkos::abs(d) < cutoff) d = (d.real() >= Real(0.0)) ? KComplex(cutoff, Real(0.0)) : KComplex(-cutoff, Real(0.0));
    return d;
}

// t(j) = r(j) / (diag(j) - lambda), with the denominator kept away from zero.
template <class DiagT, class VecT>
struct PrecondFunctor {
    DiagT diag;
    VecT r, t;
    Real lambda;
    Real cutoff;
    KOKKOS_INLINE_FUNCTION void operator()(const Index j) const {
        t(j) = r(j) / guard_denominator(diag(j) - lambda, cutoff);
    }
};

// Index of the smallest diagonal entry that is not already chosen.
template <class DiagT>
struct MinDiagFunctor {
    using Reducer = Kokkos::MinLoc<double, Index>;
    DiagT diag;
    Kokkos::Array<Index, 64> chosen;
    int n_chosen;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, typename Reducer::value_type& upd) const {
        for (int c = 0; c < n_chosen; ++c) {
            if (chosen[c] == i) return;
        }
        const double v = real_part(diag(i));
        if (v < upd.val) {
            upd.val = v;
            upd.loc = i;
        }
    }
};

template <class DiagT, class ExecSpace>
std::vector<Index> lowest_diagonal_indices(const DiagT& diag, int k)
{
    using Functor = MinDiagFunctor<DiagT>;
    std::vector<Index> picked;
    Functor f{diag, {}, 0};
    for (int n = 0; n < k && n < 64; ++n) {
        typename Functor::Reducer::value_type best;
        Kokkos::parallel_reduce("qkrylov::davidson_min_diag", Kokkos::RangePolicy<ExecSpace, Index>(0, diag.extent(0)), f,
                                typename Functor::Reducer(best));
        picked.push_back(best.loc);
        f.chosen[n] = best.loc;
        f.n_chosen = n + 1;
    }
    return picked;
}

template <class ExecSpace, class Scalar>
using vec_t = Kokkos::View<Scalar*, typename ExecSpace::memory_space>;

// The diagonal in the vector's own scalar type. (For a real H the complex diagonal is never built.)
template <class ExecSpace>
const RealVectorView<ExecSpace>& get_diag(const MatrixFreeHamiltonian<ExecSpace>& H, Real) { return H.diagonal_real(); }
template <class ExecSpace>
const VectorView<ExecSpace>& get_diag(const MatrixFreeHamiltonian<ExecSpace>& H, KComplex) { return H.diagonal(); }

// sum_i |x_i|^2 in double
template <class VecT>
struct Norm2Functor {
    VecT x;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, double& sum) const { sum += abs2(x(i)); }
};
template <class ExecSpace, class VecT>
Real norm_of(const VecT& x) {
    double s = 0.0;
    Kokkos::parallel_reduce("qkrylov::davidson_norm2", Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
                            Norm2Functor<VecT>{x}, s);
    return static_cast<Real>(std::sqrt(s));
}

template <class ExecSpace, class Scalar>
DavidsonResult davidson_run(const MatrixFreeHamiltonian<ExecSpace>& H, int n_eig, int max_subspace, Real tol, int maxiter)
{
    using MemSpace = typename ExecSpace::memory_space;
    using Vec = vec_t<ExecSpace, Scalar>;
    using Blk = linalg::Block<Scalar, MemSpace>;
    using HS = linalg::HostScalar<Scalar>;
    constexpr bool real_path = std::is_same_v<Scalar, Real>;

    const Index dim = H.dimension();
    n_eig = clamp_to_dim(n_eig, dim);
    max_subspace = std::max(max_subspace, 2 * n_eig);
    // the subspace can never hold more than dim independent vectors
    const int cap = static_cast<int>(std::min<Index>(static_cast<Index>(max_subspace), dim));

    // diagonal in the vector's scalar type (no host copy, no sorting of the whole diagonal)
    const auto& diag = get_diag(H, Scalar{});
    using DiagT = std::decay_t<decltype(diag)>;

    const Real lin_indep_tol = std::numeric_limits<Real>::epsilon() * Real(100.0);
    const Real denom_cutoff = std::numeric_limits<Real>::epsilon() * Real(100.0);

    // Subspace vectors and their H-images, stored as columns of two blocks. The blocks start small and grow
    // (geometrically, up to `cap` columns) only when the subspace actually gets that large, so a calculation that
    // converges early never pays for the full max_subspace.
    int capacity = std::min(cap, 2 * n_eig + 6);
    auto alloc_block = [&](const char* label, int cols) {
        return Blk(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), dim, cols);
    };
    Blk V = alloc_block("qkrylov::davidson::V", capacity);
    Blk HV = alloc_block("qkrylov::davidson::HV", capacity);

    auto column = [](Blk& B, int i) { return Vec(Kokkos::subview(B, Kokkos::ALL, i)); };

    // small projected matrix <V_i|H V_j>, kept up to date one column at a time
    std::vector<std::vector<Complex>> Hs(cap, std::vector<Complex>(cap, Complex(0.0, 0.0)));
    int m = 0;

    linalg::CoefBuffer<Scalar, MemSpace> coef(cap);
    Vec work(Kokkos::view_alloc(std::string("qkrylov::davidson::work"), Kokkos::WithoutInitializing), dim);

    // Make room for `need` columns, keeping the first m.
    auto ensure_capacity = [&](int need) {
        if (need <= capacity) return;
        const int new_cap = std::min(cap, std::max(need, 2 * capacity));
        Blk V2 = alloc_block("qkrylov::davidson::V", new_cap);
        Blk HV2 = alloc_block("qkrylov::davidson::HV", new_cap);
        if (m > 0) {
            Kokkos::deep_copy(Kokkos::subview(V2, Kokkos::ALL, std::make_pair(0, m)), Kokkos::subview(V, Kokkos::ALL, std::make_pair(0, m)));
            Kokkos::deep_copy(Kokkos::subview(HV2, Kokkos::ALL, std::make_pair(0, m)), Kokkos::subview(HV, Kokkos::ALL, std::make_pair(0, m)));
        }
        V = V2;
        HV = HV2;
        capacity = new_cap;
    };

    // Orthonormalise the candidate in `work` against the current subspace, append it and update H_sub.
    auto append_candidate = [&]() -> bool {
        if (m >= cap) return false;
        ensure_capacity(m + 1);
        linalg::block_orthogonalize<Scalar, MemSpace>(V, m, work, coef);
        const Real nrm = norm_of<ExecSpace>(work);
        if (!(nrm > lin_indep_tol)) return false;
        Vec vcol = column(V, m);
        scale_into(vcol, work, Real(1.0) / nrm);
        Vec hcol = column(HV, m);
        apply_h(H, vcol, hcol);
        const auto d = linalg::block_dots<Scalar, MemSpace>(V, m + 1, hcol);   // <V_i | H V_m>, i <= m
        for (int i = 0; i < m; ++i) {
            Hs[i][m] = Complex(d[i]);
            Hs[m][i] = std::conj(Complex(d[i]));
        }
        Hs[m][m] = Complex(std::real(Complex(d[m])), Real(0.0));
        ++m;
        return true;
    };

    // Initial subspace: unit vectors on the n_eig smallest diagonal entries.
    std::vector<Index> start;
    if (n_eig <= 64) {
        start = lowest_diagonal_indices<DiagT, ExecSpace>(diag, n_eig);
    } else {
        // Very large n_eig (rare): fall back to a full host sort.
        auto dh = H.diagonal_host();
        std::vector<Index> idx(dim);
        for (Index i = 0; i < dim; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](Index a, Index b) { return dh[a].real() < dh[b].real(); });
        start.assign(idx.begin(), idx.begin() + n_eig);
    }
    for (int i = 0; i < n_eig; ++i) {
        Kokkos::deep_copy(work, Scalar{});
        auto e = Kokkos::subview(work, start[i]);
        Kokkos::deep_copy(e, Scalar(1.0));
        if (!append_candidate()) {
            // unit vector not independent of the others: a random start instead
            fill_random(work, 1234ULL + static_cast<uint64_t>(i));
            append_candidate();
        }
    }

    // Persistent buffers: residual, correction vectors, restart scratch.
    Vec r(Kokkos::view_alloc(std::string("qkrylov::davidson::r"), Kokkos::WithoutInitializing), dim);
    std::vector<Vec> corrections;
    for (int k = 0; k < n_eig; ++k)
        corrections.emplace_back(Kokkos::view_alloc(std::string("qkrylov::davidson::t"), Kokkos::WithoutInitializing), dim);
    std::vector<Vec> ritz_v, ritz_hv;

    auto eigvec_coeffs = [&](const SmallEigensystem& eig, int k) {
        std::vector<HS> c(m);
        for (int i = 0; i < m; ++i) {
            if constexpr (real_path) c[i] = static_cast<Real>(eig.eigenvectors[k][i].real());
            else c[i] = eig.eigenvectors[k][i];
        }
        return c;
    };

    for (int iter = 0; iter < maxiter; ++iter) {
        if (m == 0) return {};
        std::vector<std::vector<Complex>> Hsub(m, std::vector<Complex>(m));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) Hsub[i][j] = Hs[i][j];
        const auto eig = solve_small_hermitian(Hsub);
        const int n_out = std::min<int>(n_eig, m);

        bool all_converged = true;
        std::vector<int> need;   // which Ritz pairs still need a correction
        for (int k = 0; k < n_out; ++k) {
            const Real lambda = eig.eigenvalues[k];
            const auto& s_dev = coef.set(eigvec_coeffs(eig, k), m);
            const Real res_norm = linalg::block_residual<Scalar, MemSpace>(V, HV, m, s_dev, lambda, r);
            if (res_norm > tol) {
                all_converged = false;
                Vec& t = corrections[need.size()];
                PrecondFunctor<DiagT, Vec> pf{diag, r, t, lambda, denom_cutoff};
                Kokkos::parallel_for("qkrylov::davidson_precond", Kokkos::RangePolicy<ExecSpace, Index>(0, dim), pf);
                need.push_back(k);
            }
        }

        if (all_converged || iter == maxiter - 1) {
            DavidsonResult res;
            res.iterations = iter + 1;
            res.converged = all_converged;
            for (int k = 0; k < n_out; ++k) {
                res.eigenvalues.push_back(eig.eigenvalues[k]);
                Vec ev(Kokkos::view_alloc(std::string("qkrylov::davidson::ev"), Kokkos::WithoutInitializing), dim);
                linalg::block_combine<Scalar, MemSpace>(V, m, coef.set(eigvec_coeffs(eig, k), m), ev);
                HostVector host_ev;
                to_host_vector<ExecSpace>(ev, host_ev);
                res.eigenvectors.push_back(std::move(host_ev));
            }
            return res;
        }

        // Restart: keep the n_eig Ritz vectors (and their H-images) when the subspace would overflow.
        if (static_cast<size_t>(m) + need.size() > static_cast<size_t>(max_subspace) || m + static_cast<int>(need.size()) > cap) {
            // Thick restart: keep the wanted Ritz vectors plus an equal number of the next ones. Throwing the
            // rest away (keeping only n_eig) discards the directions the next corrections build on and slows
            // convergence markedly (82 -> 51 iterations for three states of a 16-site chain).
            const int keep = std::min({m, 2 * n_eig, std::max(n_out, max_subspace - n_eig)});
            while (static_cast<int>(ritz_v.size()) < keep) {   // scratch buffers, allocated once and reused
                ritz_v.emplace_back(Kokkos::view_alloc(std::string("qkrylov::davidson::ritz_v"), Kokkos::WithoutInitializing), dim);
                ritz_hv.emplace_back(Kokkos::view_alloc(std::string("qkrylov::davidson::ritz_hv"), Kokkos::WithoutInitializing), dim);
            }
            for (int k = 0; k < keep; ++k) {
                linalg::block_combine2<Scalar, MemSpace>(V, HV, m, coef.set(eigvec_coeffs(eig, k), m), ritz_v[k], ritz_hv[k]);
            }
            for (int k = 0; k < keep; ++k) {
                Vec vc = column(V, k);
                Vec hc = column(HV, k);
                Kokkos::deep_copy(vc, ritz_v[k]);
                Kokkos::deep_copy(hc, ritz_hv[k]);
            }
            m = keep;
            for (int i = 0; i < cap; ++i)
                for (int j = 0; j < cap; ++j) Hs[i][j] = Complex(0.0, 0.0);
            for (int k = 0; k < keep; ++k) Hs[k][k] = Complex(eig.eigenvalues[k], Real(0.0));   // exact: Ritz pairs
        }

        for (size_t q = 0; q < need.size(); ++q) {
            Kokkos::deep_copy(work, corrections[q]);
            append_candidate();
        }
    }

    return {};
}

}  // namespace davidson_detail

template <typename ExecSpace>
DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig,
    int max_subspace,
    Real tol,
    int maxiter
)
{
    if (maxiter <= 0) throw std::invalid_argument("davidson_lowest: maxiter must be positive");
    if (H.dimension() == 0) return {};
    // Real Hamiltonian: the whole iteration runs in real arithmetic (half the memory and traffic).
    if (H.has_real_coefficients()) return davidson_detail::davidson_run<ExecSpace, Real>(H, n_eig, max_subspace, tol, maxiter);
    return davidson_detail::davidson_run<ExecSpace, KComplex>(H, n_eig, max_subspace, tol, maxiter);
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template DavidsonResult davidson_lowest<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, int, int, Real, int);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template DavidsonResult davidson_lowest<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, int, int, Real, int);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template DavidsonResult davidson_lowest<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, int, int, Real, int);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template DavidsonResult davidson_lowest<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, int, int, Real, int);
#endif
#ifdef KOKKOS_ENABLE_HIP
template DavidsonResult davidson_lowest<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, int, int, Real, int);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template DavidsonResult davidson_lowest<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, int, int, Real, int);
#endif
}

}
