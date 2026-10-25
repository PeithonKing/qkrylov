#pragma once
// Internal building blocks shared by the stochastic-trace and dynamics solvers (FTLM, real-time evolution).
// Not part of the public interface.

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/linalg/block_ops.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {
namespace krylov_detail {

template <class ExecSpace> using Vec = VectorView<ExecSpace>;
template <class ExecSpace> using Blk = linalg::Block<KComplex, typename ExecSpace::memory_space>;

template <class ExecSpace>
Vec<ExecSpace> make_vector(const char* label, Index dim) {
    return Vec<ExecSpace>(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), dim);
}

/// dim x ncols block of vectors (the columns are the vectors). Uninitialised.
template <class ExecSpace>
Blk<ExecSpace> make_block(const char* label, Index dim, int ncols) {
    return Blk<ExecSpace>(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), dim, ncols);
}

template <class ExecSpace>
Vec<ExecSpace> column(const Blk<ExecSpace>& B, int i) {
    return Vec<ExecSpace>(Kokkos::subview(B, Kokkos::ALL, i));
}

/// Independent random stream per sample: the seed of sample r does not depend on any other sample, so a pass
/// that revisits sample r (the streamed FTLM workflow does) regenerates exactly the same start vector.
inline uint64_t sample_seed(uint64_t seed, uint64_t r) {
    return splitmix64(seed + 0x9E3779B97F4A7C15ULL * (r + 1));
}

/// v <- random complex Gaussian vector of unit norm, generated on the device.
template <class ExecSpace>
void random_unit_vector(Vec<ExecSpace>& v, uint64_t seed, uint64_t r) {
    fill_random_normal(v, sample_seed(seed, r));
    Real nrm = norm(v);
    if (nrm < std::numeric_limits<Real>::epsilon() * Real(4.0)) nrm = Real(1.0);
    scale_into(v, v, Real(1.0) / nrm);
}

struct Tridiagonal {
    std::vector<Real> alphas;
    std::vector<Real> betas;   // betas.size() == alphas.size() - 1 (or alphas.size() after a breakdown)
};

/// Lanczos recursion from the normalised vector v_curr (no stored basis unless the sink stores it).
///
/// Per step: w = H v and alpha = <v|w> in one kernel, then w -= alpha v + beta v_prev fused with ||w||, then
/// v_next = w / beta written into the recycled buffer. `sink(j, v_curr)` sees every basis vector before H is applied.
template <class ExecSpace, class Sink>
Tridiagonal lanczos_recursion(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Vec<ExecSpace>& v_prev, Vec<ExecSpace>& v_curr, Vec<ExecSpace>& w,
    int max_steps, Sink&& sink)
{
    const Real breakdown = std::numeric_limits<Real>::epsilon() * Real(4.0);
    Tridiagonal t;
    t.alphas.reserve(max_steps);
    t.betas.reserve(max_steps);
    for (int j = 0; j < max_steps; ++j) {
        sink(j, v_curr);
        const Real alpha = H.apply_dot(v_curr, w).real();
        t.alphas.push_back(alpha);
        const Real beta_prev = j > 0 ? t.betas.back() : Real(0.0);
        const Real beta = std::sqrt(lanczos_update_norm2(w, v_curr, v_prev, alpha, beta_prev, j > 0));
        if (beta < breakdown) break;
        if (j + 1 == max_steps) break;
        t.betas.push_back(beta);
        scale_into(v_prev, w, Real(1.0) / beta);   // the oldest buffer becomes the next vector
        std::swap(v_prev, v_curr);
    }
    return t;
}

/// Regenerate the basis vectors of an earlier recursion from its coefficients (no dot products needed).
/// `sink(j, v_curr)` is called for j = 0 .. M-1; the last step skips the unneeded H·v.
template <class ExecSpace, class Sink>
void lanczos_replay(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Vec<ExecSpace>& v_prev, Vec<ExecSpace>& v_curr, Vec<ExecSpace>& w,
    const std::vector<Real>& alphas, const std::vector<Real>& betas, Sink&& sink)
{
    const int M = static_cast<int>(alphas.size());
    for (int j = 0; j < M; ++j) {
        sink(j, v_curr);
        if (j + 1 == M) break;
        H.apply(v_curr, w);
        lanczos_update_norm2(w, v_curr, v_prev, alphas[j], j > 0 ? betas[j - 1] : Real(0.0), j > 0);
        scale_into(v_prev, w, Real(1.0) / betas[j]);
        std::swap(v_prev, v_curr);
    }
}

/// Project an operator between two stored bases: mat[j * M_right + k] = <left_j | O | right_k>.
template <class ExecSpace>
std::vector<Complex> project_between(
    const MatrixFreeHamiltonian<ExecSpace>& O,
    const Blk<ExecSpace>& left, int M_left,
    const Blk<ExecSpace>& right, int M_right,
    Vec<ExecSpace>& w)
{
    using MemSpace = typename ExecSpace::memory_space;
    std::vector<Complex> mat(static_cast<size_t>(M_left) * M_right);
    for (int k = 0; k < M_right; ++k) {
        Vec<ExecSpace> rk = column<ExecSpace>(right, k);
        O.apply(rk, w);
        const auto d = linalg::block_dots<KComplex, MemSpace>(left, M_left, w);
        for (int j = 0; j < M_left; ++j) mat[static_cast<size_t>(j) * M_right + k] = d[j];
    }
    return mat;
}

/// out = sum_j c_j basis_j for real coefficients.
template <class ExecSpace>
void combine_real(const Blk<ExecSpace>& basis, const std::vector<Real>& c, Vec<ExecSpace>& out) {
    using MemSpace = typename ExecSpace::memory_space;
    const int M = static_cast<int>(c.size());
    std::vector<Complex> cc(M);
    for (int j = 0; j < M; ++j) cc[j] = Complex(c[j], Real(0.0));
    linalg::CoefBuffer<KComplex, MemSpace> buf(M);
    linalg::block_combine<KComplex, MemSpace>(basis, M, buf.set(cc, M), out);
}


// ---- real/complex-agnostic helpers (overloads, so device code never needs `if constexpr`) -------------------------

/// y = H x for a real or a complex vector.
template <class ExecSpace>
void apply_h(const MatrixFreeHamiltonian<ExecSpace>& H, const RealVectorView<ExecSpace>& x, RealVectorView<ExecSpace>& y) {
    H.apply_real(x, y);
}
template <class ExecSpace>
void apply_h(const MatrixFreeHamiltonian<ExecSpace>& H, const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) {
    H.apply(x, y);
}

/// Device vector -> host complex vector.
template <class ExecSpace>
void to_host_vector(const RealVectorView<ExecSpace>& d, HostVector& h) {
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    h.resize(d.extent(0));
    for (Index i = 0; i < d.extent(0); ++i) h[i] = Complex(m(i), Real(0.0));
}
template <class ExecSpace>
void to_host_vector(const VectorView<ExecSpace>& d, HostVector& h) {
    copy_device_to_host(d, h);
}

/// Host complex vector -> device vector (a real vector takes the real parts).
template <class ExecSpace>
void upload_vector(const HostVector& h, RealVectorView<ExecSpace>& d) {
    auto m = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, d);
    for (Index i = 0; i < d.extent(0); ++i) m(i) = h[i].real();
    Kokkos::deep_copy(d, m);
}
template <class ExecSpace>
void upload_vector(const HostVector& h, VectorView<ExecSpace>& d) {
    copy_host_to_device(h, d);
}

inline bool all_real(const HostVector& h) {
    for (const auto& c : h) {
        if (c.imag() != Real(0.0)) return false;
    }
    return true;
}

/// An uninitialised device vector of type V (real or complex view).
template <class V>
V Vec_like(const char* label, Index dim) {
    return V(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), dim);
}

}  // namespace krylov_detail
}  // namespace QKRYLOV_PRECISION_NAMESPACE
}  // namespace qkrylov
