#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/core/kokkos_types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

/// \file block_ops.hpp
/// \brief Operations on a block of vectors stored as the columns of one 2-D view.
///
/// Subspace solvers (Davidson, full reorthogonalisation, Krylov projections) combine m stored vectors with a
/// small coefficient vector. Done one vector at a time that costs m passes over memory per combination; with the
/// vectors in one column-major block it is a single pass. The scalar is `Real` (real Hamiltonians) or `KComplex`.

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {
namespace linalg {

/// Column-major block: B(j, i) is element j of vector i, and every column is contiguous in memory.
template <class Scalar, class MemSpace>
using Block = Kokkos::View<Scalar**, Kokkos::LayoutLeft, MemSpace>;

template <class Scalar> inline constexpr bool is_real_scalar_v = std::is_same_v<Scalar, Real>;

/// The matching host-side scalar for a device scalar.
template <class Scalar> using HostScalar = std::conditional_t<is_real_scalar_v<Scalar>, Real, Complex>;

namespace detail {

// Overloads instead of `if constexpr` inside the device functors: NVCC rejects a lambda/functor body whose
// captured variable is first used inside `if constexpr`.
KOKKOS_INLINE_FUNCTION void acc_dot(double* a, int i, Real x, Real v) { a[i] += static_cast<double>(x) * v; }
KOKKOS_INLINE_FUNCTION void acc_dot(double* a, int i, KComplex x, KComplex v) {
    const KComplex p = Kokkos::conj(x) * v;
    a[2 * i] += p.real();
    a[2 * i + 1] += p.imag();
}
template <class S> inline constexpr int slots_v = std::is_same_v<S, Real> ? 1 : 2;

inline Real host_to_scalar(Real v, Real) { return v; }
inline KComplex host_to_scalar(Complex v, KComplex) { return KComplex(v.real(), v.imag()); }
inline Real host_to_scalar(Complex v, Real) { return v.real(); }

/// Rows are processed in tiles. On the host a tile is a few hundred rows: a thread streams each column of the tile
/// contiguously while the tile of the other operand/accumulator stays in L1 (the column-at-a-time access that
/// separate dot/axpy calls get, but with one pass over the shared vector). On a GPU a "tile" is one row, so
/// neighbouring threads touch neighbouring rows (coalesced).
template <class MemSpace>
inline constexpr int tile_rows_v = Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible ? 256 : 1;

/// result[i] = sum_j conj(B(j,i)) * x(j), i < m (accumulated in double).
template <int TILE, class BlockT, class VecT>
struct BlockDotsFunctor {
    using value_type = double[];
    BlockT B;
    VecT x;
    Index dim;
    int col0;         // first column of this chunk
    int m;            // columns in this chunk
    int value_count;  // Kokkos array-reduction length

    KOKKOS_INLINE_FUNCTION void operator()(const Index t, value_type a) const {
        const Index j0 = t * TILE;
        const Index j1 = (j0 + TILE < dim) ? j0 + TILE : dim;
        for (int i = 0; i < m; ++i) {
            double loc[2] = {0.0, 0.0};
            for (Index j = j0; j < j1; ++j) acc_dot(loc, 0, B(j, col0 + i), x(j));
            if (slots_v<typename VecT::non_const_value_type> == 1) {
                a[i] += loc[0];
            } else {
                a[2 * i] += loc[0];
                a[2 * i + 1] += loc[1];
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void init(value_type a) const {
        for (int k = 0; k < value_count; ++k) a[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
        for (int k = 0; k < value_count; ++k) dst[k] += src[k];
    }
};

/// out(j) = [accumulate ? out(j) : 0] + sign * sum_i c(i) * B(j,i)
template <int TILE, class BlockT, class VecT, class CoefT>
struct BlockCombineFunctor {
    BlockT B;
    VecT out;
    CoefT c;
    Index dim;
    int m;
    Real sign;
    bool accumulate;
    KOKKOS_INLINE_FUNCTION void operator()(const Index t) const {
        using S = typename VecT::non_const_value_type;
        const Index j0 = t * TILE;
        const int n = static_cast<int>((j0 + TILE < dim) ? TILE : dim - j0);
        S buf[TILE];
        for (int q = 0; q < n; ++q) buf[q] = S{};
        for (int i = 0; i < m; ++i) {
            const S ci = c(i);
            for (int q = 0; q < n; ++q) buf[q] += ci * B(j0 + q, i);
        }
        for (int q = 0; q < n; ++q) out(j0 + q) = accumulate ? out(j0 + q) + sign * buf[q] : sign * buf[q];
    }
};

/// r(j) = sum_i s(i) * (HV(j,i) - lambda * V(j,i)); also returns ||r||^2.
template <int TILE, class BlockT, class VecT, class CoefT>
struct ResidualFunctor {
    BlockT V, HV;
    VecT r;
    CoefT s;
    Real lambda;
    Index dim;
    int m;
    KOKKOS_INLINE_FUNCTION void operator()(const Index t, double& sum) const {
        using S = typename VecT::non_const_value_type;
        const Index j0 = t * TILE;
        const int n = static_cast<int>((j0 + TILE < dim) ? TILE : dim - j0);
        S buf[TILE];
        for (int q = 0; q < n; ++q) buf[q] = S{};
        for (int i = 0; i < m; ++i) {
            const S si = s(i);
            for (int q = 0; q < n; ++q) buf[q] += si * (HV(j0 + q, i) - lambda * V(j0 + q, i));
        }
        double local = 0.0;
        for (int q = 0; q < n; ++q) {
            r(j0 + q) = buf[q];
            local += abs2(buf[q]);
        }
        sum += local;
    }
};

/// out1(j) = sum_i c(i) B1(j,i) and out2(j) = sum_i c(i) B2(j,i) (a Ritz vector and its H-image).
template <int TILE, class BlockT, class VecT, class CoefT>
struct Combine2Functor {
    BlockT B1, B2;
    VecT out1, out2;
    CoefT c;
    Index dim;
    int m;
    KOKKOS_INLINE_FUNCTION void operator()(const Index t) const {
        using S = typename VecT::non_const_value_type;
        const Index j0 = t * TILE;
        const int n = static_cast<int>((j0 + TILE < dim) ? TILE : dim - j0);
        S b1[TILE], b2[TILE];
        for (int q = 0; q < n; ++q) { b1[q] = S{}; b2[q] = S{}; }
        for (int i = 0; i < m; ++i) {
            const S ci = c(i);
            for (int q = 0; q < n; ++q) {
                b1[q] += ci * B1(j0 + q, i);
                b2[q] += ci * B2(j0 + q, i);
            }
        }
        for (int q = 0; q < n; ++q) { out1(j0 + q) = b1[q]; out2(j0 + q) = b2[q]; }
    }
};

}  // namespace detail

/// out[i] = <B_i | x> for the first m columns of B (one pass over B and x).
template <class Scalar, class MemSpace>
std::vector<HostScalar<Scalar>> block_dots(
    const Block<Scalar, MemSpace>& B, int m, const Kokkos::View<Scalar*, MemSpace>& x)
{
    using ExecSpace = typename MemSpace::execution_space;
    constexpr int slots = detail::slots_v<Scalar>;
    constexpr int TILE = detail::tile_rows_v<MemSpace>;
    using Vec = Kokkos::View<Scalar*, MemSpace>;
    std::vector<HostScalar<Scalar>> out(m);
    if (m <= 0) return out;
    const Index dim = x.extent(0);
    const Index tiles = (dim + TILE - 1) / TILE;
    // The reduction length is bounded (columns are processed in chunks) so it always fits a GPU block's shared memory.
    constexpr int chunk = 32 / slots;
    for (int c0 = 0; c0 < m; c0 += chunk) {
        const int mc = std::min(chunk, m - c0);
        detail::BlockDotsFunctor<TILE, Block<Scalar, MemSpace>, Vec> f{B, x, dim, c0, mc, slots * mc};
        Kokkos::View<double*, Kokkos::HostSpace> res("qkrylov::block_dots_result", slots * mc);
        Kokkos::parallel_reduce("qkrylov::block_dots", Kokkos::RangePolicy<ExecSpace, Index>(0, tiles), f, res);
        Kokkos::fence();
        for (int i = 0; i < mc; ++i) {
            if constexpr (slots == 1) out[c0 + i] = static_cast<Real>(res(i));
            else out[c0 + i] = Complex(static_cast<Real>(res(2 * i)), static_cast<Real>(res(2 * i + 1)));
        }
    }
    return out;
}

/// Reusable small coefficient array: a host mirror plus a device view of fixed capacity, so the hot loops of the
/// subspace solvers do not allocate (a device allocation is a synchronising call on GPUs).
template <class Scalar, class MemSpace>
class CoefBuffer {
public:
    using DeviceView = Kokkos::View<Scalar*, MemSpace>;
    explicit CoefBuffer(int capacity)
        : dev_("qkrylov::coef_buffer", capacity > 0 ? capacity : 1), host_(Kokkos::create_mirror_view(dev_)) {}

    /// Copy the first m coefficients to the device and return the device view.
    const DeviceView& set(const std::vector<HostScalar<Scalar>>& c, int m) {
        for (int i = 0; i < m; ++i) host_(i) = detail::host_to_scalar(c[i], Scalar{});
        Kokkos::deep_copy(dev_, host_);
        return dev_;
    }

private:
    DeviceView dev_;
    typename DeviceView::host_mirror_type host_;
};

/// out = sum_i c_i B_i  (accumulate = false)  or  out += sign * sum_i c_i B_i  (accumulate = true).
/// `c` is a device coefficient view (see CoefBuffer).
template <class Scalar, class MemSpace>
void block_combine(
    const Block<Scalar, MemSpace>& B, int m, const Kokkos::View<Scalar*, MemSpace>& c,
    Kokkos::View<Scalar*, MemSpace>& out, Real sign = Real(1.0), bool accumulate = false)
{
    using ExecSpace = typename MemSpace::execution_space;
    constexpr int TILE = detail::tile_rows_v<MemSpace>;
    using Vec = Kokkos::View<Scalar*, MemSpace>;
    if (m <= 0) return;
    const Index dim = out.extent(0);
    detail::BlockCombineFunctor<TILE, Block<Scalar, MemSpace>, Vec, Vec> f{B, out, c, dim, m, sign, accumulate};
    Kokkos::parallel_for("qkrylov::block_combine", Kokkos::RangePolicy<ExecSpace, Index>(0, (dim + TILE - 1) / TILE), f);
}

/// t -= B (B^† t), `passes` times ("twice is enough" classical Gram-Schmidt) against the first m columns:
/// 2 passes over the block per round instead of 2m separate vector operations.
template <class Scalar, class MemSpace>
void block_orthogonalize(
    const Block<Scalar, MemSpace>& B, int m, Kokkos::View<Scalar*, MemSpace>& t,
    CoefBuffer<Scalar, MemSpace>& coef, int passes = 2)
{
    for (int p = 0; p < passes; ++p) {
        const auto proj = block_dots<Scalar, MemSpace>(B, m, t);
        block_combine<Scalar, MemSpace>(B, m, coef.set(proj, m), t, Real(-1.0), true);
    }
}

/// r = sum_i s_i (HV_i - lambda V_i) over the first m columns; returns ||r||. One pass over both blocks.
template <class Scalar, class MemSpace>
Real block_residual(
    const Block<Scalar, MemSpace>& V, const Block<Scalar, MemSpace>& HV, int m,
    const Kokkos::View<Scalar*, MemSpace>& s, Real lambda, Kokkos::View<Scalar*, MemSpace>& r)
{
    using ExecSpace = typename MemSpace::execution_space;
    constexpr int TILE = detail::tile_rows_v<MemSpace>;
    using Vec = Kokkos::View<Scalar*, MemSpace>;
    const Index dim = r.extent(0);
    detail::ResidualFunctor<TILE, Block<Scalar, MemSpace>, Vec, Vec> f{V, HV, r, s, lambda, dim, m};
    double sum = 0.0;
    Kokkos::parallel_reduce("qkrylov::block_residual", Kokkos::RangePolicy<ExecSpace, Index>(0, (dim + TILE - 1) / TILE), f, sum);
    return static_cast<Real>(std::sqrt(sum));
}

/// out1 = sum_i c_i B1_i and out2 = sum_i c_i B2_i in one pass (a Ritz vector and its H-image).
template <class Scalar, class MemSpace>
void block_combine2(
    const Block<Scalar, MemSpace>& B1, const Block<Scalar, MemSpace>& B2, int m,
    const Kokkos::View<Scalar*, MemSpace>& c,
    Kokkos::View<Scalar*, MemSpace>& out1, Kokkos::View<Scalar*, MemSpace>& out2)
{
    using ExecSpace = typename MemSpace::execution_space;
    constexpr int TILE = detail::tile_rows_v<MemSpace>;
    using Vec = Kokkos::View<Scalar*, MemSpace>;
    const Index dim = out1.extent(0);
    detail::Combine2Functor<TILE, Block<Scalar, MemSpace>, Vec, Vec> f{B1, B2, out1, out2, c, dim, m};
    Kokkos::parallel_for("qkrylov::block_combine2", Kokkos::RangePolicy<ExecSpace, Index>(0, (dim + TILE - 1) / TILE), f);
}

}  // namespace linalg
}  // namespace QKRYLOV_PRECISION_NAMESPACE
}  // namespace qkrylov
