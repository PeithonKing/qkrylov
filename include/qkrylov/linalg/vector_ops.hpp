#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/core/kokkos_types.hpp"

#include <cmath>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



/// Compute the inner product <x|y> = sum_i conj(x_i) * y_i
///
/// Each product is formed in working precision but the sum is accumulated in double,
/// so single-precision runs don't lose accuracy over 1e8+ element reductions.
template <typename ViewType>
inline KComplex dot(
    const ViewType& x,
    const ViewType& y
)
{
    using ExecSpace = typename ViewType::execution_space;
    using Acc = Kokkos::complex<double>;
    Acc result(0.0, 0.0);

    Kokkos::parallel_reduce("qkrylov::dot",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i, Acc& sum) {
            const KComplex p = Kokkos::conj(x(i)) * y(i);
            sum += Acc(p.real(), p.imag());
        },
        result
    );

    return KComplex(static_cast<Real>(result.real()), static_cast<Real>(result.imag()));
}

/// Compute the 2-norm ||x||
template <typename ViewType>
inline Real norm(
    const ViewType& x
)
{
    return std::sqrt(
        dot(x, x).real()
    );
}

/// Scale: x = a * x
template <typename ViewType>
inline void scal(
    KComplex a,
    ViewType& x
)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::scal",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) {
            x(i) *= a;
        }
    );
}

/// Scale: x = a * x  (real scalar convenience overload)
template <typename ViewType>
inline void scal(
    Real a,
    ViewType& x
)
{
    scal(KComplex(a, 0.0), x);
}

/// AXPY: y = a*x + y
template <typename ViewType>
inline void axpy(
    KComplex a,
    const ViewType& x,
    ViewType& y
)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::axpy",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) {
            y(i) += a * x(i);
        }
    );
}

/// Normalize x to unit length (in-place).
template <typename ViewType>
inline void normalize(
    ViewType& x
)
{
    const Real n = norm(x);
    if (n == 0.0) return;
    scal(KComplex(1.0 / n, 0.0), x);
}

/// Real/complex overloads used inside device lambdas. NVCC rejects a lambda whose captured
/// variable is first used inside `if constexpr`, so the type dispatch lives in overloads.
KOKKOS_INLINE_FUNCTION double abs2(Real v) { return static_cast<double>(v) * v; }
KOKKOS_INLINE_FUNCTION double abs2(KComplex v) {
    return static_cast<double>(v.real()) * v.real() + static_cast<double>(v.imag()) * v.imag();
}
KOKKOS_INLINE_FUNCTION Real make_random_value(Real, double u, double) { return static_cast<Real>(u); }
KOKKOS_INLINE_FUNCTION KComplex make_random_value(KComplex, double u, double v) {
    return KComplex(static_cast<Real>(u), static_cast<Real>(v));
}
/// <x|v> accumulated in double (real: x*v; complex: conj(x)*v).
KOKKOS_INLINE_FUNCTION double dot_term(Real x, Real v) { return static_cast<double>(x) * v; }
KOKKOS_INLINE_FUNCTION Kokkos::complex<double> dot_term(KComplex x, KComplex v) {
    const KComplex p = Kokkos::conj(x) * v;
    return Kokkos::complex<double>(p.real(), p.imag());
}

/// w <- w - a*v - b*p (p skipped when has_p is false) and return ||w||^2, in one pass.
/// The Lanczos three-term update fused with the norm of the new residual.
template <typename ViewType>
inline Real lanczos_update_norm2(
    ViewType& w, const ViewType& v, const ViewType& p, Real a, Real b, bool has_p
)
{
    using ExecSpace = typename ViewType::execution_space;
    double result = 0.0;
    Kokkos::parallel_reduce("qkrylov::lanczos_update",
        Kokkos::RangePolicy<ExecSpace, Index>(0, w.extent(0)),
        KOKKOS_LAMBDA(const Index i, double& sum) {
            typename ViewType::non_const_value_type r = w(i) - a * v(i);
            if (has_p) r -= b * p(i);
            w(i) = r;
            sum += abs2(r);
        },
        result
    );
    return static_cast<Real>(result);
}

/// dst <- s * src (one pass; replaces deep_copy + scal).
template <typename ViewType>
inline void scale_into(ViewType& dst, const ViewType& src, Real s)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::scale_into",
        Kokkos::RangePolicy<ExecSpace, Index>(0, src.extent(0)),
        KOKKOS_LAMBDA(const Index i) { dst(i) = s * src(i); }
    );
}

KOKKOS_INLINE_FUNCTION uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Deterministic, parallel pseudo-random fill with entries uniform in [-1, 1) (+ i[-1, 1) if complex).
/// Counter-based (splitmix64 of seed and index), so the result does not depend on thread
/// scheduling -- two-pass solvers can regenerate exactly the same start vector.
template <typename ViewType>
inline void fill_random(ViewType& x, uint64_t seed)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::fill_random",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) {
            const uint64_t a = splitmix64(seed ^ (2 * static_cast<uint64_t>(i)));
            const uint64_t b = splitmix64(seed ^ (2 * static_cast<uint64_t>(i) + 1));
            const double inv = 1.0 / 9007199254740992.0; // 2^-53
            x(i) = make_random_value(typename ViewType::non_const_value_type{},
                                     (a >> 11) * inv * 2.0 - 1.0, (b >> 11) * inv * 2.0 - 1.0);
        }
    );
}

/// Fill a device vector with zeros.
template <typename ViewType>
inline void zero_fill(ViewType& x) {
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::deep_copy(ExecSpace(), x, KComplex(0.0, 0.0));
}

/// Deep-copy src into dst (same layout, same extent).
template <typename ViewType1, typename ViewType2>
inline void deep_copy(ViewType1& dst, const ViewType2& src) {
    using ExecSpace = typename ViewType1::execution_space;
    Kokkos::deep_copy(ExecSpace(), dst, src);
}

/// Copy a HostVector (std::vector<Complex>) into a device Vector.
/// The device view must already be allocated with the correct size.
template <typename ViewType>
inline void copy_host_to_device(const HostVector& host, ViewType& device) {
    using ExecSpace = typename ViewType::execution_space;
    auto h_view = Kokkos::View<const KComplex*, Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(host.data()),
        host.size()
    );
    Kokkos::deep_copy(ExecSpace(), device, h_view);
}

/// Copy a device Vector into a HostVector (std::vector<Complex>).
/// Resizes the host vector to match.
template <typename ViewType>
inline void copy_device_to_host(const ViewType& device, HostVector& host) {
    host.resize(device.extent(0));
    auto h_view = Kokkos::View<KComplex*, Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<KComplex*>(host.data()),
        host.size()
    );
    Kokkos::deep_copy(h_view, device);
}



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
