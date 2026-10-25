#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "krylov_detail.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

// Fused device kernels for the CG iteration (functors so the same code builds with NVCC). The vector type is
// real when H and the right-hand side are real, complex otherwise.
namespace cv_detail {

KOKKOS_INLINE_FUNCTION double re_dot(Real a, Real b) { return static_cast<double>(a) * b; }
KOKKOS_INLINE_FUNCTION double re_dot(KComplex a, KComplex b) {
    return static_cast<double>(a.real()) * b.real() + static_cast<double>(a.imag()) * b.imag();
}

/// z -= shift * y
template <class V> struct ShiftSub {
    V z, y;
    Real shift;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i) const { z(i) -= shift * y(i); }
};

/// Ap = (H z) - shift z + eta^2 p  (Ap holds H z on entry); also returns Re <p|Ap>
template <class V> struct ApFinish {
    V ap, z, p;
    Real shift, eta_sq;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, double& sum) const {
        const typename V::non_const_value_type v = ap(i) - shift * z(i) + eta_sq * p(i);
        ap(i) = v;
        sum += re_dot(p(i), v);
    }
};

/// x += a p ; r -= a Ap ; returns ||r||^2
template <class V> struct CgUpdate {
    V x, r, p, ap;
    Real a;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, double& sum) const {
        x(i) += a * p(i);
        const typename V::non_const_value_type rr = r(i) - a * ap(i);
        r(i) = rr;
        sum += abs2(rr);
    }
};

/// p = r + beta p
template <class V> struct DirUpdate {
    V p, r;
    Real beta;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i) const { p(i) = r(i) + beta * p(i); }
};

/// sum_i |x_i|^2
template <class V> struct Norm2Functor {
    V x;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, double& sum) const { sum += abs2(x(i)); }
};
/// Re <a|b>
template <class V> struct InnerFunctor {
    V a, b;
    KOKKOS_INLINE_FUNCTION void operator()(const Index i, double& sum) const { sum += re_dot(a(i), b(i)); }
};

template <class ExecSpace, class V>
Real norm2_of(const V& x) {
    double s = 0.0;
    Kokkos::parallel_reduce("qkrylov::cv_norm2", Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
                            Norm2Functor<V>{x}, s);
    return static_cast<Real>(s);
}

template <class ExecSpace, class V>
CorrectionVectorResult run(
    const MatrixFreeHamiltonian<ExecSpace>& H, const HostVector& Op_psi0,
    Real E0, Real omega, Real eta, int maxiter, Real tol)
{
    using namespace krylov_detail;
    const Index dim = H.dimension();
    using RP = Kokkos::RangePolicy<ExecSpace, Index>;

    V dev_Op_psi0 = Vec_like<V>("qkrylov::cv::Op_psi0", dim);
    upload_vector<ExecSpace>(Op_psi0, dev_Op_psi0);

    // Right hand side: b = eta * Op_psi0
    V b = Vec_like<V>("qkrylov::cv::b", dim);
    scale_into(b, dev_Op_psi0, eta);

    const Real b_norm = std::sqrt(norm2_of<ExecSpace>(b));
    const Real b_min = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (b_norm < b_min) {
        return {HostVector(dim, Complex(0.0, 0.0)), static_cast<Real>(0.0), 0, true};
    }

    // Workspace
    V z = Vec_like<V>("qkrylov::cv::z", dim);
    V Ap = Vec_like<V>("qkrylov::cv::Ap", dim);
    V x = Vec_like<V>("qkrylov::cv::x", dim);
    V r = Vec_like<V>("qkrylov::cv::r", dim);
    V p = Vec_like<V>("qkrylov::cv::p", dim);

    const Real shift = E0 + omega;
    const Real eta_sq = eta * eta;

    // Conjugate Gradient on A y = b with A = (H - shift)^2 + eta^2
    Kokkos::deep_copy(x, typename V::non_const_value_type{});
    Kokkos::deep_copy(r, b);
    Kokkos::deep_copy(p, r);

    Real rs_old = norm2_of<ExecSpace>(r);

    bool converged = false;
    int iter = 0;
    const Real denom_min = std::numeric_limits<Real>::epsilon() * std::numeric_limits<Real>::epsilon();

    for (iter = 0; iter < maxiter; ++iter) {
        // Ap = A p in three passes: z = H p - shift p ; Ap = H z ; Ap += -shift z + eta^2 p (with Re <p|Ap>)
        apply_h(H, p, z);
        Kokkos::parallel_for("qkrylov::cv_shift", RP(0, dim), ShiftSub<V>{z, p, shift});
        apply_h(H, z, Ap);
        double denom_acc = 0.0;
        Kokkos::parallel_reduce("qkrylov::cv_ap", RP(0, dim), ApFinish<V>{Ap, z, p, shift, eta_sq}, denom_acc);
        const Real denom = static_cast<Real>(denom_acc);
        if (std::abs(denom) < denom_min) {
            break;
        }

        const Real alpha = rs_old / denom;

        double rs_acc = 0.0;
        Kokkos::parallel_reduce("qkrylov::cv_update", RP(0, dim), CgUpdate<V>{x, r, p, Ap, alpha}, rs_acc);
        Real rs_new = static_cast<Real>(rs_acc);
        if (rs_new < static_cast<Real>(0.0)) {
            rs_new = static_cast<Real>(0.0);
        }

        if (std::sqrt(rs_new) / b_norm < tol) {
            converged = true;
            iter++;
            break;
        }

        const Real beta_cg = rs_new / rs_old;
        Kokkos::parallel_for("qkrylov::cv_dir", RP(0, dim), DirUpdate<V>{p, r, beta_cg});

        rs_old = rs_new;
    }

    // Spectral function S(omega) = (1/pi) * Re<Op_psi0 | x>
    double inner_acc = 0.0;
    Kokkos::parallel_reduce("qkrylov::cv_inner", RP(0, dim), InnerFunctor<V>{dev_Op_psi0, x}, inner_acc);
    constexpr Real PI = static_cast<Real>(3.141592653589793238462643383279502884L);
    Real spectral_func = (static_cast<Real>(1.0) / PI) * static_cast<Real>(inner_acc);

    HostVector host_x;
    to_host_vector<ExecSpace>(x, host_x);

    CorrectionVectorResult result;
    result.correction_vector = std::move(host_x);
    result.spectral_function = spectral_func;
    result.iterations = iter;
    result.converged = converged;

    return result;
}

}  // namespace cv_detail

template <typename ExecSpace>
CorrectionVectorResult correction_vector_spectral(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& Op_psi0,
    Real E0,
    Real omega,
    Real eta,
    int maxiter,
    Real tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) {
        return {HostVector{}, static_cast<Real>(0.0), 0, false};
    }
    if (Op_psi0.size() != dim) {
        throw std::invalid_argument("correction_vector_spectral: Op_psi0 size must match Hamiltonian dimension");
    }

    // A real Hamiltonian with a real right-hand side keeps the whole CG in real vectors: half the memory traffic.
    if (H.has_real_coefficients() && krylov_detail::all_real(Op_psi0)) {
        return cv_detail::run<ExecSpace, RealVectorView<ExecSpace>>(H, Op_psi0, E0, omega, eta, maxiter, tol);
    }
    return cv_detail::run<ExecSpace, VectorView<ExecSpace>>(H, Op_psi0, E0, omega, eta, maxiter, tol);
}

// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template CorrectionVectorResult correction_vector_spectral<Kokkos::Serial>(
    const MatrixFreeHamiltonian<Kokkos::Serial>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template CorrectionVectorResult correction_vector_spectral<Kokkos::OpenMP>(
    const MatrixFreeHamiltonian<Kokkos::OpenMP>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template CorrectionVectorResult correction_vector_spectral<Kokkos::Threads>(
    const MatrixFreeHamiltonian<Kokkos::Threads>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template CorrectionVectorResult correction_vector_spectral<Kokkos::Cuda>(
    const MatrixFreeHamiltonian<Kokkos::Cuda>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_HIP
template CorrectionVectorResult correction_vector_spectral<Kokkos::HIP>(
    const MatrixFreeHamiltonian<Kokkos::HIP>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_SYCL
template CorrectionVectorResult correction_vector_spectral<Kokkos::Experimental::SYCL>(
    const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
