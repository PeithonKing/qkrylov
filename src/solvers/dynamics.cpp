#include "qkrylov/solvers/dynamics.hpp"

#include <cmath>
#include <complex>
#include <Kokkos_Core.hpp>

namespace qkrylov
{

template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_iter
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    VectorView<ExecSpace> dev_phi0("dev_phi0", dim);
    copy_host_to_device(phi0, dev_phi0);

    double norm_phi = norm(dev_phi0);
    if (norm_phi < 1e-15) return { {}, {}, 0.0 };

    VectorView<ExecSpace> v_curr("v_curr", dim);
    Kokkos::deep_copy(v_curr, dev_phi0);
    scal(1.0/norm_phi, v_curr);

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> w("w", dim);

    DynamicsResult res;
    res.norm_phi0 = norm_phi;

    for (int iter = 0; iter < n_iter; ++iter) {
        H.apply(v_curr, w);

        double alpha = dot(v_curr, w).real();
        res.alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-res.betas.back(), v_prev, w);
        }

        double beta = norm(w);
        if (beta < 1e-15) break;

        res.betas.push_back(beta);
        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);
    }

    return res;
}

double evaluate_spectral_function(
    const double* alphas,
    const double* betas,
    size_t n,
    double norm_phi0,
    double omega,
    double E0,
    double eta
)
{
    if (n == 0) return 0.0;

    std::complex<double> z(omega + E0, eta);

    // Backward recursion for continued fraction
    // C_n = 1 / (z - a_n)
    // C_{i} = 1 / (z - a_i - b_i^2 * C_{i+1})

    std::complex<double> f = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        if (i == n - 1) {
            f = 1.0 / (z - alphas[i]);
        } else {
            f = 1.0 / (z - alphas[i] - betas[i] * betas[i] * f);
        }
    }

    return -1.0 / M_PI * std::imag(norm_phi0 * norm_phi0 * f);
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template DynamicsResult continued_fraction_coeffs<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template DynamicsResult continued_fraction_coeffs<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template DynamicsResult continued_fraction_coeffs<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template DynamicsResult continued_fraction_coeffs<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_HIP
template DynamicsResult continued_fraction_coeffs<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, const HostVector&, int);
#endif
}
