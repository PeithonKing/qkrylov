#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


struct DynamicsResult
{
    std::vector<Real> alphas;
    std::vector<Real> betas;
    Real norm_phi0;
};

// Compute the Continued Fraction coefficients starting from a vector phi0
template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_iter = 100
);

// Helper function to evaluate S(omega) from coefficients
// S(omega) = -1/pi * Im <phi0 | (omega - H + E0 + i*eta)^-1 | phi0>
Real evaluate_spectral_function(
    const Real* alphas,
    const Real* betas,
    size_t n,
    Real norm_phi0,
    Real omega,
    Real E0,
    Real eta = 0.1
);

}

}
