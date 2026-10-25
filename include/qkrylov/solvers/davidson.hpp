#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


struct DavidsonResult
{
    std::vector<Real> eigenvalues;
    std::vector<HostVector> eigenvectors;
    int iterations = 0;
    bool converged = false;
};

/// Lowest `n_eig` eigenpairs by the Davidson method with a diagonal preconditioner.
///
/// - `max_subspace`: largest number of stored vectors (kept as columns of one device block that grows on demand);
///   when it would be exceeded the subspace restarts from the wanted Ritz vectors plus an equal number of the next ones.
/// - `tol`: convergence threshold on the residual norm ||H v - lambda v|| of every wanted pair.
/// - `maxiter`: most outer iterations (one correction per unconverged pair each); if it is reached the result has
///   `converged == false` and `iterations == maxiter`.
/// - Real Hamiltonians run in real arithmetic; complex ones in complex arithmetic. Eigenvectors are returned as
///   normalised complex host vectors either way.
/// - Starting vectors are unit vectors on the lowest diagonal entries; results are deterministic and identical
///   on CPU and GPU up to rounding.
template <typename ExecSpace>
DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig = 1,
    int max_subspace = 20,
    Real tol = 1.0e-8,
    int maxiter = 100
);

}

}
