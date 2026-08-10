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
};

template <typename ExecSpace>
DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig = 1,
    int max_subspace = 20,
    Real tol = 1.0e-8
);

}

}
