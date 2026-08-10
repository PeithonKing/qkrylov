#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


struct FTLMResult
{
    Real beta;
    Real partition_function = 0.0;
    Real internal_energy = 0.0;
    Real specific_heat = 0.0;
    // We can add more observables later
};

template <typename ExecSpace>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random = 50,
    int n_steps = 100
);

}

}
