#pragma once

#include "qkrylov/core/types.hpp"

#include "qkrylov/operators/local_action.hpp"

#include <string>

/// \file site.hpp
/// \brief Abstract interface defining local quantum site physics and operator transitions.

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

/// \brief Abstract base class defining local degrees of freedom on a single lattice site.
/// <TODO-LLM: Detail virtual dispatch role in matrix element evaluation and transition to matrix-free CSR kernels here>
class Site
{
public:

    virtual ~Site() = default;

    /// \brief Apply a local operator to a many-body Fock state bitstring.
    /// \param op Operator name string (e.g. "Sz", "Sp", "Sm", "CdagUp", "CUp").
    /// \param site Zero-based lattice site index.
    /// \param state Current Fock state bitstring.
    /// \return Transition struct containing validity flag, new state bitstring, and complex matrix element.
    virtual LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const = 0;
};

}
}
