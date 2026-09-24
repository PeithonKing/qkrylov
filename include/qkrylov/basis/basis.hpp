#pragma once

#include "qkrylov/core/types.hpp"

/// \file basis.hpp
/// \brief Abstract base interface for many-body quantum Fock space bases.

namespace qkrylov {

/// \brief Abstract base class representing a many-body Hilbert space basis.
///
/// Maps between dense linear indices \f$i \in [0, D-1]\f$ and sparse Fock bitstrings \f$s \in \mathbb{N}\f$.
/// <TODO-LLM: Document asymptotic lookup complexity (O(1) binary search / hashing) and memory consumption tradeoffs here>
class Basis
{
public:

    virtual ~Basis() = default;
    virtual int bits_per_site() const { return 0; }

    /// \brief Total Hilbert space dimension \f$D\f$.
    /// \return Number of basis states in this sector.
    virtual Index size() const = 0;

    /// \brief Retrieve the Fock state bitstring corresponding to linear index \f$i\f$.
    /// \param i Zero-based linear index.
    /// \return 64-bit state bitstring.
    virtual StateID state(Index i) const = 0;

    /// \brief Search for the linear index corresponding to a Fock state bitstring.
    /// \param s 64-bit state bitstring.
    /// \return Linear index, or invalid index if state is not in sector.
    virtual Index index(StateID s) const = 0;

    /// \brief Check whether a state bitstring is contained within this basis sector.
    /// \param s 64-bit state bitstring.
    /// \return True if state belongs to this basis.
    virtual bool contains(StateID s) const = 0;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::Basis;
}

} // namespace qkrylov
