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

    /// \brief True if state(i) == i for every i (full space with implicit indexing).
    virtual bool is_identity() const { return false; }

    /// \brief The sorted state list, if the basis stores one (nullptr otherwise).
    virtual const std::vector<StateID>* state_list() const { return nullptr; }

    /// \brief Bit masks whose popcounts are fixed inside one symmetry sector.
    ///
    /// Used to build O(1) state->index tables; the tables are verified against the
    /// state list, so a wrong answer here only costs speed, never correctness.
    virtual std::vector<uint64_t> conserved_masks() const { return {~0ULL}; }
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::Basis;
}

} // namespace qkrylov
