#pragma once

#include "qkrylov/core/types.hpp"
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

/// Complex type for device-side computation.
/// Layout-compatible with std::complex<float> (guaranteed by both
/// the C++ standard and the Kokkos project).
using KComplex = Kokkos::complex<Real>;

/// Device-resident 1-D vector of complex doubles.
using Vector = Kokkos::View<KComplex*>;

template <typename ExecSpace>
using VectorView = Kokkos::View<KComplex*, typename ExecSpace::memory_space>;

/// The execution space selected at compile time by the Kokkos backend.
using ExecutionSpace = Kokkos::DefaultExecutionSpace;

/// Memory space associated with the default execution space.
using MemorySpace = ExecutionSpace::memory_space;

/// Host-space execution/memory space.
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
