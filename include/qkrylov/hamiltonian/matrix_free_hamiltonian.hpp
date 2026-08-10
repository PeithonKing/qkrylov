#pragma once

#include "qkrylov/core/types.hpp"

#include "qkrylov/basis/basis.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/core/kokkos_types.hpp"
#include "qkrylov/core/device.hpp"

#include <memory>
#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


/// Matrix-free Hamiltonian with pre-compiled operator action.
///
/// At construction time, the operator sum is evaluated for every basis state
/// to build a CSR (Compressed Sparse Row) representation on the device.
/// This one-time cost eliminates per-apply virtual dispatch, string matching,
/// and hash-map lookups — enabling both GPU execution and faster CPU paths.
template <typename ExecSpace>
class MatrixFreeHamiltonian
{
public:

    MatrixFreeHamiltonian(
        std::shared_ptr<Basis> basis,
        std::shared_ptr<Site> site,
        const OpSum& ops,
        Device device = Device()
    );

    /// Apply H to a device-resident vector: y = H * x.
    /// No host↔device copies — both x and y must already live on the device.
    void apply(const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) const;

    /// Apply H to host pointers: y = H * x.
    /// Internally copies host→device, runs the kernel, copies device→host.
    /// This overload exists for the C API and Python binding.
    void apply(const Complex* x, Complex* y) const;

    /// Return the pre-computed diagonal of H as a device-resident vector.
    const VectorView<ExecSpace>& diagonal() const { return diagonal_; }

    /// Return the pre-computed diagonal of H as a host vector.
    HostVector diagonal_host() const;

    /// Hilbert space dimension.
    Index dimension() const { return dim_; }

    /// The device this Hamiltonian was built for.
    const Device& device() const { return device_; }

private:

    Device device_;
    Index dim_ = 0;

    // ---- Pre-compiled CSR representation (device-resident) ----
    // Row-centric storage: row alpha stores all (col, value) pairs
    // where value = conj(H[col][alpha]), enabling a gather-based
    // SpMV with y[alpha] = sum_j values[j] * x[cols[j]].
    // No atomics needed — each thread owns its output element.
    Kokkos::View<int*, typename ExecSpace::memory_space>      row_offsets_;  // size = dim + 1
    Kokkos::View<int*, typename ExecSpace::memory_space>      col_indices_;  // size = nnz
    Kokkos::View<KComplex*, typename ExecSpace::memory_space> values_;       // size = nnz

    VectorView<ExecSpace> diagonal_;  // size = dim

    // ---- Original objects (kept for reference/future use) ----
    std::shared_ptr<Basis> basis_;
    std::shared_ptr<Site>  site_;
    OpSum ops_;
};



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
