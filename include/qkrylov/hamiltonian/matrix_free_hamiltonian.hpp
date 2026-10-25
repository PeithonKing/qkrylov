#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/instruction.hpp"
#include "qkrylov/core/kokkos_types.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/core/traits.hpp"
#include "qkrylov/hamiltonian/state_index.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

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

    template <typename BasisType>
        requires std::is_base_of_v<Basis, std::decay_t<BasisType>>
    MatrixFreeHamiltonian(const BasisType& basis, std::shared_ptr<Site> site, const OpSum& ops, Device device = Device())
        : MatrixFreeHamiltonian(std::make_shared<std::decay_t<BasisType>>(basis), std::move(site), ops, device) {}

    void apply(const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) const;
    void apply(const Complex* x, Complex* y) const;
    /// y = H x and return <x|y> in one pass (saves the separate dot in Krylov loops).
    KComplex apply_dot(const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) const;

    /// True if every coefficient is real (H is a real symmetric matrix in this basis).
    bool has_real_coefficients() const { return real_coeffs_; }
    /// Real-vector variants, valid only when has_real_coefficients(): half the memory traffic.
    void apply_real(const RealVectorView<ExecSpace>& x, RealVectorView<ExecSpace>& y) const;
    Real apply_dot_real(const RealVectorView<ExecSpace>& x, RealVectorView<ExecSpace>& y) const;
    
    /// Diagonal as complex values (built lazily for real Hamiltonians, which store it as Real).
    const VectorView<ExecSpace>& diagonal() const;
    HostVector diagonal_host() const;
    Index dimension() const { return dim_; }
    const Device& device() const { return device_; }
    void init();

private:
    /// Pick the fastest row kernel for this Hamiltonian and hand it to op(row).
    template <typename Scalar, typename Op>
    void dispatch_rows(const Kokkos::View<Scalar*, typename ExecSpace::memory_space>& x, const Op& op) const;

    Device device_;
    Index dim_ = 0;
    
    Kokkos::View<qkrylov::Instruction*, ExecSpace> instructions_;
    int num_instructions_ = 0;
    bool identity_basis_ = false;
    bool real_coeffs_ = true;                          ///< all coefficients real: cheaper multiply                     ///< state(i) == i: no state array stored
    Kokkos::View<StateID*, ExecSpace> basis_states_;  ///< empty for identity bases
    StateIndex<ExecSpace> index_;                     ///< state -> index lookup (Lin / bisect / identity)

    
    mutable VectorView<ExecSpace> diagonal_;     ///< complex diagonal (complex H, or lazily for diagonal())
    RealVectorView<ExecSpace> diagonal_real_;    ///< real diagonal (real H)
    // Hot-path copy of the off-diagonal instructions as flat arrays (binary sites only)
    bool all_binary_ = false;
    Kokkos::View<uint64_t*, ExecSpace> i_flip_, i_check_, i_expected_, i_alt_, i_sign_;
    // Byte-lane validity tables: for chunk c (64 instructions) and byte lane k,
    // lane_tab[(c*n_lanes + k)*256 + v] = bitmask of the chunk's lane-local instructions
    // that apply to a state whose byte k equals v. Instructions whose check bits straddle
    // lanes are listed in gen_idx and tested individually.
    int n_lanes_ = 0;
    int n_chunks_ = 0;
    Kokkos::View<uint64_t*, ExecSpace> lane_tab_;
    Kokkos::View<int32_t*, ExecSpace> gen_idx_, gen_start_;
    Kokkos::View<uint8_t*, ExecSpace> i_safe_;   ///< 1: every valid hop lands in the basis, skip bounds/class checks
    Kokkos::View<Real*, ExecSpace> i_coeff_re_, i_coeff_im_;
    mutable VectorView<ExecSpace> scratch_x_;
    mutable VectorView<ExecSpace> scratch_y_;

    std::shared_ptr<Basis> basis_;
    std::shared_ptr<Site>  site_;
    OpSum ops_; // Holds std::complex<double> intrinsically.
};

template <typename BasisType, typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(const BasisType&, std::shared_ptr<Site>, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

template <typename BasisType>
MatrixFreeHamiltonian(const BasisType&, std::shared_ptr<Site>, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

template <typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

MatrixFreeHamiltonian(std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
using Hamiltonian = MatrixFreeHamiltonian<ExecSpace>;

} // namespace QKRYLOV_PRECISION_NAMESPACE

using QKRYLOV_PRECISION_NAMESPACE::Hamiltonian;

} // namespace qkrylov
