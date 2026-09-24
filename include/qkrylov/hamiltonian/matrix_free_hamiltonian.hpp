#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/instruction.hpp"
#include "qkrylov/core/kokkos_types.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/core/traits.hpp"

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
    
    const VectorView<ExecSpace>& diagonal() const { return diagonal_; }
    HostVector diagonal_host() const;
    Index dimension() const { return dim_; }
    const Device& device() const { return device_; }

private:
    
    Device device_;
    Index dim_ = 0;
    
    Kokkos::View<qkrylov::Instruction*, ExecSpace> instructions_;
    int num_instructions_ = 0;

    
    VectorView<ExecSpace> diagonal_;
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
