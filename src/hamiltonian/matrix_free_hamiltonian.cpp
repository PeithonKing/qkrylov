#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

// Safe, fast, portable popcount for Kokkos device code
KOKKOS_INLINE_FUNCTION int qkrylov_popcount64(uint64_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

// Applies a single Instruction to a state, returning the new state.

template <typename ExecSpace>
MatrixFreeHamiltonian<ExecSpace>::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops,
    Device device
)
    : basis_(std::move(basis)),
      site_(std::move(site)),
      ops_(ops),
      device_(std::move(device))
{
    if (!basis_ || !site_) {
        throw std::invalid_argument("MatrixFreeHamiltonian requires both Basis and Site.");
    }
    if (basis_->bits_per_site() != site_->bits_per_site()) {
        throw std::invalid_argument(
            "Basis and Site dimension mismatch (bits_per_site: " + 
            std::to_string(basis_->bits_per_site()) + " vs " + 
            std::to_string(site_->bits_per_site()) + "). " +
            "You are mixing incompatible physics rules."
        );
    }

    detail::ensure_kokkos_initialized();
    dim_ = basis_->size();
    if (dim_ == 0) return;

    // 1. State list -> device, plus the state->index lookup (no virtual calls in kernels).
    //    Identity bases (full spin-1/2 space) store nothing: state == index.
    identity_basis_ = basis_->is_identity();
    int fp_nbits = 0, fp_k = 0;
    bool generated = false;
    if (!identity_basis_ && basis_->fixed_popcount_sector(fp_nbits, fp_k)) {
        // A single fixed-popcount sector: lookup tables come from binomial formulas and the
        // state list is written in parallel on this Hamiltonian's device -- the basis never
        // builds (or sorts, or uploads) an O(dim) list.
        generated = build_fixed_popcount_index<ExecSpace>(fp_nbits, fp_k, index_, basis_states_);
    }
    if (!generated) {
        const std::vector<StateID>* list = basis_->state_list();
        std::vector<StateID> owned;
        if (!identity_basis_ && !list) {
            owned.resize(dim_);
            for (Index i = 0; i < dim_; ++i) owned[i] = basis_->state(i);
            list = &owned;
        }
        if (!identity_basis_) {
            identity_basis_ = true;
            for (Index i = 0; i < dim_; ++i) {
                if ((*list)[i] != static_cast<StateID>(i)) { identity_basis_ = false; break; }
            }
        }
        if (identity_basis_) {
            basis_states_ = Kokkos::View<StateID*, ExecSpace>();
            index_ = make_identity_index<ExecSpace>(dim_);
        } else {
            if (std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace> && list != &owned) {
                // Host execution: alias the basis' own state vector instead of copying it
                // (saves 8 bytes per state; H keeps the basis alive via basis_).
                basis_states_ = Kokkos::View<StateID*, ExecSpace, Kokkos::MemoryUnmanaged>(
                    const_cast<StateID*>(list->data()), dim_);
            } else {
                basis_states_ = Kokkos::View<StateID*, ExecSpace>(
                    Kokkos::view_alloc("qkrylov::basis_states", Kokkos::WithoutInitializing), dim_);
                Kokkos::View<const StateID*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(list->data(), dim_);
                Kokkos::deep_copy(basis_states_, h);
            }
            index_ = build_state_index<ExecSpace>(*list, basis_->conserved_masks(), basis_states_);
        }
    }

    // 2. Compile VM instructions on host and split diag/offdiag
    std::vector<Instruction> diag_insts;
    std::vector<Instruction> offdiag_insts;
    for (const auto& term : ops_.terms()) {
        auto compiled = site_->compile(term);
        for (const auto& inst : compiled) {
            bool is_diagonal;
            if (inst.based.d == 0) {
                is_diagonal = (inst.bin.flip_mask == 0);
            } else {
                is_diagonal = true;
                for(int k=0; k<inst.based.num_sites; ++k) {
                    if (inst.based.src_sigma[k] != inst.based.tgt_sigma[k]) { is_diagonal = false; break; }
                }
            }
            if (is_diagonal) {
                diag_insts.push_back(inst);
            } else {
                offdiag_insts.push_back(inst);
            }
        }
    }

    // 3. Setup and execute Diagonal VM on device
    int num_diag = diag_insts.size();
    Kokkos::View<Instruction*, ExecSpace> d_diag_inst("diag_inst", num_diag);
    if (num_diag > 0) {
        auto h_diag_inst = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(diag_insts.data(), num_diag);
        Kokkos::deep_copy(ExecSpace(), d_diag_inst, h_diag_inst);
    }
    
    diagonal_ = VectorView<ExecSpace>("qkrylov::diagonal", dim_);
    auto diag_view = diagonal_;
    auto states_view = basis_states_;
    const bool identity = identity_basis_;

    Kokkos::parallel_for("qkrylov::init_diagonal",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim_),
        KOKKOS_LAMBDA(const Index alpha) {
            const StateID state = identity ? static_cast<StateID>(alpha) : states_view(alpha);
            double diag_r = 0.0;
            double diag_i = 0.0;
            for (int i = 0; i < num_diag; ++i) {
                const auto& inst = d_diag_inst(i);
                // Binary: bitmask check. Base-d: modular residue check.
                if (inst.based.d == 0) {
                    if ((state & inst.check_mask) != inst.expected_bits) continue;
                } else {
                    uint64_t d = inst.based.d;
                    bool valid = true;
                    for(int k=0; k<inst.based.num_sites; ++k) {
                        uint64_t d_pow = 1; for(int p=0; p<inst.based.site[k]; ++p) d_pow *= d;
                        uint64_t local = (state / d_pow) % d;
                        if (local != inst.based.tgt_sigma[k]) { valid = false; break; }
                    }
                    if (!valid) continue;
                }

                double real_c = reinterpret_cast<const double*>(&inst.coeff)[0];
                double imag_c = reinterpret_cast<const double*>(&inst.coeff)[1];
                
                if (inst.sign_mask) {
                    if (qkrylov_popcount64(state & inst.sign_mask) % 2 != 0) {
                        real_c = -real_c;
                        imag_c = -imag_c;
                    }
                }
                diag_r += real_c;
                diag_i += imag_c;
            }
            diag_view(alpha) = KComplex(static_cast<Real>(diag_r), static_cast<Real>(diag_i));
        }
    );

    // 4. Store off-diagonal VM instructions
    num_instructions_ = offdiag_insts.size();
    if (num_instructions_ > 0) {
        instructions_ = Kokkos::View<Instruction*, ExecSpace>("qkrylov::instructions", num_instructions_);
        auto host_view = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(offdiag_insts.data(), num_instructions_);
        Kokkos::deep_copy(ExecSpace(), instructions_, host_view);
    }
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;
    Kokkos::deep_copy(ExecSpace(), y, KComplex(0.0, 0.0));

    auto  diag_view = diagonal_;
    auto  inst_view = instructions_;
    int   num_inst  = num_instructions_;
    auto  states_view = basis_states_;
    auto  index = index_;
    const bool identity = identity_basis_;

    Kokkos::parallel_for("qkrylov::H_apply",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
        KOKKOS_LAMBDA(const Index beta) {
            const StateID state_beta = identity ? static_cast<StateID>(beta) : states_view(beta);
            
            // 1. Diagonal contribution
            KComplex y_val = diag_view(beta) * x(beta);

            // 2. Off-diagonal VM execution
            for (int i = 0; i < num_inst; ++i) {
                const auto& inst = inst_view(i);
                
                StateID state_alpha;
                if (inst.based.d == 0) {
                    // Binary site: reverse XOR to find source state alpha
                    state_alpha = state_beta ^ inst.bin.flip_mask;
                    if ((state_alpha & inst.check_mask) != inst.expected_bits) continue;
                } else {
                    // Explicit base-d loop:
                    uint64_t d = inst.based.d;
                    bool valid = true;
                    state_alpha = state_beta;
                    for(int k=0; k<inst.based.num_sites; ++k) {
                        uint64_t d_pow = 1; for(int p=0; p<inst.based.site[k]; ++p) d_pow *= d;
                        uint64_t local = (state_beta / d_pow) % d;
                        // For off-diagonal, state_beta has tgt_sigma. For diagonal, src==tgt.
                        if (local != inst.based.tgt_sigma[k]) { valid = false; break; }
                        state_alpha = state_alpha - inst.based.tgt_sigma[k] * d_pow + inst.based.src_sigma[k] * d_pow;
                    }
                    if (!valid) continue;
                }
                
                Index alpha;
                if (!index.find(state_alpha, alpha)) continue;
                
                double real_c = reinterpret_cast<const double*>(&inst.coeff)[0];
                double imag_c = reinterpret_cast<const double*>(&inst.coeff)[1];
                
                if (inst.sign_mask) {
                    if (qkrylov_popcount64(state_alpha & inst.sign_mask) % 2 != 0) {
                        real_c = -real_c;
                        imag_c = -imag_c;
                    }
                }
                
                KComplex term_val(static_cast<Real>(real_c), static_cast<Real>(imag_c));
                y_val += term_val * x(alpha);
            }
            y(beta) = y_val;
        }
    );
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const Complex* x,
    Complex* y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;

    if (scratch_x_.extent(0) != dim) {
        scratch_x_ = VectorView<ExecSpace>("scratch_x", dim);
        scratch_y_ = VectorView<ExecSpace>("scratch_y", dim);
    }

    auto x_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<const KComplex*>(x), dim);
    Kokkos::deep_copy(ExecSpace(), scratch_x_, x_host);

    apply(scratch_x_, scratch_y_);

    auto y_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<KComplex*>(y), dim);
    Kokkos::deep_copy(ExecSpace(), y_host, scratch_y_);
}

template <typename ExecSpace>
HostVector MatrixFreeHamiltonian<ExecSpace>::diagonal_host() const
{
    HostVector result;
    copy_device_to_host(diagonal_, result);
    return result;
}

#ifdef KOKKOS_ENABLE_SERIAL
template class MatrixFreeHamiltonian<Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class MatrixFreeHamiltonian<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class MatrixFreeHamiltonian<Kokkos::Threads>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class MatrixFreeHamiltonian<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_HIP
template class MatrixFreeHamiltonian<Kokkos::HIP>;
#endif
#ifdef KOKKOS_ENABLE_SYCL
template class MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>;
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
