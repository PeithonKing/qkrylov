import re

with open("src/hamiltonian/matrix_free_hamiltonian.cpp", "r") as f:
    content = f.read()

# 1. Constructor modification
# We need to add the compile phase AFTER the diagonal phase.
compile_logic = """
    auto h_dg = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_diagonal.data(), dim_);
    Kokkos::deep_copy(ExecSpace(), diagonal_, h_dg);

    // --- NEW: VM Compilation Phase ---
    std::vector<Instruction> host_insts;
    for (const auto& term : ops_.terms()) {
        auto compiled = site_->compile(term);
        for (auto& inst : compiled) {
            // Only store OFF-DIAGONAL instructions in the VM to avoid double-counting
            // the diagonal (which is already fully captured in diagonal_ above).
            if (inst.flip_mask != 0) {
                host_insts.push_back(inst);
            }
        }
    }
    num_instructions_ = host_insts.size();
    if (num_instructions_ > 0) {
        instructions_ = Kokkos::View<Instruction*, ExecSpace>("qkrylov::instructions", num_instructions_);
        auto host_view = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(host_insts.data(), num_instructions_);
        Kokkos::deep_copy(ExecSpace(), instructions_, host_view);
    }
}
"""
content = re.sub(r'auto h_dg.*?diagonal_, h_dg\);\n\}', compile_logic, content, flags=re.DOTALL)


# 2. apply() modification
apply_logic = """
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

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;
    Kokkos::deep_copy(ExecSpace(), y, KComplex(0.0, 0.0));

    auto* basis_ptr = basis_.get();
    auto  diag_view = diagonal_;
    auto  inst_view = instructions_;
    int   num_inst  = num_instructions_;

    Kokkos::parallel_for("qkrylov::H_apply",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
        KOKKOS_LAMBDA(const Index alpha) {
            const StateID state = basis_ptr->state(alpha);
            const KComplex xin = x(alpha);

            // 1. Diagonal contribution
            Kokkos::atomic_add(&y(alpha), diag_view(alpha) * xin);

            // 2. Off-diagonal VM execution
            for (int i = 0; i < num_inst; ++i) {
                const auto& inst = inst_view(i);
                
                if ((state & inst.check_mask) != inst.expected_bits) continue;

                StateID new_state = state ^ inst.flip_mask;
                if (!basis_ptr->contains(new_state)) continue;
                
                Index beta = basis_ptr->index(new_state);
                
                double real_c = inst.coeff.real();
                double imag_c = inst.coeff.imag();
                
                if (inst.sign_mask) {
                    if (qkrylov_popcount64(state & inst.sign_mask) % 2 != 0) {
                        real_c = -real_c;
                        imag_c = -imag_c;
                    }
                }
                
                KComplex term_val(static_cast<Real>(real_c), static_cast<Real>(imag_c));
                Kokkos::atomic_add(&y(beta), term_val * xin);
            }
        }
    );
}
"""
content = re.sub(r'template <typename ExecSpace>\nvoid MatrixFreeHamiltonian<ExecSpace>::apply\(\n    const VectorView<ExecSpace>& x,\n    VectorView<ExecSpace>& y\n\) const\n\{.*?\n\}\n\ntemplate <typename ExecSpace>\nvoid MatrixFreeHamiltonian<ExecSpace>::apply\(\n    const Complex\* x,', apply_logic + '\ntemplate <typename ExecSpace>\nvoid MatrixFreeHamiltonian<ExecSpace>::apply(\n    const Complex* x,', content, flags=re.DOTALL)

with open("src/hamiltonian/matrix_free_hamiltonian.cpp", "w") as f:
    f.write(content)

print("Updated matrix_free_hamiltonian.cpp")
