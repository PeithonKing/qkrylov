import re

with open("src/hamiltonian/matrix_free_hamiltonian.cpp", "r") as f:
    content = f.read()

new_apply = """
    Kokkos::parallel_for("qkrylov::H_apply",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
        KOKKOS_LAMBDA(const Index beta) {
            const StateID state_beta = basis_ptr->state(beta);
            
            // 1. Diagonal contribution
            KComplex y_val = diag_view(beta) * x(beta);

            // 2. Off-diagonal VM execution (pull-based, no atomics)
            for (int i = 0; i < num_inst; ++i) {
                const auto& inst = inst_view(i);
                
                StateID state_alpha = state_beta ^ inst.flip_mask;
                
                if ((state_alpha & inst.check_mask) != inst.expected_bits) continue;
                if (!basis_ptr->contains(state_alpha)) continue;
                
                Index alpha = basis_ptr->index(state_alpha);
                
                double real_c = inst.coeff.real();
                double imag_c = inst.coeff.imag();
                
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
"""

content = re.sub(r'Kokkos::parallel_for\("qkrylov::H_apply".*?\);\n\}', new_apply, content, flags=re.DOTALL)

with open("src/hamiltonian/matrix_free_hamiltonian.cpp", "w") as f:
    f.write(content)
print("Optimized VM")
