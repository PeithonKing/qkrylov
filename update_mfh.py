import re

with open("include/qkrylov/hamiltonian/matrix_free_hamiltonian.hpp", "r") as f:
    content = f.read()

# Add instruction include
content = content.replace('#include "qkrylov/operators/opsum.hpp"', '#include "qkrylov/operators/opsum.hpp"\n#include "qkrylov/core/instruction.hpp"')

# Add instruction view members
add_members = """
    Device device_;
    Index dim_ = 0;
    
    Kokkos::View<qkrylov::Instruction*, ExecSpace> instructions_;
    int num_instructions_ = 0;
"""
content = re.sub(r'Device device_;\s+Index dim_ = 0;', add_members, content)

with open("include/qkrylov/hamiltonian/matrix_free_hamiltonian.hpp", "w") as f:
    f.write(content)

print("Updated matrix_free_hamiltonian.hpp")
