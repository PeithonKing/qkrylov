# qkrylov

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics.

## Overview

`qkrylov` provides a high-performance core for performing exact diagonalization and Krylov-based calculations (like Lanczos and Davidson) without explicitly constructing Hamiltonian matrices. By implementing the matrix-free action $y = Hx$, the library enables the study of much larger Hilbert spaces than traditional matrix-based methods.

## Features Completed

- **C++20 Core**: Leveraging modern C++ features for performance and safety.
- **Basis Abstraction**: Generic basis management with support for symmetry sectors.
- **Supported Models**:
    - **Spin-Half Systems**: Heisenberg, transverse-field Ising, etc.
    - **Fermionic Systems**: Spinless fermions with Jordan-Wigner phases.
    - **Hubbard Models**: Interacting electrons with spin conservation.
    - **t-J Models**: Doped antiferromagnets with no-double-occupancy constraint.
- **Matrix-Free Hamiltonian**: Efficient application of operator sums (`OpSum`) to state vectors.
- **Advanced Solvers**:
    - **Lanczos**: Accurate ground-state energy and Ritz vector calculation.
    - **Davidson**: Iterative solver for the lowest few eigenpairs.
    - **Dynamics**: Continued Fraction Lanczos for dynamical structure factor $S(\omega)$ calculations.
- **Multi-Language Support**: Robust Python interface via `nanobind` and native Julia package [`QuantumKrylov.jl`](bindings/julia/README.md) backed by prebuilt `qkrylov_jll` binary artifacts.

## Build Requirements

- C++20 compatible compiler (e.g., GCC 11+, Clang 13+, MSVC 19.30+)
- CMake 3.20+
- `nanobind` (install via `pip install nanobind`)

## Quick Start

### For Julia Users (No C++ Compiler Required)
```julia
using Pkg
Pkg.add("QuantumKrylov")
```

### For Python Users
```bash
pip install qkrylov
```

### For C++ Developers & Local Building
If you are developing or modifying the C++ core engine:

```bash
make build
make test
pytest bindings/python/tests/test_basic.py
julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
```

### C++ Example

```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;

int main() {
    int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();

    OpSum os;
    for (int i = 0; i < N - 1; ++i) {
        // Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
        os += {1.0, {{"Sz", i}, {"Sz", i+1}}};
        os += {0.5, {{"Sp", i}, {"Sm", i+1}}};
        os += {0.5, {{"Sm", i}, {"Sp", i+1}}};
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto result = lanczos_ground_state(H);

    std::cout << "Ground state energy: " << result.energy << std::endl;
    return 0;
}
```

### Python Example

```python
import qkrylov

# 4-site Heisenberg chain
N = 4
basis = qkrylov.SpinHalfBasis(N)
site = qkrylov.SpinHalfSite()

os = qkrylov.OpSum()
for i in range(N - 1):
    # Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    os += 1.0, "Sz", i, "Sz", i+1
    os += 0.5, "Sp", i, "Sm", i+1
    os += 0.5, "Sm", i, "Sp", i+1

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
result = qkrylov.lanczos_ground_state(H)

print(f"Ground state energy: {result.energy}")
```

### Julia Example

```julia
using QuantumKrylov

# 4-site Heisenberg chain
N = 4
basis = SpinHalfBasis(N)
site = SpinHalfSite()

op = OpSum()
for i in 0:(N - 2)
    # Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    add_term!(op, 1.0, "Sz", i, "Sz", i + 1)
    add_term!(op, 0.5, "Sp", i, "Sm", i + 1)
    add_term!(op, 0.5, "Sm", i, "Sp", i + 1)
end

H = MatrixFreeHamiltonian(basis, site, op)
result = lanczos_ground_state(H)

println("Ground state energy: ", result.energy)
```

## Things To Be Done (Roadmap)

- **GPU Acceleration**: CUDA/HIP support for Hamiltonian application.
- **HDF5 Integration**: Efficient storage of large eigenvectors and results.
- **Finite Temperature**: Finite Temperature Lanczos Method (FTLM).

## Documentation

Full documentation is available in the `docs/` directory. See `docs/source/tutorial.md` for a comprehensive guide modeled after iTensor and `docs/api/julia_api.md` for the Julia API reference.

