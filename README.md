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
    - **Lanczos**: Accurate ground-state energy, lowest-$k$ eigenpairs, iterations/convergence tracking, and single-pass or memory-frugal two-pass Ritz vector reconstruction.
    - **Davidson**: Iterative subspace solver with diagonal preconditioning for lowest-$k$ eigenpairs simultaneously.
    - **Dynamics**: Continued Fraction Lanczos and Correction Vector linear solves for dynamical structure factor $S(\omega)$ and Green's function calculations.
    - **Finite Temperature**: Decoupled Finite Temperature Lanczos Method (FTLM) for multi-temperature sweeps of thermodynamic equations of state ($Z, F, E, C_v, S$) and arbitrary physical observables with error bars.
- **Multi-Language Support**: Robust Python interface via `nanobind` with SciPy `LinearOperator` interoperability, native Julia package [`QuantumKrylov.jl`](bindings/julia/README.md) supporting the SciML `solve(prob, alg)` dispatch interface, and dual-precision C ABI ([`include/qkrylov/c_api.h`](include/qkrylov/c_api.h)).

## Build Requirements

- C++20 compatible compiler (e.g., GCC 11+, Clang 13+, MSVC 19.30+)
- CMake 3.20+
- `nanobind` (install via `pip install nanobind`)

## Quick Start

### For Julia Users

#### Option 1: Install the latest prebuilt release (`julia-latest`)
Automatically downloads and configures the latest native prebuilt binaries (`libqkrylov.so`, `libqkrylov.dylib`, or `qkrylov.dll`). On Linux systems with an NVIDIA GPU and driver 12+, it automatically downloads the **CUDA 12 accelerated** binary with zero manual compilation:
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="julia-latest", subdir="bindings/julia")
```

#### Option 2: Pin to a specific historical build
Every CI build is permanently archived with its own tagged release. To install an exact, pinned build, specify its commit SHA (e.g. `rev="1d00f75"`):
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="<commit-sha>", subdir="bindings/julia")
```

#### Option 3: Julia General Registry (once registered)
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

### Installing and linking from C and C++

qkrylov is built with CMake and installs a regular CMake package, so there is nothing special to do:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DQKRYLOV_BUILD_TESTS=OFF
cmake --build build -j
cmake --install build --prefix ~/qkrylov-install
```

Then, in your own project:

```cmake
find_package(qkrylov 0.1 REQUIRED)          # needs CMAKE_PREFIX_PATH=~/qkrylov-install
add_executable(app main.cpp)                # C or C++
target_link_libraries(app PRIVATE qkrylov::qkrylov)
```

Without CMake, use `pkg-config` (shared builds): `cc main.c $(pkg-config --cflags --libs qkrylov)` with
`PKG_CONFIG_PATH=~/qkrylov-install/lib/pkgconfig`.

- **Kokkos.** qkrylov is built on [Kokkos](https://github.com/kokkos/kokkos) 4.x. If an installed Kokkos is found
  (`CMAKE_PREFIX_PATH` / `Kokkos_ROOT`), it is used; otherwise Kokkos 4.7.04 is downloaded and built with the
  project and installed alongside it. Use `-DQKRYLOV_USE_SYSTEM_KOKKOS=OFF` to always fetch. GPU support follows
  the Kokkos you build against (for example `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_...`).
- **C API.** `<qkrylov/c_api.h>` is plain C99; the overview at the top of the header explains ownership, errors,
  precision, start-up and threading. `qkrylov_version()` and `QKRYLOV_VERSION_*` let you check that the header
  and the loaded library match. See [`examples/quickstart.c`](examples/quickstart.c).
- **C++ API.** Pick the precision with the namespace (`qkrylov::fp64`, `qkrylov::fp32`) and the device with the
  Hamiltonian's execution space; see [`examples/quickstart.cpp`](examples/quickstart.cpp).
- **Prebuilt binaries** are provided for Python (wheels) and Julia (artifacts). For C and C++ please build from
  source as above.

### C++ Example

```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;                       // double precision; qkrylov::fp32 for single

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const int N = 4;
        Sector sector;
        sector.sz = {0.0};                           // total Sz = 0
        auto basis = std::make_shared<SpinHalfBasis>(N, sector);
        auto site  = std::make_shared<SpinHalfSite>();

        OpSum ops;                                   // open Heisenberg chain
        for (int i = 0; i < N - 1; ++i) {
            ops.add_term(1.0 * Sz(i) * Sz(i + 1));
            ops.add_term(0.5 * Sp(i) * Sm(i + 1));
            ops.add_term(0.5 * Sm(i) * Sp(i + 1));
        }

        MatrixFreeHamiltonian<Kokkos::DefaultHostExecutionSpace> H(basis, site, ops);   // Kokkos::Cuda for a GPU
        LanczosConfig cfg;
        auto res = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
        std::cout << "Ground state energy: " << res.energy << std::endl;
    }
    Kokkos::finalize();
}
```

The same problem as a longer, periodic-chain example in all four languages: [`examples/`](examples/).

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

# 4-site 1D Heisenberg chain
N = 4
basis = SpinHalfBasis(N)

op = OpSum()
for i in 0:(N - 2)
    # Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    # Note: `global` is needed when running as a top-level script, but can be
    # omitted if this loop is inside a function or run directly in the REPL.
    global op += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
end

# Construct MatrixFreeHamiltonian (site model automatically inferred from basis)
# Targets GPU if available ("cuda:0"), otherwise falls back to CPU
target_device = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=target_device)

# Compute ground state energy and wavefunction
res = lanczos_ground_state(H, return_state=true)

println("Execution device:    ", target_device)
println("Ground state energy: ", res.energy)
println("Iterations executed: ", res.iterations)
println("Convergence status:  ", res.converged)
```

## Things To Be Done (Roadmap)

- **Distributed Multi-GPU**: Multi-node MPI + CUDA/HIP Kokkos execution space scaling for very large Hilbert spaces.
- **HDF5 Integration**: Efficient storage of large eigenvectors and Krylov subspace results.

## Documentation

- **C API Reference (`extern "C"`)**: [`include/qkrylov/c_api.h`](include/qkrylov/c_api.h) (overview and per-function notes in the header)
- **Julia Core Concepts Guide**: [`bindings/julia/documentation.md`](bindings/julia/documentation.md)
- **Julia Package README**: [`bindings/julia/README.md`](bindings/julia/README.md)

