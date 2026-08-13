# QuantumKrylov.jl: Julia Interface for `qkrylov`

`QuantumKrylov.jl` provides idiomatic Julia bindings for [`qkrylov`](../../README.md), a high-performance C++ library for matrix-free Krylov subspace methods in quantum many-body physics.

The Julia interface is constructed directly on top of the binary-stable C ABI exposed by `libqkrylov.so` via Julia's native `ccall` mechanism. It delivers **zero memory overhead**, **automatic garbage collection**, and **idiomatic mathematical syntax** (e.g., `y = H * x`).

---

## Table of Contents
- [Installation & Build](#installation--build)
- [Quickstart Example](#quickstart-example)
- [Complete API Reference](#complete-api-reference)
  - [Symmetry Sectors (`Sector`)](#symmetry-sectors-sector)
  - [Site Models (`AbstractSite`)](#site-models-abstractsite)
  - [Hilbert Space Bases (`AbstractBasis`)](#hilbert-space-bases-abstractbasis)
  - [Operator Terms (`OpSum`)](#operator-terms-opsum)
  - [Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)](#matrix-free-hamiltonian-matrixfreehamiltonian)
  - [Eigensolvers (`lanczos_ground_state`)](#eigensolvers-lanczos_ground_state)
- [Memory Safety & Architecture](#memory-safety--architecture)
- [Running Unit Tests](#running-unit-tests)

---

## Installation

Install the package directly via Julia's Package Manager:

```julia
using Pkg
Pkg.add("QuantumKrylov")
```

No C++ compiler or CMake build is required—`QuantumKrylov.jl` automatically downloads precompiled `qkrylov_jll` binary artifacts for your OS and CPU architecture.

### Development Setup (Local Repository)
If you are developing locally from the source repository:
```bash
julia --project=bindings/julia
```

Inside Julia:
```julia
using QuantumKrylov
```

---

## Quickstart Example

Here is a complete example constructing a 4-site spin-1/2 Heisenberg chain and solving for its ground state energy:

```julia
using QuantumKrylov

# 1. Define Hilbert space basis with Sz = 0 symmetry
sec = Sector()
set_sz!(sec, 0)

basis = SpinHalfBasis(4, sec)
site  = SpinHalfSite()

println("Hilbert space dimension: ", dimension(basis)) # Outputs 6

# 2. Build 1D Heisenberg model Hamiltonian terms: H = \sum_i (S^z_i S^z_{i+1} + 0.5(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}))
op = OpSum()
N = 4
for i in 0:(N - 1)
    next_i = mod(i + 1, N)
    add_term!(op, 1.0, "Sz", i, "Sz", next_i)
    add_term!(op, 0.5, "Sp", i, "Sm", next_i)
    add_term!(op, 0.5, "Sm", i, "Sp", next_i)
end

# 3. Create MatrixFreeHamiltonian
H = MatrixFreeHamiltonian(basis, site, op)

# 4. Perform matrix-vector multiplication (y = H * x)
x = zeros(ComplexF64, dimension(basis))
x[1] = 1.0
y = H * x

# 5. Compute ground state energy via Lanczos solver
res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
println("Ground State Energy: ", res.energy) # Outputs -2.0
```

---

## Complete API Reference

### Symmetry Sectors (`Sector`)

Symmetry sectors allow restricting Hilbert spaces to targeted quantum numbers ($S_z$ projection, particle numbers).

#### `Sector()`
- **Description**: Constructs a new quantum symmetry sector handle.
- **Return**: `Sector` object.

#### `set_sz!(sec::Sector, sz2::Integer)`
- **Description**: Sets the total $S_z$ projection ($2 \times S_z$). For $S_z = 0$, pass `0`. For $S_z = 1/2$, pass `1`.
- **Arguments**:
  - `sec`: Target `Sector`.
  - `sz2`: Twice the $S_z$ quantum number ($2 S_z$).
- **Return**: `sec`

#### `set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)`
- **Description**: Sets electron particle counts for spin-up ($N_{\uparrow}$) and spin-down ($N_{\downarrow}$) sectors in electronic models.
- **Arguments**:
  - `sec`: Target `Sector`.
  - `nup`: Number of spin-up particles.
  - `ndn`: Number of spin-down particles.
- **Return**: `sec`

---

### Site Models (`AbstractSite`)

Site objects define the local degree of freedom and operator matrices.

- **`SpinHalfSite()`**: Creates a spin-1/2 site model (dimension 2: $|\uparrow\rangle, |\downarrow\rangle$).
- **`FermionSite()`**: Creates a spinless fermion site model (dimension 2: $|0\rangle, |1\rangle$).
- **`HubbardSite()`**: Creates a spinful Fermi-Hubbard site model (dimension 4: $|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$).
- **`TJSite()`**: Creates a $t$-$J$ model site model with constrained double-occupancy (dimension 3: $|0\rangle, |\uparrow\rangle, |\downarrow\rangle$).

---

### Hilbert Space Bases (`AbstractBasis`)

Basis objects construct quantum many-body state representations across lattice sites.

#### Constructors
- **`SpinHalfBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)`**
- **`FermionBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)`**
- **`HubbardBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)`**
- **`TJBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)`**

#### Query Methods
- **`dimension(b::AbstractBasis)::UInt64`**: Returns the total dimension of the basis.
- **`nsites(b::AbstractBasis)::Int`**: Returns the number of physical lattice sites.
- **`Base.size(b::AbstractBasis)`**: Returns `(dim, dim)` tuple representing matrix size.
- **`Base.length(b::AbstractBasis)`**: Returns `dim`.

---

### Operator Terms (`OpSum`)

`OpSum` stores operator term expressions used to construct matrix-free Hamiltonians.

#### `OpSum()`
- **Description**: Constructs an empty `OpSum` handle.

#### `add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer)`
- **Description**: Adds a 1-body operator term $\text{coeff} \cdot \hat{O}_{1, \text{site1}}$.
- **Arguments**:
  - `op`: Target `OpSum`.
  - `coeff`: Real or complex coupling coefficient (`Number`).
  - `op1`: Operator string label (e.g. `"Sz"`, `"Sp"`, `"Sm"`, `"n"`).
  - `site1`: 0-indexed site index.

#### `add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer, op2::AbstractString, site2::Integer)`
- **Description**: Adds a 2-body operator term $\text{coeff} \cdot \hat{O}_{1, \text{site1}} \hat{O}_{2, \text{site2}}$.
- **Arguments**:
  - `op`: Target `OpSum`.
  - `coeff`: Real or complex coupling coefficient (`Number`).
  - `op1`, `op2`: Operator string labels.
  - `site1`, `site2`: 0-indexed site indices.

#### `clear!(op::OpSum)`
- **Description**: Clears all operator terms stored in `op`.

---

### Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

#### `MatrixFreeHamiltonian(basis::AbstractBasis, site::AbstractSite, opsum::OpSum)`
- **Description**: Constructs a matrix-free Hamiltonian. Holds reference guards to `basis`, `site`, and `opsum` to ensure underlying C++ objects are not garbage collected while `H` is active.

#### `dimension(H::MatrixFreeHamiltonian)::UInt64`
- **Description**: Returns the dimension of the Hamiltonian matrix.

#### `Base.size(H::MatrixFreeHamiltonian)`
- **Description**: Returns `(dim, dim)`.

#### `Base.:*(H::MatrixFreeHamiltonian, x::AbstractVector{<:Number})::Vector{ComplexF64}`
- **Description**: Computes the matrix-vector product $y = H \cdot x$ using zero-copy memory arrays and `GC.@preserve` pointer protection.
- **Return**: `Vector{ComplexF64}` of length `dimension(H)`.

---

### Eigensolvers (`lanczos_ground_state`)

#### `lanczos_ground_state(H::MatrixFreeHamiltonian; maxiter::Integer=100, tol::Real=1e-12)::LanczosResult`
- **Description**: Computes the ground state energy of `H` using the matrix-free Lanczos algorithm.
- **Keyword Arguments**:
  - `maxiter`: Maximum Lanczos iterations (default: `100`).
  - `tol`: Convergence tolerance for residual norm (default: `1e-12`).
- **Return**: `LanczosResult` struct containing `energy::Float64`.

---

## Memory Safety & Architecture

- **Automatic Garbage Collection**: Every Julia struct (`Sector`, `AbstractSite`, `AbstractBasis`, `OpSum`, `MatrixFreeHamiltonian`) registers a Julia `finalizer` block on creation. When Julia garbage collects an object, the corresponding C ABI destructor function (`qkrylov_*_destroy`) is automatically called.
- **Reference Preservation**: `MatrixFreeHamiltonian` stores fields pointing to its `basis`, `site`, and `opsum`. This guarantees that Julia will never garbage collect dependency handles while the Hamiltonian object exists.
- **Zero-Copy Matrix-Vector Application**: Vector operations use `GC.@preserve` during `ccall` invocations, passing raw pointers to Julia array memory directly into C++ core kernels without copying data.

---

## Running Unit Tests

Run the full package test suite using Julia:
```bash
julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
```
