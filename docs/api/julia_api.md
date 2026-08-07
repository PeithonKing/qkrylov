# Julia API Documentation (`QKrylov.jl`)

`QKrylov.jl` provides native Julia bindings for `qkrylov` via zero-copy C ABI calls (`ccall`).

---

## 1. Module Overview & Package Setup

To use `QKrylov.jl` in your Julia environment:

```julia
using QKrylov
```

### Shared Library Loading
`QKrylov.jl` automatically locates `libqkrylov.so` in your build tree. If custom library placement is used, set the environment variable:
```bash
export QKRYLOV_LIB_PATH=/path/to/libqkrylov.so
```

---

## 2. Symmetry Sectors (`Sector`)

Symmetry sectors restrict the Hilbert space dimension to specific quantum number sectors.

```julia
sec = Sector()
```

### Function Usage

#### `Sector()`
- **Description**: Allocates a new quantum symmetry sector object.
- **Returns**: `Sector` object with automatic GC finalizer.

#### `set_sz!(sec::Sector, sz2::Integer)`
- **Description**: Restricts to a total spin projection $S_z$. Note that `sz2` represents $2 \times S_z$.
- **Arguments**:
  - `sec::Sector`: Sector object to modify.
  - `sz2::Integer`: Twice the $S_z$ quantum number (e.g. `0` for $S_z=0$, `1` for $S_z=1/2$, `-2` for $S_z=-1$).
- **Returns**: `sec::Sector`

#### `set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)`
- **Description**: Restricts particle counts for Fermi-Hubbard and $t$-$J$ models.
- **Arguments**:
  - `sec::Sector`: Sector object to modify.
  - `nup::Integer`: Number of spin-up particles ($N_{\uparrow}$).
  - `ndn::Integer`: Number of spin-down particles ($N_{\downarrow}$).
- **Returns**: `sec::Sector`

---

## 3. Site Definitions (`AbstractSite`)

Site types describe local site degrees of freedom and state spaces.

```julia
s1 = SpinHalfSite()
s2 = FermionSite()
s3 = HubbardSite()
s4 = TJSite()
```

### Types & Constructors

| Type | Parent Type | Description | Local Dimension |
|------|-------------|-------------|----------------|
| `SpinHalfSite()` | `AbstractSite` | Spin-1/2 local site ($|\uparrow\rangle, |\downarrow\rangle$) | 2 |
| `FermionSite()` | `AbstractSite` | Spinless fermion site ($|0\rangle, |1\rangle$) | 2 |
| `HubbardSite()` | `AbstractSite` | Spinful Fermi-Hubbard site ($|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$) | 4 |
| `TJSite()` | `AbstractSite` | $t$-$J$ model site without double occupancy ($|0\rangle, |\uparrow\rangle, |\downarrow\rangle$) | 3 |

---

## 4. Hilbert Space Bases (`AbstractBasis`)

Basis classes construct quantum state representations across $N$ lattice sites.

### Constructors

```julia
b1 = SpinHalfBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
b2 = FermionBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
b3 = HubbardBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
b4 = TJBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing)
```

### Methods

#### `dimension(b::AbstractBasis)::UInt64`
- **Description**: Returns the total dimension of the Hilbert space.
- **Example**:
  ```julia
  dim = dimension(b1)
  ```

#### `nsites(b::AbstractBasis)::Int`
- **Description**: Returns the number of physical lattice sites.
- **Example**:
  ```julia
  sites = nsites(b1)
  ```

#### `Base.size(b::AbstractBasis)`
- **Description**: Overloads Julia `size()` to return matrix dimensions `(dim, dim)`.

---

## 5. Operator Terms (`OpSum`)

`OpSum` constructs Hamiltonian operator expressions from 1-body and 2-body local site operators.

```julia
op = OpSum()
```

### Function Usage

#### `OpSum()`
- **Description**: Creates a new operator sum container.

#### `add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer)`
- **Description**: Adds a 1-body operator term $\text{coeff} \cdot \hat{O}_{1, \text{site1}}$.
- **Arguments**:
  - `op::OpSum`: Target operator sum object.
  - `coeff::Number`: Coupling constant (real or complex).
  - `op1::AbstractString`: Name of local site operator (e.g. `"Sz"`, `"Sp"`, `"Sm"`, `"n"`).
  - `site1::Integer`: 0-indexed site location.

#### `add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer, op2::AbstractString, site2::Integer)`
- **Description**: Adds a 2-body interaction term $\text{coeff} \cdot \hat{O}_{1, \text{site1}} \hat{O}_{2, \text{site2}}$.
- **Arguments**:
  - `op::OpSum`: Target operator sum object.
  - `coeff::Number`: Coupling constant.
  - `op1::AbstractString`, `op2::AbstractString`: Names of local site operators.
  - `site1::Integer`, `site2::Integer`: 0-indexed site locations.

#### `clear!(op::OpSum)`
- **Description**: Clears all terms stored inside `op`.

---

## 6. Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

The `MatrixFreeHamiltonian` evaluates matrix-vector products $y = H \cdot x$ on-the-fly without constructing explicit matrix representations in memory.

### Constructor & Methods

#### `MatrixFreeHamiltonian(basis::AbstractBasis, site::AbstractSite, opsum::OpSum)`
- **Description**: Constructs a matrix-free Hamiltonian wrapper. Automatically retains references to `basis`, `site`, and `opsum` to prevent GC release of dependencies while active.

#### `dimension(H::MatrixFreeHamiltonian)::UInt64`
- **Description**: Returns matrix dimension $\mathcal{D}$.

#### `Base.:*(H::MatrixFreeHamiltonian, x::AbstractVector{<:Number})::Vector{ComplexF64}`
- **Description**: Computes the matrix-vector multiplication $y = H \cdot x$ in zero-copy mode.
- **Example**:
  ```julia
  x = rand(ComplexF64, dimension(H))
  y = H * x
  ```

---

## 7. Solvers (`lanczos_ground_state`)

### Function Usage

#### `lanczos_ground_state(H::MatrixFreeHamiltonian; maxiter::Integer=100, tol::Real=1e-12)::LanczosResult`
- **Description**: Computes the ground state energy using Krylov-subspace Lanczos iteration.
- **Keyword Arguments**:
  - `maxiter::Integer`: Maximum number of Lanczos iterations (default: `100`).
  - `tol::Real`: Residual tolerance (default: `1e-12`).
- **Returns**: `LanczosResult(energy::Float64)`.

---

## 8. Full End-to-End Code Example

```julia
using QKrylov

# Create a 6-site Spin-1/2 Heisenberg chain with Sz=0 sector
N = 6
sec = Sector()
set_sz!(sec, 0)

basis = SpinHalfBasis(N, sec)
site  = SpinHalfSite()
op    = OpSum()

# Add Heisenberg terms H = \sum_i (S^z_i S^z_{i+1} + 0.5(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}))
for i in 0:(N-1)
    next_i = mod(i + 1, N)
    add_term!(op, 1.0, "Sz", i, "Sz", next_i)
    add_term!(op, 0.5, "Sp", i, "Sm", next_i)
    add_term!(op, 0.5, "Sm", i, "Sp", next_i)
end

H = MatrixFreeHamiltonian(basis, site, op)
println("Hamiltonian Dimension: ", dimension(H))

# Solve for ground state energy
res = lanczos_ground_state(H, maxiter=100, tol=1e-12)
println("Calculated Ground State Energy: ", res.energy)
```
