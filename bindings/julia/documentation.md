# QuantumKrylov.jl Documentation: Core Concepts

`QuantumKrylov.jl` provides three fundamental building blocks for setting up quantum many-body systems: **Sectors**, **Sites**, and **Bases**.

---

## 1. Symmetry Sectors (`Sector`)

### What is a Sector?
A `Sector` represents quantum conservation laws and symmetry restrictions (such as total spin $S_z$, spin-up/spin-down particle numbers $N_\uparrow, N_\downarrow$, total fermion count $N$, or boson count $N_b$).

Using a symmetry sector restricts the Hilbert space to states matching specific quantum numbers, significantly reducing memory consumption and matrix dimensions.

### How to use `Sector`

```julia
using QuantumKrylov

# Create an empty sector object (no constraints enabled initially)
sec = Sector()
println(sec) # Outputs: Sector(unconstrained)

# Constrain total Sz projection (2 * Sz)
set_sz!(sec, 0) # Sz = 0
println(sec) # Outputs: Sector(2*Sz = 0)

# Check active constraints
sz_val = get_sz(sec) # Returns 0 (Int)
n_val  = get_n(sec)  # Returns nothing (unconstrained)

# Constrain electron numbers for Fermi-Hubbard models
set_hubbard_particles!(sec, 1, 1) # N_up = 1, N_down = 1
hp_val = get_hubbard_particles(sec) # Returns (1, 1)

# Constrain spinless fermion particle count
set_n!(sec, 2) # N = 2 particles
get_n(sec)     # Returns 2

# Constrain boson particle count
set_nb!(sec, 1) # Nb = 1 boson
get_nb(sec)     # Returns 1
```

### Functions & Default Values

| Function | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `Sector()` | None | Unconstrained sector handle | Constructs a new symmetry sector with no quantum number constraints active. |
| `set_sz!(sec, sz2)` | `sec::Sector`<br>`sz2::Integer` | None (required) | Restricts basis to states with total $S_z = \text{sz2} / 2$. Returns `sec`. |
| `set_hubbard_particles!(sec, nup, ndn)` | `sec::Sector`<br>`nup::Integer`<br>`ndn::Integer` | None (required) | Restricts basis to states with $N_\uparrow = \text{nup}$ and $N_\downarrow = \text{ndn}$. Returns `sec`. |
| `set_n!(sec, n)` | `sec::Sector`<br>`n::Integer` | None (required) | Restricts basis to states with $N = \text{n}$ total spinless fermions. Returns `sec`. |
| `set_nb!(sec, nb)` | `sec::Sector`<br>`nb::Integer` | None (required) | Restricts basis to states with $N_b = \text{nb}$ total bosons. Returns `sec`. |
| `get_sz(sec)` | `sec::Sector` | None | Returns active $2 S_z$ value (`Int`), or `nothing` if unconstrained. |
| `get_hubbard_particles(sec)` | `sec::Sector` | None | Returns `(nup, ndn)` tuple (`Tuple{Int, Int}`), or `nothing` if unconstrained. |
| `get_n(sec)` | `sec::Sector` | None | Returns active $N$ particle count (`Int`), or `nothing` if unconstrained. |
| `get_nb(sec)` | `sec::Sector` | None | Returns active $N_b$ boson count (`Int`), or `nothing` if unconstrained. |

---

## 2. Site Models (`AbstractSite`)

### What is a Site?
A `Site` object defines the physical degree of freedom at each lattice site, specifying the local Hilbert space dimension and local quantum operators (such as $S^z, S^+, S^-, c, c^\dagger, n$).

### How to use `Site`

```julia
# Spin-1/2 local degree of freedom (dimension 2)
site_spin = SpinHalfSite()
println(site_spin) # Outputs: SpinHalfSite(dim = 2, states = [↑, ↓])

# Spinless fermion local degree of freedom (dimension 2)
site_fermion = FermionSite()
println(site_fermion) # Outputs: FermionSite(dim = 2, states = [0, 1])

# Spinful Fermi-Hubbard electron site (dimension 4)
site_hubbard = HubbardSite()
println(site_hubbard) # Outputs: HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])

# t-J model site with constrained double-occupancy (dimension 3)
site_tj = TJSite()
println(site_tj) # Outputs: TJSite(dim = 3, states = [0, ↑, ↓])
```

### Available Site Types & Default Values

| Site Type | Local Dimension | Local Physical Basis States | Arguments & Default Values |
| :--- | :--- | :--- | :--- |
| `SpinHalfSite()` | 2 | $|\uparrow\rangle, |\downarrow\rangle$ | No arguments required. |
| `FermionSite()` | 2 | $|0\rangle, |1\rangle$ (empty, occupied) | No arguments required. |
| `HubbardSite()` | 4 | $|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$ | No arguments required. |
| `TJSite()` | 3 | $|0\rangle, |\uparrow\rangle, |\downarrow\rangle$ (no double occupancy) | No arguments required. |

---

## 3. Hilbert Space Bases (`AbstractBasis`)

### What is a Basis?
A `Basis` represents the complete many-body Hilbert space constructed across $N$ physical lattice sites. It can either represent the full unconstrained Hilbert space or be restricted to a specific `Sector`.

### How to use `Basis`

```julia
# 1. Construct a full (unconstrained) 4-site spin-1/2 basis
basis_full = SpinHalfBasis(4)
println(basis_full) # Outputs: SpinHalfBasis(sites = 4, dim = 16)

# 2. Construct a 4-site spin-1/2 basis restricted to Sz = 0 sector
sec = Sector()
set_sz!(sec, 0)
basis_sec = SpinHalfBasis(4, sec)
println(basis_sec) # Outputs: SpinHalfBasis(sites = 4, dim = 6, sector = Sector(2*Sz = 0))

# 3. Query properties
N = nsites(basis_sec) # 4
dim = dimension(basis_sec) # 6

# 4. Inspect basis state bitstrings
st0 = state(basis_sec, 0) # Bitstring of 0-th state (0-indexed)
st_first = basis_sec[1]   # Bitstring of 1st state (1-indexed Julia syntax)

# 5. Look up index from bitstring
idx = basis_index(basis_sec, st0) # Returns 0

# 6. Check if state bitstring belongs to basis
is_present = st0 in basis_sec # Returns true
```

### Basis Types, Parameters & Default Values

| Basis Constructor | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `SpinHalfBasis(num_sites, sector)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}` | `num_sites`: Required<br>`sector`: `nothing` | Spin-1/2 basis on `num_sites` sites. Default `sector=nothing` creates full $2^N$ basis. |
| `FermionBasis(num_sites, sector)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}` | `num_sites`: Required<br>`sector`: `nothing` | Spinless fermion basis on `num_sites` sites. Default `sector=nothing` creates full $2^N$ basis. |
| `HubbardBasis(num_sites, sector)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}` | `num_sites`: Required<br>`sector`: `nothing` | Fermi-Hubbard basis on `num_sites` sites. Default `sector=nothing` creates full $4^N$ basis. |
| `TJBasis(num_sites, sector)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}` | `num_sites`: Required<br>`sector`: `nothing` | $t$-$J$ model basis on `num_sites` sites. Default `sector=nothing` creates full $3^N$ basis. |

### Basis Operations & Query Functions

| Function / Syntax | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `dimension(b)` | `b::AbstractBasis` | `UInt64` | Returns total dimension of the basis space. |
| `nsites(b)` | `b::AbstractBasis` | `Int` | Returns number of physical lattice sites. |
| `state(b, index)` | `b::AbstractBasis`, `index::Integer` | `UInt64` | Returns 64-bit integer bitstring of state at 0-based `index`. |
| `b[i]` | `b::AbstractBasis`, `i::Integer` | `UInt64` | Returns 64-bit integer bitstring of state at 1-based index `i`. |
| `basis_index(b, bitstring)`| `b::AbstractBasis`, `bitstring::Unsigned` | `Int64` | Returns 0-based index of `bitstring` in basis, or `-1` if not present. |
| `bitstring in b` | `bitstring::Unsigned`, `b::AbstractBasis` | `Bool` | Returns `true` if `bitstring` state belongs to basis `b`. |
| `size(b)` | `b::AbstractBasis` | `(Int, Int)` | Returns `(dim, dim)` tuple. |
| `length(b)` | `b::AbstractBasis` | `Int` | Returns total dimension `dim`. |

---

## 4. Operator Terms (`OpSum`) & Matrix-Free Hamiltonians (`MatrixFreeHamiltonian`)

### 4.1 Operator Terms (`OpSum`)

#### What is an `OpSum`?
`OpSum` stores operator term expressions (such as $\hat{S}_i^z \hat{S}_j^z$, $\hat{S}_i^+ \hat{S}_j^-$, or $c_i^\dagger c_j$) used to construct a Hamiltonian or observable operator. Terms are built using local operator string names (e.g. `"Sz"`, `"Sp"`, `"Sm"`, `"n"`, `"c"`, `"cdag"`) and 0-indexed site numbers.

#### How to use `OpSum`

```julia
# 1. Create an empty OpSum
op = OpSum()

# 2. Operator Arithmetic Syntax (Recommended)
# Build 1D Heisenberg model terms directly using operator generators:
for i in 0:3
    next_i = mod(i + 1, 4)
    op += 1.0 * Sz(i) * Sz(next_i) + 0.5 * (Sp(i) * Sm(next_i) + Sm(i) * Sp(next_i))
end

# 3. Add a 3-body (N-body) term using operator generators
op += 0.25 * Sz(0) * Sz(1) * Sz(2)

# 4. Inspect OpSum
println(op)
# Outputs: OpSum(Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1) + 0.5 * Sm(0) * Sp(1) + ...)

num_terms = length(op) # Returns 5
is_empty  = isempty(op) # Returns false

# 5. Validate site index bounds for a 4-site system
valid, errors = validate(op, 4) # Returns (true, String[])
validate!(op, 4)                # Throws ArgumentError if any site >= 4 or < 0

# 6. Alternatively, use string-based add_term! functions:
add_term!(op, 1.0, "Sz", 0, "Sz", 1)

# 7. Clear all terms
clear!(op)
```

#### Operator Generators & Arithmetic Overloads

| Generator / Syntax | Arguments | Description |
| :--- | :--- | :--- |
| `Sz(site)`, `Sp(site)`, `Sm(site)` | `site::Integer` | Spin-1/2 operators ($S^z, S^+, S^-$) at 0-indexed `site`. |
| `Sx(site)`, `Sy(site)` | `site::Integer` | Spin-1/2 operators ($S^x, S^y$) at 0-indexed `site`. |
| `n(site)` | `site::Integer` | Particle number operator ($n_i$) at 0-indexed `site`. |
| `c(site)`, `cdag(site)` | `site::Integer` | Fermionic annihilation ($c_i$) and creation ($c_i^\dagger$) operators at 0-indexed `site`. |
| `coeff * term * ...` | `coeff::Number`, `term::OpTerm` | Multiplies operator factors and scales coupling coefficient. |
| `term1 + term2` | `term1`, `term2` | Combines operator terms into an `OpExpr` term collection. |
| `op += expr` | `op::OpSum`, `expr::OpExpr` | Appends operator terms into `OpSum`. |

#### `OpSum` Functions & Default Values

| Function | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `OpSum()` | None | Empty `OpSum` handle | Constructs a new empty operator sum container. |
| `length(op)` | `op::OpSum` | None | Returns the total number of terms stored in `op`. |
| `isempty(op)` | `op::OpSum` | None | Returns `true` if `op` has zero terms. |
| `validate(op, num_sites)` | `op::OpSum`<br>`num_sites::Integer` | None (required) | Validates term site indices against `0 <= site < num_sites`. Returns `(valid::Bool, errors::Vector{String})`. |
| `validate!(op, num_sites)` | `op::OpSum`<br>`num_sites::Integer` | None (required) | Validates term site indices. Throws `ArgumentError` if any site index is out of bounds or coefficient is non-finite. |
| `add_term!(op, coeff, op1, site1)` | `op::OpSum`<br>`coeff::Number`<br>`op1::AbstractString`<br>`site1::Integer` | None (required) | Adds 1-body term $\text{coeff} \cdot \hat{O}_{1, \text{site1}}$. Returns `op`. |
| `add_term!(op, coeff, op1, site1, op2, site2)` | `op::OpSum`<br>`coeff::Number`<br>`op1::AbstractString`<br>`site1::Integer`<br>`op2::AbstractString`<br>`site2::Integer` | None (required) | Adds 2-body term $\text{coeff} \cdot \hat{O}_{1, \text{site1}} \hat{O}_{2, \text{site2}}$. Returns `op`. |
| `add_term!(op, coeff, ops, sites)` | `op::OpSum`<br>`coeff::Number`<br>`ops::Vector{<:AbstractString}`<br>`sites::Vector{<:Integer}` | None (required) | Adds general $N$-body term $\text{coeff} \cdot \prod_{k=1}^N \hat{O}_{k, \text{sites}[k]}$. Returns `op`. |
| `clear!(op)` | `op::OpSum` | None | Clears all added operator terms from `op`. Returns `op`. |

---

### 4.2 Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

#### What is a `MatrixFreeHamiltonian`?
A `MatrixFreeHamiltonian` evaluates matrix-vector products $y = H \cdot x$ on-the-fly without ever storing the full $N \times N$ matrix in memory. It combines a `Basis`, a `Site` model, and an `OpSum`.

#### How to use `MatrixFreeHamiltonian`

```julia
# 1. Setup basis and opsum
basis = SpinHalfBasis(4)
op    = OpSum()
add_term!(op, 1.0, "Sz", 0, "Sz", 1)

# 2. Construct MatrixFreeHamiltonian (site is automatically inferred from basis)
H = MatrixFreeHamiltonian(basis, op)
println(H) # Outputs: MatrixFreeHamiltonian(dim = 16, basis = SpinHalfBasis(sites = 4, dim = 16))

# Alternatively, explicitly specify the site model:
site = SpinHalfSite()
H_explicit = MatrixFreeHamiltonian(basis, site, op)

# 3. Perform matrix-vector multiplication (y = H * x)
x = zeros(ComplexF64, dimension(H))
x[1] = 1.0 + 0.0im
y = H * x # Vector{ComplexF64} of length dimension(H)

# 4. Extract diagonal elements H_ii without matrix allocation
diag_H = diagonal(H) # Vector{Float64} of length dimension(H)

# 5. Query dimensions
dim = dimension(H) # 16
sz  = size(H)      # (16, 16)
```

#### `MatrixFreeHamiltonian` Functions & Operations

| Function / Syntax | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `MatrixFreeHamiltonian(basis, opsum)` | `basis::AbstractBasis`<br>`opsum::OpSum` | `MatrixFreeHamiltonian` | **Convenience Constructor**. Automatically infers the matching default `Site` model (`SpinHalfSite`, `FermionSite`, `HubbardSite`, or `TJSite`) from the basis type. |
| `MatrixFreeHamiltonian(basis, site, opsum)` | `basis::AbstractBasis`<br>`site::AbstractSite`<br>`opsum::OpSum` | `MatrixFreeHamiltonian` | **Explicit Constructor**. Constructs a matrix-free Hamiltonian operator with a specified site model. |
| `H * x` | `H::MatrixFreeHamiltonian`<br>`x::AbstractVector{<:Number}` | `Vector{ComplexF64}` | Performs zero-copy matrix-vector multiplication $y = H \cdot x$. Length of `x` must equal `dimension(H)`. |
| `diagonal(H)` | `H::MatrixFreeHamiltonian` | `Vector{Float64}` | Computes and returns matrix-free diagonal elements $H_{ii}$. |
| `dimension(H)` | `H::MatrixFreeHamiltonian` | `UInt64` | Returns total matrix dimension of `H`. |
| `size(H)` | `H::MatrixFreeHamiltonian` | `(Int, Int)` | Returns `(dim, dim)` matrix shape tuple. |
