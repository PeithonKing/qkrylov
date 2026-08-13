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

# Spinless fermion local degree of freedom (dimension 2)
site_fermion = FermionSite()

# Spinful Fermi-Hubbard electron site (dimension 4)
site_hubbard = HubbardSite()

# t-J model site with constrained double-occupancy (dimension 3)
site_tj = TJSite()
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
println("Full Dimension: ", dimension(basis_full)) # 2^4 = 16

# 2. Construct a 4-site spin-1/2 basis restricted to Sz = 0 sector
sec = Sector()
set_sz!(sec, 0)
basis_sec = SpinHalfBasis(4, sec)
println("Symmetry Sector Dimension: ", dimension(basis_sec)) # 6

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
