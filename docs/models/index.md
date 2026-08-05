# Supported Physical Models

`qkrylov` supports several quantum lattice models out of the box, categorized by their local degrees of freedom and basis states. The Hilbert space for these models scales exponentially with the number of sites, making matrix-free techniques essential.

## [Spin-1/2 Models](spin.md)
**Basis:** `SpinHalfBasis`, **Site:** `SpinHalfSite`
Used for modeling spin systems like the Heisenberg or Ising models. The local Hilbert space dimension is 2. Sectors can be restricted using the $S^z$ quantum number (total magnetization).

## [Fermionic Models](fermion.md)
**Basis:** `FermionBasis`, **Site:** `FermionSite`
Models spinless or identical fermions. The local Hilbert space dimension is 2 (empty or occupied). Sectors can be restricted by the total particle number $N$.

## Full Hubbard Model
**Basis:** `HubbardBasis`, **Site:** `HubbardSite`
Models spin-1/2 fermions with on-site interaction (Hubbard $U$). The local Hilbert space dimension is 4 (empty, up, down, double occupancy). Available quantum number sectors include total spin $S^z$, total number $N$, and specific spin numbers $N_{\uparrow}$ and $N_{\downarrow}$.

## t-J Model
**Basis:** `TJBasis`, **Site:** `TJSite`
Models strongly correlated spin-1/2 fermions where double occupancy is strictly prohibited. The local Hilbert space dimension is 3 (empty, up, down). Like the Hubbard model, sectors include $S^z$, $N$, $N_{\uparrow}$, and $N_{\downarrow}$.
