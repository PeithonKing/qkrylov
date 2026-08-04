# Fermionic Models

Fermionic systems like the tight-binding model or spinless fermions can be built using `FermionBasis` and `FermionSite`.

## Basis and Sectors
The `FermionBasis` handles the Hilbert space mapping. For identical/spinless fermions, sectors are defined by the total particle number.

**Sectors:**
- `use_n`: Boolean flag to enable total number conservation.
- `n`: Total number of particles.

For spin-1/2 models, use `HubbardBasis` where sectors include `use_nup`, `nup`, `use_ndn`, `ndn`, `use_n`, `n`.

## Site Operators
The `FermionSite` provides standard creation/annihilation operators:
- `Cdag`: Creation operator $c^\dagger$
- `C`: Annihilation operator $c$

## Example: Free Fermion Chain (Tight-Binding)
Hamiltonian: $H = -t \sum_{\langle i, j \rangle} (c^\dagger_i c_j + h.c.)$

=== "Python"
    ```python
    from qkrylov import FermionBasis, FermionSite, Sector, OpSum, MatrixFreeHamiltonian

    # 10 sites, half-filling (N=5)
    basis = FermionBasis(10, Sector(use_n=True, n=5))
    site = FermionSite()
    opsum = OpSum()
    
    t = 1.0
    for i in range(10):
        j = (i + 1) % 10
        opsum += (-t, site.Cdag, i, site.C, j)
        opsum += (-t, site.Cdag, j, site.C, i)
        
    H = MatrixFreeHamiltonian(basis, opsum)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/models/fermion.hpp>
    #include <qkrylov/core/opsum.hpp>
    #include <qkrylov/core/hamiltonian.hpp>
    
    using namespace qkrylov;
    
    int main() {
        Sector sector;
        sector.use_n = true;
        sector.n = 5;
        
        FermionBasis basis(10, sector);
        FermionSite site;
        OpSum opsum;
        
        double t = 1.0;
        for(int i = 0; i < 10; ++i) {
            int j = (i + 1) % 10;
            opsum.add({-t, site.Cdag, i, site.C, j});
            opsum.add({-t, site.Cdag, j, site.C, i});
        }
        
        MatrixFreeHamiltonian H(basis, opsum);
        return 0;
    }
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](#).
