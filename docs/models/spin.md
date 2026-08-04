# Spin-1/2 Models

The spin-1/2 framework in `qkrylov` is built around the `SpinHalfBasis` and `SpinHalfSite` classes.

## Basis and Sectors
The `SpinHalfBasis` class manages the many-body Hilbert space. 

**Constructor Arguments:**
- `N`: Number of sites.
- `sector`: An optional `Sector` object.

The `Sector` object for spin-1/2 systems supports the following fields:
- `use_sz`: Boolean flag to enable total $S^z$ conservation.
- `sz2`: Integer value representing $2S^z$ (twice the total magnetization, to ensure integer values).

### Sector Table
Here is a quick reference for `sz2` values for different numbers of spin-up and spin-down particles:

| $N_{\uparrow}$ | $N_{\downarrow}$ | Total $N$ | `sz2` ($2S^z$) |
|----------------|------------------|-----------|----------------|
| 3              | 0                | 3         | +3             |
| 2              | 1                | 3         | +1             |
| 1              | 2                | 3         | -1             |
| 0              | 3                | 3         | -3             |

## Site Operators
The `SpinHalfSite` provides local operators:
- `Sz`: $S^z$ operator.
- `Sp`: $S^+$ raising operator.
- `Sm`: $S^-$ lowering operator.

## Examples

### Heisenberg Chain
The canonical anti-ferromagnetic Heisenberg spin chain: $H = J \sum_{\langle i, j \rangle} \vec{S}_i \cdot \vec{S}_j$.

=== "Python"
    ```python
    from qkrylov import SpinHalfBasis, SpinHalfSite, Sector, OpSum, MatrixFreeHamiltonian

    # 10 sites, sz=0 sector
    basis = SpinHalfBasis(10, Sector(use_sz=True, sz2=0))
    site = SpinHalfSite()
    opsum = OpSum()
    
    J = 1.0
    for i in range(10):
        j = (i + 1) % 10
        # S^z_i S^z_j
        opsum += (J, site.Sz, i, site.Sz, j)
        # 0.5 * (S^+_i S^-_j + S^-_i S^+_j)
        opsum += (0.5 * J, site.Sp, i, site.Sm, j)
        opsum += (0.5 * J, site.Sm, i, site.Sp, j)
        
    H = MatrixFreeHamiltonian(basis, opsum)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/models/spin_half.hpp>
    #include <qkrylov/core/opsum.hpp>
    #include <qkrylov/core/hamiltonian.hpp>
    
    using namespace qkrylov;
    
    int main() {
        Sector sector;
        sector.use_sz = true;
        sector.sz2 = 0;
        
        SpinHalfBasis basis(10, sector);
        SpinHalfSite site;
        OpSum opsum;
        
        double J = 1.0;
        for(int i = 0; i < 10; ++i) {
            int j = (i + 1) % 10;
            opsum.add({J, site.Sz, i, site.Sz, j});
            opsum.add({0.5 * J, site.Sp, i, site.Sm, j});
            opsum.add({0.5 * J, site.Sm, i, site.Sp, j});
        }
        
        MatrixFreeHamiltonian H(basis, opsum);
        return 0;
    }
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](#).

### Transverse Field Ising Model
Model: $H = -J \sum_{\langle i, j \rangle} S^z_i S^z_j - h \sum_i S^x_i$.
Note: $S^x = \frac{1}{2}(S^+ + S^-)$.

=== "Python"
    ```python
    from qkrylov import SpinHalfBasis, SpinHalfSite, OpSum, MatrixFreeHamiltonian

    basis = SpinHalfBasis(10) # No sz conservation due to Sx
    site = SpinHalfSite()
    opsum = OpSum()
    
    J = 1.0
    h = 0.5
    for i in range(10):
        j = (i + 1) % 10
        opsum += (-J, site.Sz, i, site.Sz, j)
        opsum += (-h * 0.5, site.Sp, i)
        opsum += (-h * 0.5, site.Sm, i)
        
    H = MatrixFreeHamiltonian(basis, opsum)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/models/spin_half.hpp>
    #include <qkrylov/core/opsum.hpp>
    #include <qkrylov/core/hamiltonian.hpp>
    
    using namespace qkrylov;
    
    int main() {
        SpinHalfBasis basis(10);
        SpinHalfSite site;
        OpSum opsum;
        
        double J = 1.0, h = 0.5;
        for(int i = 0; i < 10; ++i) {
            int j = (i + 1) % 10;
            opsum.add({-J, site.Sz, i, site.Sz, j});
            opsum.add({-h * 0.5, site.Sp, i});
            opsum.add({-h * 0.5, site.Sm, i});
        }
        
        MatrixFreeHamiltonian H(basis, opsum);
        return 0;
    }
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](#).
