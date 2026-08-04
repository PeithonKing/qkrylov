# Lanczos Algorithm

The Lanczos algorithm is a Krylov subspace method designed to find extremal eigenvalues (typically the ground state) of large, sparse Hermitian matrices. By constructing an orthogonal basis for the Krylov subspace iteratively, `qkrylov` avoids building the full dense Hamiltonian matrix in memory.

**Note:** `qkrylov` leverages OpenMP parallelism during matrix-vector multiplications in the Lanczos iterations to drastically reduce computation time.

## Function Signature

=== "Python"
    ```python
    def lanczos_ground_state(H, maxiter=200, tol=1e-12):
        """
        Compute the ground state energy and eigenvector.
        """
        ...
    ```

=== "C++"
    ```cpp
    std::pair<double, std::vector<double>> lanczos_ground_state(const MatrixFreeHamiltonian& H, int maxiter=200, double tol=1e-12);
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](#).

### Parameters
- `H`: A `MatrixFreeHamiltonian` instance.
- `maxiter`: Maximum number of Lanczos iterations allowed.
- `tol`: Convergence tolerance for the eigenvalue residual.

### Returns
- `energy`: The scalar ground state energy.
- `eigenvector`: The corresponding ground state wave function array.

## Example Usage

=== "Python"
    ```python
    from qkrylov import SpinHalfBasis, SpinHalfSite, Sector, OpSum, MatrixFreeHamiltonian
    from qkrylov import lanczos_ground_state
    
    basis = SpinHalfBasis(10, Sector(use_sz=True, sz2=0))
    site = SpinHalfSite()
    opsum = OpSum()
    
    # 1D Heisenberg model setup
    for i in range(10):
        j = (i + 1) % 10
        opsum += (1.0, site.Sz, i, site.Sz, j)
        opsum += (0.5, site.Sp, i, site.Sm, j)
        opsum += (0.5, site.Sm, i, site.Sp, j)
        
    H = MatrixFreeHamiltonian(basis, opsum)
    
    E0, psi0 = lanczos_ground_state(H, maxiter=200, tol=1e-12)
    print(f"Ground State Energy: {E0}")
    ```

=== "C++"
    ```cpp
    #include <iostream>
    #include <qkrylov/models/spin_half.hpp>
    #include <qkrylov/core/hamiltonian.hpp>
    #include <qkrylov/solvers/lanczos.hpp>
    
    using namespace qkrylov;
    
    int main() {
        Sector sector;
        sector.use_sz = true;
        sector.sz2 = 0;
        
        SpinHalfBasis basis(10, sector);
        SpinHalfSite site;
        OpSum opsum;
        
        for(int i = 0; i < 10; ++i) {
            int j = (i + 1) % 10;
            opsum.add({1.0, site.Sz, i, site.Sz, j});
            opsum.add({0.5, site.Sp, i, site.Sm, j});
            opsum.add({0.5, site.Sm, i, site.Sp, j});
        }
        
        MatrixFreeHamiltonian H(basis, opsum);
        
        auto [E0, psi0] = lanczos_ground_state(H, 200, 1e-12);
        std::cout << "Ground State Energy: " << E0 << std::endl;
        
        return 0;
    }
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](#).
