# Davidson Algorithm

!!! note "Under Construction"
    This page will cover the Davidson diagonalization algorithm for computing multiple low-lying eigenvalues.

## Overview

The Davidson algorithm is an iterative diagonalization method that is superior to Lanczos when you need **multiple eigenvalues** simultaneously. It builds a Krylov-like subspace but applies diagonal preconditioning to improve convergence.

## API

=== "Python"
    ```python
    import qkrylov as qk

    result = qk.davidson_lowest(H, n_eig=3, max_subspace=20, tol=1e-8)
    print(result.eigenvalues)   # first 3 eigenvalues
    print(result.eigenvectors)  # corresponding eigenvectors
    ```

=== "C++"
    ```cpp
    auto result = davidson_lowest(H, /*n_eig=*/3, /*max_subspace=*/20, /*tol=*/1e-8);
    for (double e : result.eigenvalues) {
        std::cout << e << "\n";
    }
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](../../contributing.md).

## Coming Soon

- Full parameter documentation
- Convergence analysis
- Comparison with Lanczos for multiple eigenvalues
