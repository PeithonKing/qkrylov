# Theory

This section details the theoretical foundations behind `qkrylov`. These pages provide mathematical background and explain the algorithms used under the hood.

- **[Exact Diagonalization](exact_diag.md)**: Discusses what ED is, the exponential scaling of Hilbert spaces, and its limits.
- **[Krylov Subspace Methods](krylov.md)**: Explains the Lanczos and Davidson algorithms for finding extremal eigenvalues in sparse systems.
- **[Matrix-Free Methods](#)**: Why no matrix is explicitly stored in memory, relying instead on fast action of the Hamiltonian.
- **[Spectral Functions](spectral.md)**: Details the continued fraction expansion approach for dynamical responses.
- **[Finite Temperature Lanczos Method (FTLM)](#)**: Describes how thermal averages are approximated using Krylov subspaces.

*Note: Most of these pages are currently under construction.*
