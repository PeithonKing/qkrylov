# Solvers

`qkrylov` includes several highly optimized solvers tailored for matrix-free quantum systems. 

- **[Lanczos Algorithm](lanczos.md)**: Best for finding the exact ground state and a few of the lowest excited states of large, sparse Hermitian matrices.
- **[Davidson Algorithm](#)**: Ideal when you need multiple eigenvalues/eigenvectors simultaneously, offering better convergence properties for clustered spectra.
- **[Spectral Functions & Dynamics](dynamics.md)**: Tools to compute dynamical correlation functions, density of states (DOS), and spectral functions using continued fraction expansions.
- **[Finite Temperature Lanczos Method (FTLM)](#)**: Evaluates thermal expectation values and finite-temperature dynamics without fully diagonalizing the Hamiltonian.
