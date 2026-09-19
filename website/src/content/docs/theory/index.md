---
title: "Theoretical Foundations"
description: "Theoretical physics and numerical analysis foundations underpinning qkrylov"
---

This section details the theoretical physics and numerical analysis foundations underpinning `qkrylov`.

---

- **[Exact Diagonalization](/qkrylov/theory/exact_diag/)**: Exponential scaling of Hilbert spaces, quantum symmetry reduction, Hamiltonian sparsity, and the matrix-free paradigm.
- **[Krylov Subspace Methods](/qkrylov/theory/krylov/)**: Mathematical derivation of the Lanczos three-term recurrence, Rayleigh-Ritz projections, Kaniel-Paige-Saad convergence bounds, and finite-precision reorthogonalization strategies.
- **[Matrix-Free SpMV on GPUs](/qkrylov/theory/matrix_free_spmv/)**: Concrete comparison of dense, CSR sparse, and matrix-free memory consumption across lattice sizes $N=12 \dots 32$, two-pass algorithms, and multi-threaded scaling.
- **[Spectral Functions & Dynamics](/qkrylov/theory/spectral/)**: Zero-temperature Green's functions, continued fraction expansion, and the correction vector shifted-linear solve alternative.
- **[Single-Pass vs Two-Pass Lanczos](/qkrylov/theory/lanczos_memory/)**: High-temperature random trace estimation, decoupled multi-temperature sweeps, and arbitrary observable projections with error bars.
- **[Precision Stability](/qkrylov/theory/precision_stability/)**: Dual-precision (FP32/FP64) stability, ghost states, and loss of orthogonality in Lanczos.
