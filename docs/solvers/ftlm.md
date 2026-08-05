# Finite Temperature Lanczos Method (FTLM)

!!! note "Under Construction"
    This page will cover the FTLM solver for computing thermodynamic properties.

## Overview

The Finite Temperature Lanczos Method (FTLM) computes thermodynamic observables such as partition function, internal energy, and specific heat **without explicit full diagonalization**. It uses stochastic sampling over random initial vectors combined with Lanczos iteration.

## API

=== "Python"
    ```python
    import qkrylov as qk

    result = qk.ftlm(H, beta=10.0, n_random=50, n_steps=100)
    print(f"Z = {result.partition_function:.4f}")
    print(f"E = {result.internal_energy:.4f}")
    print(f"Cv = {result.specific_heat:.4f}")
    ```

=== "C++"
    ```cpp
    auto result = ftlm(H, /*beta=*/10.0, /*n_random=*/50, /*n_steps=*/100);
    std::cout << "Z = " << result.partition_function << "\n";
    std::cout << "E = " << result.internal_energy << "\n";
    ```

=== "Julia"
    !!! note "Coming Soon"
        Julia bindings are planned via `extern "C"` FFI. See the [roadmap](../../contributing.md).

## Coming Soon

- Parameter guide: choosing `n_random` and `n_steps`
- Error estimation and convergence
- Temperature sweeps and thermodynamic curves
