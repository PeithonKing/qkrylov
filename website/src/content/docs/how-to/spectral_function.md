---
title: "Compute a Spectral Function"
description: "How to compute the dynamical spectral function S(ω) of a quantum lattice model using qkrylov's continued-fraction Lanczos solver."
---

:::note[Work in progress]
This page is being written by Ananya.
:::

## What you'll learn

- How to obtain the ground state vector using Lanczos.
- How to apply a perturbation operator to create an excited state.
- How to run the `ContinuedFraction` solver to get Lanczos coefficients.
- How to evaluate S(ω) on a frequency grid.

## Prerequisites

You should be comfortable with:
- [Building a Hamiltonian](/qkrylov/getting_started/tutorial_hamiltonian/)
- [Quickstart](/qkrylov/quickstart/)

<!--
ANANYA: Write the full How-To here.
Structure suggestion:
  1. 2-sentence physics motivation: what is a spectral function and why do we want it
  2. Step 1: Solve the ground state (Lanczos/LanczosTwoPass)
  3. Step 2: Apply perturbation operator — e.g., S^+_i |psi0>
  4. Step 3: Run ContinuedFraction(n_iter=200).solve(H, phi0=...)
  5. Step 4: Evaluate S(omega) on a frequency grid with a chosen broadening eta
  6. Step 5: Plot (suggest matplotlib / Plots.jl / gnuplot)
  7. Tabbed code blocks: Python | Julia | C++
  8. Common pitfall: too small n_iter → artificial termination artifacts
  9. See Also: theory/spectral, API: solvers.ContinuedFraction, API: solvers.spectral_function
-->
