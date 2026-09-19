---
title: "Finite Temperature with FTLM"
description: "How to compute finite-temperature thermodynamic properties (energy, specific heat, susceptibility) using the Finite-Temperature Lanczos Method."
---

:::note[Work in progress]
This page is being written by Ananya.
:::

## What you'll learn

- What the FTLM is and when to use it vs. exact diagonalization.
- How to compute thermodynamic quantities at a single temperature.
- How to efficiently sweep a temperature grid using `FTLMSamples`.

## Prerequisites

- [Building a Hamiltonian](/qkrylov/getting_started/tutorial_hamiltonian/)
- [Quickstart](/qkrylov/quickstart/)

<!--
ANANYA: Write the full How-To here.
Structure suggestion:
  1. 2-sentence motivation: when you want thermodynamics but full diagonalization is too slow
  2. Step 1: Choose beta (inverse temperature). Remind that beta = 1 / (k_B * T)
  3. Step 2: Single-point FTLM — FTLM(beta=2.0, n_random=100).solve(H)
  4. Step 3: Temperature sweep (the key qkrylov advantage) — 
             samples = FTLM(mode="cached").sample(H)
             results = samples.evaluate_sweep(betas)
  5. Step 4: Extract energy, specific_heat, susceptibility from the result
  6. Step 5: Plot C_v vs T curve
  7. Tabbed code blocks: Python | Julia | C++
  8. Common pitfall: n_random too small at low T → high variance; use n_random=200+
  9. See Also: theory/spectral, API: solvers.FTLM, API: solvers.FTLMSamples
-->
