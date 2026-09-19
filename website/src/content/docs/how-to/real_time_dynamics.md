---
title: "Real-Time Dynamics"
description: "How to propagate a quantum state in real time and compute time-dependent expectation values using qkrylov's Krylov matrix exponential solver."
---

:::note[Work in progress]
This page is being written by Ananya.
:::

## What you'll learn

- How to prepare an initial state vector (e.g., Néel state, random state).
- How to propagate it with `TimeEvolve` across a time grid.
- How to track expectation values of observables over time.

## Prerequisites

- [Building a Hamiltonian](/qkrylov/getting_started/tutorial_hamiltonian/)
- [Quickstart](/qkrylov/quickstart/)

<!--
ANANYA: Write the full How-To here.
Structure suggestion:
  1. 2-sentence motivation: quantum quench dynamics, thermalization, entanglement growth
  2. Step 1: Prepare initial state (Neel state is a great example — just set every other bit)
  3. Step 2: Define time grid — np.linspace(0, t_max, N_steps)
  4. Step 3: Run TimeEvolve(time_grid=times, n_steps=30).solve(H, psi0)
  5. Step 4: Define observables (e.g., Sz at site 0) to track during evolution
  6. Step 5: Plot <Sz(t)> — show oscillations / relaxation
  7. Tabbed code blocks: Python | Julia | C++
  8. Common pitfall: n_steps too small → inaccurate propagator; check by doubling n_steps
  9. See Also: theory/spectral, API: solvers.TimeEvolve, API: solvers.RealTimeResult
-->
