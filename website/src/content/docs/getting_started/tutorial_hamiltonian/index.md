---
title: "Building the Hamiltonian"
description: "How to construct any quantum lattice Hamiltonian using qkrylov's operator sum interface across Python, Julia, and C++."
---

:::note[Work in progress]
This page is being written by Ananya. The polyglot code examples (Python, Julia, C++) are already in the source files `main.ipynb`, `main.jl`, and `main.cpp` in this folder.
:::

## What you'll learn

- How `OpSum`, sites, and bases fit together to define a Hamiltonian.
- How to construct the 1D Heisenberg, Hubbard, and t-J models.
- How to target CPU or GPU with a single `.to("cuda")` call.

<!-- 
ANANYA: Write the full tutorial here.
Structure suggestion:
  1. Intro paragraph — what a Hamiltonian is in qkrylov terms
  2. Step 1: Choose your lattice model / physical system
  3. Step 2: Define the site (SpinHalfSite, FermionSite, etc.)
  4. Step 3: Define the basis + sector (Sz=0, half-filling, etc.)
  5. Step 4: Build the OpSum (show Heisenberg as canonical example)
  6. Step 5: Construct the Hamiltonian and move to device
  7. Tabbed code blocks: Python | Julia | C++
  8. See Also: theory/exact_diag, API: hamiltonian, API: basis
-->
