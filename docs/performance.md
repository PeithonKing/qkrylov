# Performance Guide

!!! note "Under Construction"
    This page will cover performance benchmarks, tips for large system sizes, and memory usage profiling.

## Key Facts

- **Matrix-free:** Memory scales as $O(\dim)$, not $O(\dim^2)$.  
  A 20-site Heisenberg chain has $\dim = \binom{20}{10} = 184{,}756$. Storing its Hamiltonian as a dense matrix would require **~500 GB**. qkrylov uses **~3 MB**.
- **OpenMP:** All matrix-vector products are parallelized across all CPU cores automatically.
- **Zero-copy Python bridge:** NumPy arrays passed to C++ are never copied — raw pointers are passed directly.

## Coming Soon

- Benchmarks vs. QuSpin, EDLib, and Lanczos reference implementations
- Scaling plots: time vs. system size
- Memory usage measurements
- Tips for choosing solver parameters
