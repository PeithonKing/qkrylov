---
title: "Choosing Precision & Backend"
description: "How to select between FP32/FP64 precision and CPU/GPU/multi-GPU backends for optimal performance in qkrylov."
---

:::note[Work in progress]
This page is being written by Ananya.
:::

## What you'll learn

- When to use FP32 vs FP64 and what you lose or gain.
- How to move a Hamiltonian to GPU with `.to("cuda")`.
- How to detect available hardware at runtime.

## Prerequisites

- [Quickstart](/qkrylov/quickstart/)

<!--
ANANYA: Write the full How-To here.
Structure suggestion:
  1. Decision table at the top (the most important thing) — 4 rows:
     | System size | Precision | Solver | Reason |
     | L < 18      | FP32 or FP64 | Lanczos | Both fine, FP32 is faster |
     | L 18-24     | FP64     | Lanczos | Ghost states risk with FP32 |
     | L > 24      | FP64     | LanczosTwoPass | Memory wall |
     | Any, GPU    | FP32     | LanczosTwoPass | Max GPU throughput |
  2. How to check what hardware you have: qk.find_gpu(), qk.gpu_count()
  3. How to move to GPU: H = qk.Hamiltonian(...).to("cuda")
  4. How to move back to CPU: H = H.to("cpu")
  5. What happens if GPU is not available (graceful fallback)
  6. Tabbed code blocks: Python | Julia | C++
  7. See Also: theory/precision_stability, theory/lanczos_memory, API: qk.find_gpu
-->
