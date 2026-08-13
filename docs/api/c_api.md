# `qkrylov` C API Reference (`extern "C"`)

The **`qkrylov` C API** provides a binary-stable, unmangled `extern "C"` interface for performing matrix-free exact diagonalization and Krylov calculations in quantum many-body physics. 

It enables seamless zero-copy interop with languages such as **C**, **Julia (`ccall`)**, **Rust (FFI)**, **Python (`ctypes`/`cffi`)**, **Fortran**, and **Go**.

---

## Table of Contents
1. [Overview & Core Features](#overview--core-features)
2. [Include & Linking](#include--linking)
3. [Opaque Handles & Data Types](#opaque-handles--data-types)
4. [Error Codes](#error-codes)
5. [API Function Reference](#api-function-reference)
   - [Sector API](#1-sector-api)
   - [Basis API](#2-basis-api)
   - [Site API](#3-site-api)
   - [OpSum API](#4-opsum-api)
   - [Matrix-Free Hamiltonian API](#5-matrix-free-hamiltonian-api)
   - [Solvers API](#6-solvers-api)
6. [Complete C Example](#complete-c-example)

---

## Overview & Core Features

- **Flat C ABI Linkage**: All functions are declared inside `extern "C"` blocks without C++ name mangling or class vtable dependencies.
- **Opaque Handle Architecture**: C++ class instances (`Basis`, `Site`, `OpSum`, `MatrixFreeHamiltonian`) are encapsulated behind opaque struct pointers (handles).
- **Exception Safety**: Every C function catches internal C++ exceptions and returns standard integer error codes to prevent process crashes across language boundaries.
- **RAII Leak Protection**: Handle allocations use RAII mechanisms internally to ensure memory is never leaked even if construction fails.
- **Zero-Copy Performance**: Supports direct matrix-vector evaluation (`y = H * x`) using contiguous memory pointers without vector re-allocations or buffer copies.

---

## Include & Linking

To use the C API in your program, include the header file:

```c
#include <qkrylov/c_api.h>
```

Link against the built `qkrylov` library:

```bash
# Compile and link a C program
gcc -O3 main.c -I/path/to/qkrylov/include -L/path/to/qkrylov/build -lqkrylov -lm -o main_c
```

---

## Opaque Handles & Data Types

The C API uses incomplete struct pointers to represent C++ object instances:

| Handle Type | Encapsulated C++ Object | Description |
| :--- | :--- | :--- |
| `qkrylov_sector_h` | `qkrylov::Sector` | Quantum number conservation law / symmetry sector |
| `qkrylov_basis_h` | `std::shared_ptr<qkrylov::Basis>` | Basis (SpinHalf, Fermion, Hubbard, t-J) |
| `qkrylov_site_h` | `std::shared_ptr<qkrylov::Site>` | Site operator definition (SpinHalf, Fermion, Hubbard, t-J) |
| `qkrylov_opsum_h` | `qkrylov::OpSum` | Linear combination of local operator product terms |
| `qkrylov_hamiltonian_h` | `std::unique_ptr<MatrixFreeHamiltonian>` | Matrix-free Hamiltonian evaluator ($y = Hx$) |

---

## Error Codes

Functions returning an `int` status code return one of the following macros defined in `qkrylov/c_api.h`:

| Constant | Value | Description |
| :--- | :---: | :--- |
| `QKRYLOV_SUCCESS` | `0` | Operation completed successfully. |
| `QKRYLOV_ERROR_INVALID_ARG` | `-1` | Null pointer passed or invalid argument range. |
| `QKRYLOV_ERROR_EXCEPTION` | `-2` | An internal C++ exception was caught safely. |

---

## API Function Reference

### 1. Sector API
Symmetries and quantum number sectors (e.g. total $S^z$, particle numbers).

#### `qkrylov_sector_create`
```c
qkrylov_sector_h qkrylov_sector_create(void);
```
Creates a new default symmetry sector handle. Returns `NULL` on allocation failure.

#### `qkrylov_sector_destroy`
```c
void qkrylov_sector_destroy(qkrylov_sector_h sector);
```
Frees the memory associated with a sector handle.

#### `qkrylov_sector_set_sz`
```c
int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
```
Enforces total $S^z$ conservation sector where `sz2` is $2 \times S^z$ (e.g., `sz2 = 0` for $S^z = 0$). Returns `QKRYLOV_SUCCESS` on success.

#### `qkrylov_sector_set_hubbard_particles`
```c
int qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn);
```
Enforces particle number conservation for electron systems ($N_\uparrow$ and $N_\downarrow$).

---

### 2. Basis API
Constructs many-body Hilbert space bases.

#### `qkrylov_spinhalf_basis_create`
```c
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a Spin-1/2 basis for `num_sites` sites, optionally restricted by `sector` (pass `NULL` for full basis).

#### `qkrylov_fermion_basis_create`
```c
qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a spinless fermion basis with Jordan-Wigner signs.

#### `qkrylov_hubbard_basis_create`
```c
qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates an interacting Fermi-Hubbard basis (spin-$\uparrow$ and spin-$\downarrow$).

#### `qkrylov_tj_basis_create`
```c
qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a t-J model basis enforcing the no-double-occupancy constraint.

#### `qkrylov_basis_destroy`
```c
void qkrylov_basis_destroy(qkrylov_basis_h basis);
```
Destroys a basis handle and frees associated resources.

#### `qkrylov_basis_dimension`
```c
uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis);
```
Returns the total dimension of the Hilbert space basis.

#### `qkrylov_basis_nsites`
```c
int qkrylov_basis_nsites(qkrylov_basis_h basis);
```
Returns the number of lattice sites in the basis.

---

### 3. Site API
Defines site operator representations.

#### Constructor Functions
```c
qkrylov_site_h qkrylov_spinhalf_site_create(void);
qkrylov_site_h qkrylov_fermion_site_create(void);
qkrylov_site_h qkrylov_hubbard_site_create(void);
qkrylov_site_h qkrylov_tj_site_create(void);
```
Creates a site handle matching the corresponding model system.

#### `qkrylov_site_destroy`
```c
void qkrylov_site_destroy(qkrylov_site_h site);
```
Frees a site handle.

---

### 4. OpSum API
Constructs linear combinations of operator strings.

#### `qkrylov_opsum_create`
```c
qkrylov_opsum_h qkrylov_opsum_create(void);
```
Creates a new empty `OpSum` operator sum container.

#### `qkrylov_opsum_destroy`
```c
void qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
```
Frees an `OpSum` container.

#### `qkrylov_opsum_clear`
```c
int qkrylov_opsum_clear(qkrylov_opsum_h opsum);
```
Removes all operator terms from the container.

#### `qkrylov_opsum_add_term_1body`
```c
int qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1);
```
Adds a single-body operator term (e.g. $h \cdot S_i^z$).

#### `qkrylov_opsum_add_term_2body`
```c
int qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1,
                                 const char* op2, int site2);
```
Adds a two-body operator interaction term (e.g. $J \cdot S_i^z S_j^z$ or $\frac{J}{2} S_i^+ S_j^-$).

---

### 5. Matrix-Free Hamiltonian API
Evaluates matrix-vector multiplication $y = Hx$ without storing the matrix.

#### `qkrylov_hamiltonian_create`
```c
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum);
```
Creates a `MatrixFreeHamiltonian` evaluator handle combining a basis, site operator rules, and operator sum terms.

#### `qkrylov_hamiltonian_destroy`
```c
void qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
```
Frees a Hamiltonian handle.

#### `qkrylov_hamiltonian_dimension`
```c
uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);
```
Returns the matrix dimension of the Hamiltonian.

#### `qkrylov_hamiltonian_apply`
```c
int qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                              const float* x_real, const float* x_imag,
                              float* y_real, float* y_imag);
```
Applies matrix action using separate real and imaginary arrays of size `dimension()`. `x_imag` and `y_imag` may be `NULL` if vectors are purely real.

#### `qkrylov_hamiltonian_apply_complex` (Zero-Copy)
```c
int qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                        const float* x_complex,
                                        float* y_complex);
```
Performs **direct zero-copy** matrix-vector multiplication where `x_complex` and `y_complex` are contiguous arrays of $2 \times \text{dimension()}$ floats `[re, im, re, im...]`.

---

### 6. Solvers API

#### `qkrylov_lanczos_ground_state`
```c
typedef struct {
    float energy;
} qkrylov_lanczos_result_c_t;

int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 float tol,
                                 qkrylov_lanczos_result_c_t* result);
```
Computes the ground state energy of the Hamiltonian using the Lanczos algorithm. Stores the result in `result`.

---

## Complete C Example

Below is a complete, working C program demonstrating exact diagonalization of a 4-site 1D Heisenberg antiferromagnet ($H = \sum_i S_i^z S_{i+1}^z + \frac{1}{2} (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+)$) in the $S^z = 0$ symmetry sector:

```c
#include <qkrylov/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    int N = 4;
    printf("--- qkrylov C API Demo: 4-site Heisenberg Chain ---\n");

    // 1. Create Sz=0 sector
    qkrylov_sector_h sector = qkrylov_sector_create();
    qkrylov_sector_set_sz(sector, 0);

    // 2. Create Spin-1/2 basis in Sz=0 sector
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sector);
    uint64_t dim = qkrylov_basis_dimension(basis);
    printf("Basis dimension: %llu\n", (unsigned long long)dim);

    // 3. Create SpinHalf Site operator rules
    qkrylov_site_h site = qkrylov_spinhalf_site_create();

    // 4. Construct Heisenberg OpSum
    qkrylov_opsum_h opsum = qkrylov_opsum_create();
    for (int i = 0; i < N - 1; ++i) {
        // Sz_i Sz_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 1.0f, 0.0f, "Sz", i, "Sz", i + 1);
        // 0.5 * Sp_i Sm_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 0.5f, 0.0f, "Sp", i, "Sm", i + 1);
        // 0.5 * Sm_i Sp_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 0.5f, 0.0f, "Sm", i, "Sp", i + 1);
    }

    // 5. Create Matrix-Free Hamiltonian
    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);

    // 6. Run Lanczos Solver
    qkrylov_lanczos_result_c_t result;
    int status = qkrylov_lanczos_ground_state(H, 200, 1e-12f, &result);

    if (status == QKRYLOV_SUCCESS) {
        printf("Ground State Energy: %.10f\n", result.energy);
    } else {
        printf("Lanczos solver failed with error code: %d\n", status);
    }

    // 7. Cleanup Handles safely
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sector);

    return 0;
}
```
