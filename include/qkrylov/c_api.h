#ifndef QKRYLOV_C_API_H
#define QKRYLOV_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * Return Error Codes
 * ----------------------------------------------------------------------------- */
#define QKRYLOV_SUCCESS                0
#define QKRYLOV_ERROR_INVALID_ARG     -1
#define QKRYLOV_ERROR_EXCEPTION       -2

/* -----------------------------------------------------------------------------
 * Opaque Handles
 * ----------------------------------------------------------------------------- */
typedef struct qkrylov_sector_t*      qkrylov_sector_h;
typedef struct qkrylov_basis_t*       qkrylov_basis_h;
typedef struct qkrylov_site_t*        qkrylov_site_h;
typedef struct qkrylov_opsum_t*       qkrylov_opsum_h;
typedef struct qkrylov_hamiltonian_t* qkrylov_hamiltonian_h;

/* -----------------------------------------------------------------------------
 * Sector API
 * ----------------------------------------------------------------------------- */
qkrylov_sector_h qkrylov_sector_create(void);
void             qkrylov_sector_destroy(qkrylov_sector_h sector);
int              qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
int              qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn);

/* -----------------------------------------------------------------------------
 * Basis API
 * ----------------------------------------------------------------------------- */
qkrylov_basis_h  qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
void             qkrylov_basis_destroy(qkrylov_basis_h basis);
uint64_t         qkrylov_basis_dimension(qkrylov_basis_h basis);
int              qkrylov_basis_nsites(qkrylov_basis_h basis);

/* -----------------------------------------------------------------------------
 * Site API
 * ----------------------------------------------------------------------------- */
qkrylov_site_h   qkrylov_spinhalf_site_create(void);
qkrylov_site_h   qkrylov_fermion_site_create(void);
qkrylov_site_h   qkrylov_hubbard_site_create(void);
qkrylov_site_h   qkrylov_tj_site_create(void);
void             qkrylov_site_destroy(qkrylov_site_h site);

/* -----------------------------------------------------------------------------
 * OpSum API
 * ----------------------------------------------------------------------------- */
qkrylov_opsum_h  qkrylov_opsum_create(void);
void             qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
int              qkrylov_opsum_clear(qkrylov_opsum_h opsum);
int              qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                             const char* op1, int site1);
int              qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                             const char* op1, int site1,
                                             const char* op2, int site2);

/* -----------------------------------------------------------------------------
 * Matrix-Free Hamiltonian API
 * ----------------------------------------------------------------------------- */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum);
void                  qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
uint64_t              qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);

/* Zero-copy matrix-vector apply: y = H * x */
/* x and y are pointers to complex double arrays of size dimension() */
int                   qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                                                const double* x_real, const double* x_imag,
                                                double* y_real, double* y_imag);

/* Direct zero-copy complex apply: x_complex and y_complex are contiguous arrays of 2*dimension() doubles [re, im, re, im...] */
int                   qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                                        const double* x_complex,
                                                        double* y_complex);

/* -----------------------------------------------------------------------------
 * Solvers API
 * ----------------------------------------------------------------------------- */
typedef struct {
    double energy;
} qkrylov_lanczos_result_c_t;

int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 double tol,
                                 qkrylov_lanczos_result_c_t* result);

#ifdef __cplusplus
}
#endif

#endif /* QKRYLOV_C_API_H */
