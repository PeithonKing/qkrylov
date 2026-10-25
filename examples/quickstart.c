/* Ground state of the periodic Heisenberg chain, Sz = 0 sector (C).
 * The same problem is solved in quickstart.py, quickstart.jl and quickstart.cpp.
 *
 *     cc examples/quickstart.c -I<prefix>/include -L<prefix>/lib -lqkrylov -o quickstart
 *     ./quickstart [L] [device]          e.g.  ./quickstart 20 cpu     ("gpu" needs a GPU build)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <qkrylov/c_api.h>

#define CHECK(cond, what)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "%s failed: %s\n", what, qkrylov_get_last_error_message()); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

int main(int argc, char** argv) {
    const int L = argc > 1 ? atoi(argv[1]) : 20;
    const char* device = argc > 2 ? argv[2] : "cpu";

    int major, minor, patch;
    qkrylov_version(&major, &minor, &patch);
    printf("qkrylov %d.%d.%d (header %s)\n", major, minor, patch, QKRYLOV_VERSION_STRING);

    qkrylov_sector_h sector = qkrylov_sector_create();
    CHECK(sector && qkrylov_sector_set_sz(sector, 0) == QKRYLOV_SUCCESS, "sector");   /* 2*Sz = 0 */
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(L, sector);                /* sector only: no states built yet */
    qkrylov_site_h site = qkrylov_spinhalf_site_create();
    qkrylov_opsum_h ops = qkrylov_opsum_create();
    CHECK(basis && site && ops, "basis/site/opsum");

    for (int i = 0; i < L; ++i) {
        const int j = (i + 1) % L;
        CHECK(qkrylov_opsum_add_term_2body(ops, 1.0, 0.0, "Sz", i, "Sz", j) == QKRYLOV_SUCCESS, "add Sz*Sz");
        CHECK(qkrylov_opsum_add_term_2body(ops, 0.5, 0.0, "Sp", i, "Sm", j) == QKRYLOV_SUCCESS, "add Sp*Sm");
        CHECK(qkrylov_opsum_add_term_2body(ops, 0.5, 0.0, "Sm", i, "Sp", j) == QKRYLOV_SUCCESS, "add Sm*Sp");
    }

    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create_device(basis, site, ops, device);   /* double precision */
    CHECK(H != NULL, "hamiltonian");
    /* The Hamiltonian keeps its own references, so these can go now. */
    qkrylov_opsum_destroy(ops); qkrylov_site_destroy(site); qkrylov_sector_destroy(sector);

    const unsigned long long dim = (unsigned long long)qkrylov_basis_dimension(basis);
    double* psi = (double*)malloc(dim * sizeof(double));
    CHECK(psi != NULL, "malloc");

    qkrylov_lanczos_config_t cfg;
    qkrylov_lanczos_config_default(&cfg);                     /* two-pass, real output for a real H */
    qkrylov_lanczos_result_fp64_t res;
    CHECK(qkrylov_lanczos_solve_fp64(H, &cfg, &res, psi) == QKRYLOV_SUCCESS, "lanczos");

    double norm2 = 0.0;
    for (unsigned long long k = 0; k < dim; ++k) norm2 += psi[k] * psi[k];
    printf("L = %d, device = %s, dimension = %llu\n", L, qkrylov_hamiltonian_device(H), dim);
    printf("ground-state energy  E0 = %.10f   (E0/L = %.8f)\n", res.energy, res.energy / L);
    printf("iterations = %d, converged = %d, |psi| = %.6f\n", res.iterations, res.converged, sqrt(norm2));

    free(psi);
    qkrylov_hamiltonian_destroy(H);
    qkrylov_basis_destroy(basis);
    return 0;
}
