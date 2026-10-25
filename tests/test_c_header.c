/* Compiled as plain C99: proves the public C header is valid C (not only C++) and that the version and error
 * functions agree with the header the program was built against. */
#include <stdio.h>
#include <string.h>

#include <qkrylov/c_api.h>

int main(void) {
    int major = -1, minor = -1, patch = -1;
    qkrylov_version(&major, &minor, &patch);
    if (major != QKRYLOV_VERSION_MAJOR || minor != QKRYLOV_VERSION_MINOR || patch != QKRYLOV_VERSION_PATCH) {
        fprintf(stderr, "header %s does not match library %d.%d.%d\n", QKRYLOV_VERSION_STRING, major, minor, patch);
        return 1;
    }
    if (strcmp(qkrylov_version_string(), QKRYLOV_VERSION_STRING) != 0) {
        fprintf(stderr, "version string mismatch: '%s' vs '%s'\n", qkrylov_version_string(), QKRYLOV_VERSION_STRING);
        return 1;
    }
    if (QKRYLOV_VERSION_NUMBER != major * 10000 + minor * 100 + patch) return 1;

    /* a failing call must return an error and leave a readable message; no exception may escape */
    qkrylov_clear_last_error();
    if (qkrylov_sector_set_sz(NULL, 0) == QKRYLOV_SUCCESS) return 1;
    if (strlen(qkrylov_get_last_error_message()) == 0) return 1;

    /* a normal create / use / destroy round trip */
    qkrylov_sector_h sector = qkrylov_sector_create();
    if (!sector || qkrylov_sector_set_sz(sector, 0) != QKRYLOV_SUCCESS) return 1;
    qkrylov_basis_h basis = qkrylov_basis_create_spin_half(6, sector);
    if (!basis || qkrylov_basis_dimension(basis) != 20u) return 1;   /* C(6,3) */
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sector);

    printf("C header ok, qkrylov %s\n", qkrylov_version_string());
    return 0;
}
