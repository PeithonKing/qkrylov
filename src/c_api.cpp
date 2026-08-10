#include "qkrylov/c_api.h"

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <memory>
#include <vector>
#include <complex>
#include <cstring>
#include <exception>

using namespace qkrylov;

// Internal wrapper structs holding shared/unique pointers to C++ objects
struct qkrylov_sector_t {
    Sector sector;
};

struct qkrylov_basis_t {
    std::shared_ptr<Basis> ptr;
};

struct qkrylov_site_t {
    std::shared_ptr<Site> ptr;
};

struct qkrylov_opsum_t {
    OpSum opsum;
};

struct qkrylov_hamiltonian_t {
    std::unique_ptr<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>> ptr;
};

extern "C" {

/* Sector API */
qkrylov_sector_h qkrylov_sector_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_sector_t>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_sector_destroy(qkrylov_sector_h sector) {
    if (sector) delete sector;
}

int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_sz = true;
        sector->sector.sz2 = sz2;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_nup = true;
        sector->sector.use_ndn = true;
        sector->sector.nup = nup;
        sector->sector.ndn = ndn;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* Basis API */
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<SpinHalfBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<FermionBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<HubbardBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<TJBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_basis_destroy(qkrylov_basis_h basis) {
    if (basis) delete basis;
}

uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis) {
    if (!basis || !basis->ptr) return 0;
    try {
        return basis->ptr->size();
    } catch (...) {
        return 0;
    }
}

int qkrylov_basis_nsites(qkrylov_basis_h basis) {
    if (!basis || !basis->ptr) return 0;
    try {
        if (auto b = std::dynamic_pointer_cast<SpinHalfBasis>(basis->ptr)) return b->nsites();
        if (auto b = std::dynamic_pointer_cast<FermionBasis>(basis->ptr))  return b->nsites();
        if (auto b = std::dynamic_pointer_cast<HubbardBasis>(basis->ptr))  return b->nsites();
        if (auto b = std::dynamic_pointer_cast<TJBasis>(basis->ptr))       return b->nsites();
        return 0;
    } catch (...) {
        return 0;
    }
}

/* Site API */
qkrylov_site_h qkrylov_spinhalf_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<SpinHalfSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_fermion_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<FermionSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_hubbard_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<HubbardSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_tj_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<TJSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_site_destroy(qkrylov_site_h site) {
    if (site) delete site;
}

/* OpSum API */
qkrylov_opsum_h qkrylov_opsum_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_opsum_t>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_opsum_destroy(qkrylov_opsum_h opsum) {
    if (opsum) delete opsum;
}

int qkrylov_opsum_clear(qkrylov_opsum_h opsum) {
    if (!opsum) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        opsum->opsum.clear();
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1) {
    if (!opsum || !op1) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        OperatorTerm term;
        term.coeff = std::complex<float>(coeff_real, coeff_imag);
        term.factors.push_back({std::string(op1), site1});
        opsum->opsum.add_term(term);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1,
                                 const char* op2, int site2) {
    if (!opsum || !op1 || !op2) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        OperatorTerm term;
        term.coeff = std::complex<float>(coeff_real, coeff_imag);
        term.factors.push_back({std::string(op1), site1});
        term.factors.push_back({std::string(op2), site2});
        opsum->opsum.add_term(term);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> API */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum) {
    if (!basis || !basis->ptr || !site || !site->ptr || !opsum) return nullptr;
    try {
        auto handle = std::make_unique<qkrylov_hamiltonian_t>();
        handle->ptr = std::make_unique<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>>(basis->ptr, site->ptr, opsum->opsum, Device());
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h) {
    if (h) delete h;
}

uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->dimension();
    } catch (...) {
        return 0;
    }
}

int qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                              const float* x_real, const float* x_imag,
                              float* y_real, float* y_imag) {
    if (!h || !h->ptr || !x_real || !y_real) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        const uint64_t dim = h->ptr->dimension();
        std::vector<std::complex<float>> x(dim);
        std::vector<std::complex<float>> y(dim);

        for (uint64_t i = 0; i < dim; ++i) {
            float imag = x_imag ? x_imag[i] : 0.0;
            x[i] = std::complex<float>(x_real[i], imag);
        }

        h->ptr->apply(x.data(), y.data());

        for (uint64_t i = 0; i < dim; ++i) {
            y_real[i] = y[i].real();
            if (y_imag) y_imag[i] = y[i].imag();
        }

        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                        const float* x_complex,
                                        float* y_complex) {
    if (!h || !h->ptr || !x_complex || !y_complex) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        const auto* x_c = reinterpret_cast<const std::complex<float>*>(x_complex);
        auto* y_c = reinterpret_cast<std::complex<float>*>(y_complex);
        h->ptr->apply(x_c, y_c);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* Solvers API */
int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 float tol,
                                 qkrylov_lanczos_result_c_t* result) {
    if (!h || !h->ptr || !result) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        auto res = lanczos_ground_state(*(h->ptr), maxiter, tol);
        result->energy = res.energy;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

} // extern "C"
