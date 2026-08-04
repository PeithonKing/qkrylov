import qkrylov
import pytest

def test_basis():
    N = 4
    basis = qkrylov.SpinHalfBasis(N)
    assert basis.size == 16
    assert basis.nsites == 4

def test_fermion_basis():
    basis = qkrylov.FermionBasis(4, conserve_n=True, n=2)
    assert basis.size == 6

def test_hamiltonian():
    N = 2
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()
    os = qkrylov.OpSum()

    os.add_term(1.0, "Sz", 0, "Sz", 1)

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    assert H.dimension == 4
    diag = H.diagonal()
    assert len(diag) == 4

def test_tj_basis():
    basis = qkrylov.TJBasis(2)
    assert basis.size == 9

def test_safety_validation():
    import numpy as np
    basis = qkrylov.SpinHalfBasis(2)
    site = qkrylov.SpinHalfSite()
    os = qkrylov.OpSum()
    os += (1.0, "Sz", 0, "Sz", 1)
    
    # Test invalid tuple length for OpSum +=
    with pytest.raises(ValueError):
        os += (1.0, "Sz")
        
    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    
    # Test wrong dimension vector passed to H.apply
    x_wrong = np.zeros(10, dtype=np.complex128)
    with pytest.raises(ValueError):
        H.apply(x_wrong)
