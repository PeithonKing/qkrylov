import qkrylov as qk
import pytest
import math

def test_heisenberg_workflow():
    # 4-site chain with Sz=0 sector (dimension = C(4,2) = 6)
    basis = qk.SpinHalfBasis(N=4, sz=0)

    assert basis.size == 6, "Sz=0 sector of 4-site chain should have dim 6"
    assert basis.nsites == 4, "Basis should have 4 sites"

    site = qk.SpinHalfSite()

    # 2. Define the Physics / Interactions (OpSum)
    os = qk.OpSum()
    for i in range(3):
        # Algebraic Expression Template Syntax!
        os += 1.0 * qk.Sz(i) * qk.Sz(i+1) + 0.5 * (qk.Sp(i) * qk.Sm(i+1) + qk.Sm(i) * qk.Sp(i+1))

    assert os.size == 9, "Should have 9 terms for 3 bonds"

    # 3. Generate the Matrix-Free Hamiltonian
    H = qk.MatrixFreeHamiltonian(basis, site, os)
    assert H.dimension == 6, "Hamiltonian dimension should match Sz=0 sector size"

    # 4. Compute Low-Energy States (Algorithms)
    res = qk.lanczos_ground_state(H, 200, 1e-12)
    
    # Exact energy for 4-site Heisenberg chain OBC
    exact_energy = -1.6160254038
    assert math.isclose(res.energy, exact_energy, abs_tol=1e-5), "Ground state energy should match exact 4-site Heisenberg value"
    assert len(res.eigenvector) == 6, "Eigenvector dimension should match Sz=0 sector size"

if __name__ == "__main__":
    test_heisenberg_workflow()
