import qkrylov
import numpy as np

def main():
    N = 4
    # Create basis and site
    basis = qkrylov.SpinHalfBasis(N=N)
    site = qkrylov.SpinHalfSite()

    # Create Heisenberg Hamiltonian
    os = qkrylov.OpSum()
    for i in range(N - 1):
        os += 1.0, qkrylov.Op.Sz, i, qkrylov.Op.Sz, i+1
        os += 0.5, qkrylov.Op.Sp, i, qkrylov.Op.Sm, i+1
        os += 0.5, qkrylov.Op.Sm, i, qkrylov.Op.Sp, i+1

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)

    # Solve using Lanczos
    result = qkrylov.lanczos_ground_state(H)
    print(f"Lanczos Ground state energy: {result.energy}")

    # Solve using Davidson
    dav_result = qkrylov.davidson_lowest(H, n_eig=1)
    print(f"Davidson Ground state energy: {dav_result.eigenvalues[0]}")

if __name__ == "__main__":
    main()
