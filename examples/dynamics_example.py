import qkrylov
import numpy as np

def main():
    N = 4
    basis = qkrylov.SpinHalfBasis(N=N)
    site = qkrylov.SpinHalfSite()

    # Heisenberg model
    os = qkrylov.OpSum()
    for i in range(N - 1):
        os += 1.0, qkrylov.Op.Sz, i, qkrylov.Op.Sz, i+1
        os += 0.5, qkrylov.Op.Sp, i, qkrylov.Op.Sm, i+1
        os += 0.5, qkrylov.Op.Sm, i, qkrylov.Op.Sp, i+1

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    gs = qkrylov.lanczos_ground_state(H)

    print(f"Ground state energy: {gs.energy}")

    # Initial vector phi0 = gs eigenvector
    phi0 = gs.eigenvector

    res = qkrylov.continued_fraction_coeffs(H, phi0, n_iter=20)

    omegas = np.linspace(0, 5, 100)
    S = [qkrylov.evaluate_spectral_function(res, w, gs.energy, eta=0.1) for w in omegas]

    print(f"Spectral function calculated. First few values: {S[:5]}")

if __name__ == "__main__":
    main()
