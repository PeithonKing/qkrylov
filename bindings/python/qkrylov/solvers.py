import numpy as np
from typing import List, Tuple
from . import _qkrylov_cpp as _cpp
from .hamiltonian import MatrixFreeHamiltonian

class LanczosResult:
    """Result of a Lanczos ground state calculation.
    
    Attributes
    ----------
    energy : float
        The computed ground state energy.
    eigenvector : np.ndarray
        The ground state eigenvector.
    """
    def __init__(self, energy: float, eigenvector: np.ndarray):
        self.energy = energy
        self.eigenvector = eigenvector
        
    def __repr__(self) -> str:
        return f"LanczosResult(energy={self.energy:.10f})"


def lanczos_ground_state(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12
) -> LanczosResult:
    """Find the ground state of a Hamiltonian using the Lanczos algorithm.
    
    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    maxiter : int, optional
        Maximum number of Lanczos iterations (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-12).
        
    Returns
    -------
    LanczosResult
        The ground state energy and eigenvector.
    """
    res = _cpp.lanczos_ground_state(H._cpp_obj, maxiter, tol)
    return LanczosResult(
        energy=res.energy,
        eigenvector=np.array(res.eigenvector, dtype=np.complex128)
    )


class DavidsonResult:
    """Result of a Davidson calculation.
    
    Attributes
    ----------
    eigenvalues : np.ndarray
        The lowest eigenvalues.
    eigenvectors : List[np.ndarray]
        The corresponding eigenvectors.
    """
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: List[np.ndarray]):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
    def __repr__(self) -> str:
        return f"DavidsonResult(energies={self.eigenvalues})"


def davidson_lowest(
    H: MatrixFreeHamiltonian,
    n_eig: int = 1,
    max_subspace: int = 20,
    tol: float = 1e-8
) -> DavidsonResult:
    """Find the lowest eigenpairs using the Davidson algorithm.
    
    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    n_eig : int, optional
        Number of eigenpairs to compute (default 1).
    max_subspace : int, optional
        Maximum subspace size before restart (default 20).
    tol : float, optional
        Convergence tolerance (default 1e-8).
        
    Returns
    -------
    DavidsonResult
        The lowest eigenvalues and eigenvectors.
    """
    res = _cpp.davidson_lowest(H._cpp_obj, n_eig, max_subspace, tol)
    
    return DavidsonResult(
        eigenvalues=np.array(res.eigenvalues, dtype=float),
        eigenvectors=[np.array(ev, dtype=np.complex128) for ev in res.eigenvectors]
    )

# For Dynamics and FTLM, we wrap them directly as well.

class DynamicsResult:
    """Result of continued fraction Lanczos."""
    def __init__(self, cpp_res):
        self._cpp_obj = cpp_res
        self.alphas = np.array(cpp_res.alphas, dtype=float)
        self.betas = np.array(cpp_res.betas, dtype=float)
        self.norm_phi0 = cpp_res.norm_phi0

def continued_fraction_coeffs(
    H: MatrixFreeHamiltonian,
    phi0: np.ndarray,
    n_iter: int = 100
) -> DynamicsResult:
    phi0_list = phi0.tolist()
    res = _cpp.continued_fraction_coeffs(H._cpp_obj, phi0_list, n_iter)
    return DynamicsResult(res)

def evaluate_spectral_function(
    res: DynamicsResult,
    omega: float,
    E0: float,
    eta: float = 0.1
) -> float:
    return _cpp.evaluate_spectral_function(res._cpp_obj, omega, E0, eta)

class FTLMResult:
    def __init__(self, cpp_res):
        self.beta = cpp_res.beta
        self.partition_function = cpp_res.partition_function
        self.internal_energy = cpp_res.internal_energy
        self.specific_heat = cpp_res.specific_heat

def ftlm(
    H: MatrixFreeHamiltonian,
    beta: float,
    n_random: int = 50,
    n_steps: int = 100
) -> FTLMResult:
    res = _cpp.ftlm(H._cpp_obj, beta, n_random, n_steps)
    return FTLMResult(res)
