import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Union, Sequence
from dataclasses import dataclass
from . import _qkrylov_cpp as _cpp
from .hamiltonian import MatrixFreeHamiltonian

class LanczosResult:
    """Container holding the converged ground state energy and eigenvector.

    Lanczos convergence is superexponential: the lowest Ritz value typically
    converges to machine precision in O(sqrt(N)) iterations, far fewer than the
    full Hilbert space dimension N. The exact rate depends on the spectral gap
    — the energy difference between the ground state and the first excited state.
    A larger gap means faster convergence.

    The residual norm ``||H|psi> - E|psi>||`` is the rigorous convergence
    diagnostic. ``converged=True`` means this residual fell below ``tol``.

    See Also
    --------
    theory: :doc:`/theory/lanczos_memory`

    Attributes
    ----------
    energy : float
        Ground state eigenvalue E0.
    eigenvector : np.ndarray
        Normalized ground state eigenvector |psi0>.
    iterations : int
        Number of Krylov iterations completed.
    converged : bool
        Whether the iteration satisfied convergence tolerance.
    """
    def __init__(self, energy: float, eigenvector: np.ndarray, iterations: int = 0, converged: bool = True):
        self.energy = energy
        self.eigenvector = eigenvector
        self.iterations = iterations
        self.converged = converged

    def __iter__(self):
        return iter((self.energy, self.eigenvector))

    def __getitem__(self, idx):
        return (self.energy, self.eigenvector)[idx]

    def __len__(self):
        return 2
        
    def __repr__(self) -> str:
        return f"LanczosResult(energy={self.energy:.10f})"


class Solver(ABC):
    """Abstract base class for all eigensolvers and dynamical solvers.

    All solvers in qkrylov follow the same two-step pattern::

        solver = qk.solvers.LanczosTwoPass(maxiter=300, tol=1e-10)
        result = solver.solve(H)

    Solver instances are also directly callable (``solver(H)``), making them
    composable with higher-order functions and SciPy-style pipelines.
    """

    @abstractmethod
    def solve(self, H: MatrixFreeHamiltonian, *args, **kwargs):
        """Execute the solver algorithm on the Hamiltonian."""
        pass

    def __call__(self, H: MatrixFreeHamiltonian, *args, **kwargs):
        """Allow calling solver instance directly: `solver(H)`."""
        return self.solve(H, *args, **kwargs)


class Lanczos(Solver):
    """Standard single-pass Lanczos eigensolver.

    Starting from a random unit vector ``|v0>``, repeatedly applies the
    Hamiltonian to build a Krylov subspace ``K_k(H, v0) = span{v0, Hv0, H²v0, ...}``.
    At each step the three-term recurrence

    .. code-block:: text

        beta_{j+1} |v_{j+1}> = H|v_j> - alpha_j|v_j> - beta_j|v_{j-1}>

    produces the diagonal (alpha) and off-diagonal (beta) elements of a
    tridiagonal matrix T_k. The eigenvalues of T_k — called **Ritz values** —
    converge to the true eigenvalues of H from the outside in.

    **Memory cost:** O(k * N) — all k Krylov vectors are stored simultaneously.
    For L=20 spin-1/2 sites (N = 2^20 ≈ 10^6 complex floats), this is ~1.6 GB
    at k=100 in FP64. Use :class:`LanczosTwoPass` for larger systems.

    .. note::
        **FP32 vs FP64:** FP32 is faster and uses half the memory, but loss of
        orthogonality accumulates sooner, producing ghost eigenvalues. For
        production ground-state calculations, prefer FP64.
        See :doc:`/theory/precision_stability`.

    See Also
    --------
    LanczosTwoPass : Memory-efficient two-pass variant. Use for L > 22.
    theory: :doc:`/theory/lanczos_memory`

    Parameters
    ----------
    maxiter : int, default=200
        Maximum Lanczos iteration steps.
    tol : float, default=1e-12
        Convergence tolerance on the lowest Ritz value residual norm.
    max_iter : Optional[int], default=None
        Alias for maxiter (for SciPy-style compatibility).

    Examples
    --------
    >>> import qkrylov as qk
    >>> basis = qk.basis.SpinHalfBasis(N=16, sz=0)
    >>> site  = qk.site.SpinHalfSite()
    >>> ops   = qk.OpSum()
    >>> for i in range(15):
    ...     ops += qk.Sz(i)*qk.Sz(i+1) + 0.5*(qk.Sp(i)*qk.Sm(i+1) + qk.Sm(i)*qk.Sp(i+1))
    >>> H = qk.Hamiltonian(basis, site, ops)
    >>> result = qk.solvers.Lanczos(maxiter=200).solve(H)
    >>> print(result.energy)   # Ground state energy of 16-site Heisenberg chain
    -7.1420...
    """

    def __init__(self, maxiter: int = 200, tol: float = 1e-12, max_iter: Optional[int] = None):
        if max_iter is not None:
            maxiter = max_iter
        self.maxiter = int(maxiter)
        self.tol = float(tol)

    @property
    def max_iter(self) -> int:
        return self.maxiter

    @max_iter.setter
    def max_iter(self, val: int):
        self.maxiter = int(val)

    def solve(self, H: MatrixFreeHamiltonian) -> LanczosResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"lanczos_ground_state_{H._backend_suffix}{s_dtype}")
        energy, eigenvector = fn(H._cpp_obj, self.maxiter, self.tol)
        return LanczosResult(energy=energy, eigenvector=eigenvector)


class LanczosTwoPass(Solver):
    """Memory-efficient two-pass Lanczos eigensolver for extreme-scale Hilbert spaces.

    The standard Lanczos algorithm stores all k Krylov vectors simultaneously —
    O(k * N) memory. For L=28 spin-1/2 sites this is already hundreds of GB.
    The Two-Pass algorithm breaks this wall at the cost of one extra Hamiltonian
    application per iteration:

    - **Pass 1:** Run k Lanczos steps. Store only the tridiagonal (alpha, beta)
      arrays (O(k)). Diagonalize to get Ritz values and Ritz coefficients.
      **Discard all Krylov vectors.**
    - **Pass 2:** Regenerate every Krylov vector from scratch using the same
      recurrence, immediately contracting it with the Ritz coefficient to
      accumulate the eigenvector. Memory footprint: only 3 vectors (O(3N)).

    The total memory cost drops from O(k * N) to **O(3N)**, enabling ground
    state calculations at L=28–32 on a single GPU.

    .. warning::
        Two-Pass doubles the number of SpMV operations versus single-pass.
        The memory saving is always worth it at large L, but for small systems
        (L < 18) the single-pass :class:`Lanczos` solver is faster overall.

    See Also
    --------
    Lanczos : Single-pass variant. Faster for small systems.
    theory: :doc:`/theory/lanczos_memory`

    Parameters
    ----------
    maxiter : int, default=200
        Maximum Lanczos iteration steps.
    tol : float, default=1e-12
        Eigenvalue convergence tolerance.
    max_iter : Optional[int], default=None
        Alias for maxiter.

    Examples
    --------
    >>> import qkrylov as qk
    >>> # L=24 needs Two-Pass; single-pass would require ~12 GB at k=200
    >>> basis  = qk.basis.SpinHalfBasis(N=24, sz=0)
    >>> site   = qk.site.SpinHalfSite()
    >>> ops    = qk.OpSum()
    >>> for i in range(23):
    ...     ops += qk.Sz(i)*qk.Sz(i+1) + 0.5*(qk.Sp(i)*qk.Sm(i+1) + qk.Sm(i)*qk.Sp(i+1))
    >>> H      = qk.Hamiltonian(basis, site, ops).to("cuda")
    >>> result = qk.solvers.LanczosTwoPass(maxiter=200).solve(H)
    >>> print(result.energy)
    -10.7...
    """

    def __init__(self, maxiter: int = 200, tol: float = 1e-12, max_iter: Optional[int] = None):
        if max_iter is not None:
            maxiter = max_iter
        self.maxiter = int(maxiter)
        self.tol = float(tol)

    @property
    def max_iter(self) -> int:
        return self.maxiter

    @max_iter.setter
    def max_iter(self, val: int):
        self.maxiter = int(val)

    def solve(self, H: MatrixFreeHamiltonian) -> LanczosResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"lanczos_two_pass_{H._backend_suffix}{s_dtype}")
        energy, eigenvector = fn(H._cpp_obj, self.maxiter, self.tol)
        return LanczosResult(energy=energy, eigenvector=eigenvector)


def lanczos_ground_state(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12
) -> LanczosResult:
    """Find the ground state of a Hamiltonian using the single-pass Lanczos algorithm.

    Functional convenience wrapper around :class:`Lanczos`. Prefer the class
    interface when you want to reuse the same solver configuration across
    multiple Hamiltonians.

    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian to diagonalize.
    maxiter : int, default=200
        Maximum number of Lanczos iterations.
    tol : float, default=1e-12
        Convergence tolerance on the residual norm.

    Returns
    -------
    LanczosResult
        The ground state energy and eigenvector.

    See Also
    --------
    lanczos_two_pass : Memory-efficient variant for large systems.
    Lanczos : OOP interface.

    Examples
    --------
    >>> result = qk.solvers.lanczos_ground_state(H, maxiter=300)
    >>> energy, psi0 = result   # unpack like a tuple
    """
    return Lanczos(maxiter=maxiter, tol=tol).solve(H)


def lanczos_two_pass(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12
) -> LanczosResult:
    """Find the ground state using the memory-efficient Two-Pass Lanczos algorithm.

    Functional convenience wrapper around :class:`LanczosTwoPass`. Uses O(3N)
    memory instead of O(k*N), enabling systems with L > 22 on modest hardware.

    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian to diagonalize.
    maxiter : int, default=200
        Maximum number of Lanczos iterations.
    tol : float, default=1e-12
        Convergence tolerance.

    Returns
    -------
    LanczosResult
        The ground state energy and eigenvector.

    See Also
    --------
    lanczos_ground_state : Single-pass variant.
    LanczosTwoPass : OOP interface.
    """
    return LanczosTwoPass(maxiter=maxiter, tol=tol).solve(H)


class DavidsonResult:
    """Result of a Davidson eigensolver calculation.

    Unlike Lanczos (which targets the single lowest eigenvalue), Davidson
    can simultaneously converge multiple lowest eigenvalues. Use it when you
    need the first few excited states in addition to the ground state.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Array of the lowest ``n_eig`` eigenvalues, sorted ascending.
    eigenvectors : List[np.ndarray]
        Corresponding normalized eigenvectors.
    """
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: List[np.ndarray]):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def __iter__(self):
        return iter((self.eigenvalues, self.eigenvectors))

    def __getitem__(self, idx):
        return (self.eigenvalues, self.eigenvectors)[idx]

    def __len__(self):
        return 2
        
    def __repr__(self) -> str:
        return f"DavidsonResult(energies={self.eigenvalues})"


class Davidson(Solver):
    """Preconditioned Davidson subspace eigensolver.

    The Davidson algorithm builds a **growing subspace** V, unlike the fixed
    three-term Lanczos recurrence. At each step, it solves the small projected
    eigenvalue problem in V, computes the residual ``r = H|u> - theta|u>``
    for the current Ritz pair, then adds the **preconditioned correction**
    ``|t> = (D - theta I)^{-1} |r>`` (where D is the diagonal of H) to V.

    Because the diagonal preconditioner targets the dominant part of H,
    Davidson converges much faster than Lanczos on diagonally dominant
    Hamiltonians (e.g., Hubbard model with large on-site U) but offers
    no advantage for Heisenberg-like models where diagonal and off-diagonal
    contributions are comparable.

    **Subspace collapse:** When the subspace dimension hits ``max_subspace``,
    it is collapsed back to the ``n_eig`` current best Ritz vectors before
    expanding again.

    See Also
    --------
    Lanczos : Better choice for Heisenberg-type models.
    LanczosTwoPass : Better choice for large Hubbard models.

    Parameters
    ----------
    n_eig : int, default=1
        Number of lowest eigenvalues and eigenvectors to compute.
    max_subspace : int, default=20
        Maximum dimension of the projected subspace before collapse.
    tol : float, default=1e-8
        Convergence tolerance for residual norms.

    Examples
    --------
    >>> # Get the 3 lowest states of the Hubbard model
    >>> result = qk.solvers.Davidson(n_eig=3).solve(H)
    >>> for i, (E, psi) in enumerate(zip(result.eigenvalues, result.eigenvectors)):
    ...     print(f"E{i} = {E:.8f}")
    """

    def __init__(self, n_eig: int = 1, max_subspace: int = 20, tol: float = 1e-8):
        self.n_eig = int(n_eig)
        self.max_subspace = int(max_subspace)
        self.tol = float(tol)

    def solve(self, H: MatrixFreeHamiltonian) -> DavidsonResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"davidson_lowest_{H._backend_suffix}{s_dtype}")
        res = fn(H._cpp_obj, self.n_eig, self.max_subspace, self.tol)
        return DavidsonResult(
            eigenvalues=np.array(res.eigenvalues, dtype=float),
            eigenvectors=[np.asarray(ev) for ev in res.eigenvectors]
        )


def davidson_lowest(
    H: MatrixFreeHamiltonian,
    n_eig: int = 1,
    max_subspace: int = 20,
    tol: float = 1e-8
) -> DavidsonResult:
    """Find the lowest eigenpairs using the Davidson algorithm.

    Convenience wrapper around :class:`Davidson`. Prefer Davidson over
    Lanczos when computing multiple eigenstates simultaneously or when
    the Hamiltonian is diagonally dominant (large on-site interactions).

    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    n_eig : int, default=1
        Number of lowest eigenpairs to compute.
    max_subspace : int, default=20
        Maximum projected subspace size before collapse.
    tol : float, default=1e-8
        Residual norm convergence tolerance.

    Returns
    -------
    DavidsonResult
        Eigenvalues and eigenvectors.
    """
    return Davidson(n_eig=n_eig, max_subspace=max_subspace, tol=tol).solve(H)


class DynamicsResult:
    """Result of the continued-fraction Lanczos dynamical solver.

    Stores the raw Lanczos coefficients (alpha, beta) and the norm of the
    excited state ``|phi0> = Op|psi0>`` produced during the run. These are
    the minimal ingredients needed to evaluate the Green's function

    .. code-block:: text

        G(omega) = <phi0| (omega - H + i*eta)^{-1} |phi0>
                 = norm_phi0^2 / (omega - alpha_0 - beta_1^2 / (omega - alpha_1 - ...))

    via the continued fraction expansion. Post-process with ``spectral_function``
    to convert to S(omega) = -(1/pi) Im G(omega).

    See Also
    --------
    qk.solvers.spectral_function : Convert DynamicsResult to S(omega).
    theory: :doc:`/theory/spectral`

    Attributes
    ----------
    alphas : np.ndarray
        Diagonal Lanczos coefficients.
    betas : np.ndarray
        Off-diagonal Lanczos coefficients.
    norm_phi0 : float
        Norm of the excited state Op|psi0>.
    """
    def __init__(self, alphas, betas, norm_phi0):
        self.alphas = np.asarray(alphas)
        self.betas  = np.asarray(betas)
        self.norm_phi0 = norm_phi0

    def __iter__(self):
        return iter((self.alphas, self.betas, self.norm_phi0))

    def __getitem__(self, idx):
        return (self.alphas, self.betas, self.norm_phi0)[idx]

    def __len__(self):
        return 3

    def __repr__(self) -> str:
        return f"DynamicsResult(n_steps={len(self.alphas)}, norm_phi0={self.norm_phi0:.6f})"


class ContinuedFraction(Solver):
    """Continued fraction Lanczos solver for dynamical Green's functions.

    Given an excited state ``|phi0> = Op|psi0>``, applies the Lanczos
    recursion to compute the tridiagonal coefficients (alpha_j, beta_j).
    These encode the Green's function as a nested fraction:

    .. code-block:: text

        G(z) = norm^2 / (z - a0 - b1^2 / (z - a1 - b2^2 / (z - a2 - ...)))

    where ``z = omega + i*eta``. Taking the imaginary part gives the
    spectral function ``S(omega) = -(1/pi) Im G(omega + i*eta)``.

    This method is **decoupled from the eigensolver** — it only needs the
    initial perturbed state and the Hamiltonian. No ground state vector or
    energy is required as input. The Lanczos chain is seeded from ``|phi0>``
    directly.

    See Also
    --------
    spectral_function : Evaluate S(omega) from a DynamicsResult.
    DynamicsResult : The raw tridiagonal data container.
    theory: :doc:`/theory/spectral`

    Parameters
    ----------
    n_iter : int, default=100
        Number of Lanczos recursion steps.
    phi0 : Optional[np.ndarray], default=None
        Initial perturbed state vector |phi0> = Op |psi0>.

    Examples
    --------
    >>> E0, psi0 = qk.solvers.lanczos_ground_state(H)
    >>> op_psi0  = H.apply(psi0)  # or any Op applied to psi0
    >>> dyn = qk.solvers.ContinuedFraction(n_iter=200).solve(H, phi0=op_psi0)
    >>> omegas = np.linspace(-10, 10, 500)
    >>> Sw     = qk.solvers.spectral_function(dyn, omegas, eta=0.05)
    """

    def __init__(self, n_iter: int = 100, phi0: Optional[np.ndarray] = None):
        self.n_iter = int(n_iter)
        self.phi0 = phi0

    def solve(self, H: MatrixFreeHamiltonian, phi0: Optional[np.ndarray] = None) -> DynamicsResult:
        vec = phi0 if phi0 is not None else self.phi0
        if vec is None:
            raise ValueError("phi0 vector must be provided to ContinuedFraction either at initialization or in solve()")
        vec = np.ascontiguousarray(
            vec,
            dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
        )
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"continued_fraction_coeffs_{H._backend_suffix}{s_dtype}")
        alphas, betas, norm_phi0 = fn(H._cpp_obj, vec, self.n_iter)
        return DynamicsResult(alphas, betas, norm_phi0)


def continued_fraction_coeffs(
    H: MatrixFreeHamiltonian,
    phi0: np.ndarray,
    n_iter: int = 100
) -> DynamicsResult:
    """Compute continued fraction coefficients for dynamical spectral function."""
    return ContinuedFraction(n_iter=n_iter).solve(H, phi0)


def evaluate_spectral_function(
    res: DynamicsResult,
    omega: float,
    E0: float,
    eta: float = 0.1
) -> float:
    """Evaluate spectral function A(omega) from continued fraction coefficients."""
    alphas = np.ascontiguousarray(res.alphas, dtype=np.float64)
    betas  = np.ascontiguousarray(res.betas,  dtype=np.float64)
    return _cpp.evaluate_spectral_function(alphas, betas, res.norm_phi0, omega, E0, eta)


class FTLMResult:
    """Result of a Finite-Temperature Lanczos Method calculation."""
    def __init__(self, cpp_res):
        self.dimension = int(getattr(cpp_res, "dimension", 0))
        self.beta = float(getattr(cpp_res, "beta", 0.0))
        self.partition_function = float(getattr(cpp_res, "partition_function", 0.0))
        self.free_energy = float(getattr(cpp_res, "free_energy", 0.0))
        self.internal_energy = float(getattr(cpp_res, "internal_energy", 0.0))
        self.specific_heat = float(getattr(cpp_res, "specific_heat", 0.0))
        self.entropy = float(getattr(cpp_res, "entropy", 0.0))
        self.effective_samples = float(getattr(cpp_res, "effective_samples", 0.0))
        self.observable_expectations = [complex(x) for x in getattr(cpp_res, "observable_expectations", [])]
        self.observable_errors = [float(x) for x in getattr(cpp_res, "observable_errors", [])]

    def __iter__(self):
        return iter((self.beta, self.partition_function, self.internal_energy, self.specific_heat))

    def __getitem__(self, idx):
        return (self.beta, self.partition_function, self.internal_energy, self.specific_heat)[idx]

    def __len__(self):
        return 4

    def __repr__(self) -> str:
        return (
            f"FTLMResult(beta={self.beta:.4f}, Z={self.partition_function:.6e}, "
            f"E={self.internal_energy:.6f}, C={self.specific_heat:.6f})"
        )


class FTLMSweepResult:
    """Result of an FTLM multi-temperature sweep with observables."""
    def __init__(self, cpp_res):
        self.dimension = int(getattr(cpp_res, "dimension", 0))
        self.beta_grid = np.array(cpp_res.beta_grid, dtype=float)
        self.partition_functions = np.array(cpp_res.partition_functions, dtype=float)
        self.free_energies = np.array(cpp_res.free_energies, dtype=float)
        self.internal_energies = np.array(cpp_res.internal_energies, dtype=float)
        self.specific_heats = np.array(cpp_res.specific_heats, dtype=float)
        self.entropies = np.array(cpp_res.entropies, dtype=float)
        self.effective_samples = np.array(cpp_res.effective_samples, dtype=float)
        self.observable_expectations = [np.array(x, dtype=complex) for x in cpp_res.observable_expectations]
        self.observable_errors = [np.array(x, dtype=float) for x in cpp_res.observable_errors]

    def __repr__(self) -> str:
        return (
            f"FTLMSweepResult(num_betas={len(self.beta_grid)}, "
            f"num_observables={len(self.observable_expectations)})"
        )


class FTLMSamples:
    """Opaque handle holding stochastically generated Krylov subspace samples.

    Allows zero-cost re-evaluation of thermodynamic properties and observables
    across arbitrary temperature grids without repeating SpMV products.
    """
    def __init__(self, cpp_obj, backend_suffix: str, s_dtype: str):
        self._cpp_obj = cpp_obj
        self._backend_suffix = backend_suffix
        self._s_dtype = s_dtype

    def __len__(self) -> int:
        return len(self._cpp_obj)

    @property
    def num_samples(self) -> int:
        return self._cpp_obj.num_samples

    def evaluate_sweep(self, betas: Union[float, Sequence[float]]) -> FTLMSweepResult:
        """Evaluate thermodynamic sweep on temperature grid betas with zero SpMV cost."""
        fn = getattr(_cpp, f"ftlm_evaluate_sweep_{self._backend_suffix}{self._s_dtype}")
        b_list = [float(betas)] if isinstance(betas, (int, float)) else [float(b) for b in betas]
        res = fn(self._cpp_obj, b_list)
        return FTLMSweepResult(res)

    def __repr__(self) -> str:
        return f"FTLMSamples(num_samples={len(self)})"


class FTLM(Solver):
    """Finite-Temperature Lanczos Method (FTLM) thermodynamic solver.

    Computes canonical thermodynamic partition functions, specific heat, and
    observable expectations using **stochastic trace estimation**. The exact
    canonical trace ``Tr[e^{-beta H} O]`` is approximated by sampling:

    .. code-block:: text

        Tr[A] ≈ (N/R) * sum_{r=1}^{R} <r|A|r>

    where ``|r>`` are random unit vectors. Each ``<r|e^{-beta H} O|r>`` is
    evaluated using a short Lanczos expansion of ``e^{-beta H}|r>`` in a
    Krylov subspace of dimension ``n_steps``.

    **Why this works:** For a Hilbert space of dimension N, even R=50 random
    vectors (``n_random=50``) suffice for relative errors below 1% because
    the variance of the estimator scales as O(1/sqrt(R*N)). More steps
    (``n_steps``) improve the accuracy of the Krylov approximation of the
    matrix exponential.

    **Decoupled workflow:** ``mode='cached'`` runs Stage 1 (expensive SpMV)
    once and returns an :class:`FTLMSamples` handle. You can then call
    ``samples.evaluate_sweep(betas)`` to sweep over any temperature grid
    with zero additional matrix-vector products. This is crucial for
    computing thermodynamic curves.

    See Also
    --------
    FTLMSamples : Cached sample handle for zero-cost temperature sweeps.
    theory: :doc:`/theory/spectral`

    Parameters
    ----------
    beta : float, default=1.0
        Single-point inverse temperature beta = 1 / (k_B * T).
    n_random : int, default=50
        Number of stochastic random starting vectors for trace averaging.
    n_steps : int, default=100
        Krylov subspace expansion depth per random vector.
    seed : int, default=42
        Deterministic pseudo-random number generator seed.
    mode : str, default="streamed"
        Execution mode: ``'streamed'`` (memory-bounded, single temperature point)
        or ``'cached'`` (stores samples for zero-SpMV temperature sweeps).

    Examples
    --------
    >>> # Single temperature point
    >>> result = qk.solvers.FTLM(beta=2.0, n_random=100).solve(H)
    >>> print(result.energy, result.specific_heat)

    >>> # Efficient temperature sweep (Stage 1 runs once)
    >>> samples = qk.solvers.FTLM(mode="cached").sample(H)
    >>> betas   = np.linspace(0.1, 10.0, 100)
    >>> sweep   = samples.evaluate_sweep(betas)
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_random: int = 50,
        n_steps: int = 100,
        seed: int = 42,
        mode: str = "streamed"
    ):
        self.beta = float(beta)
        self.n_random = int(n_random)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.mode = mode.lower()
        if self.mode not in ("streamed", "cached"):
            raise ValueError(f"Unknown FTLM mode '{mode}', expected 'streamed' or 'cached'")

    def solve(
        self,
        H: MatrixFreeHamiltonian,
        betas: Optional[Sequence[float]] = None,
        observables: Optional[Sequence[MatrixFreeHamiltonian]] = None,
        mode: Optional[str] = None
    ) -> Union[FTLMResult, FTLMSweepResult]:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        workflow_mode = (mode or self.mode).lower()
        if betas is not None or observables:
            b_list = [float(b) for b in betas] if betas is not None else [self.beta]
            obs_cpp = [obs._cpp_obj for obs in (observables or [])]
            if workflow_mode == "cached":
                fn = getattr(_cpp, f"ftlm_sweep_{H._backend_suffix}{s_dtype}")
            else:
                fn = getattr(_cpp, f"ftlm_sweep_streamed_{H._backend_suffix}{s_dtype}")
            res = fn(H._cpp_obj, b_list, obs_cpp, self.n_random, self.n_steps, self.seed)
            return FTLMSweepResult(res)
        else:
            fn = getattr(_cpp, f"ftlm_{H._backend_suffix}{s_dtype}")
            res = fn(H._cpp_obj, self.beta, self.n_random, self.n_steps)
            return FTLMResult(res)

    def sample(
        self,
        H: MatrixFreeHamiltonian,
        observables: Optional[Sequence[MatrixFreeHamiltonian]] = None
    ) -> FTLMSamples:
        """Stage 1: Generate Krylov subspace samples and project observables."""
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"ftlm_sample_{H._backend_suffix}{s_dtype}")
        obs_cpp = [obs._cpp_obj for obs in (observables or [])]
        cpp_samples = fn(H._cpp_obj, obs_cpp, self.n_random, self.n_steps, self.seed)
        return FTLMSamples(cpp_samples, H._backend_suffix, s_dtype)

    def evaluate_sweep(
        self,
        samples: FTLMSamples,
        betas: Union[float, Sequence[float]]
    ) -> FTLMSweepResult:
        """Stage 2: Evaluate thermodynamic sweep on temperature grid betas with zero SpMV cost."""
        return samples.evaluate_sweep(betas)


def ftlm(
    H: MatrixFreeHamiltonian,
    beta: float = 1.0,
    n_random: int = 50,
    n_steps: int = 100,
    betas: Optional[Sequence[float]] = None,
    observables: Optional[Sequence[MatrixFreeHamiltonian]] = None,
    seed: int = 42,
    mode: str = "streamed"
) -> Union[FTLMResult, FTLMSweepResult]:
    """Compute finite-temperature thermodynamic observables using FTLM."""
    return FTLM(beta=beta, n_random=n_random, n_steps=n_steps, seed=seed, mode=mode).solve(
        H, betas=betas, observables=observables
    )


def ftlm_sweep_streamed(
    H: MatrixFreeHamiltonian,
    betas: Sequence[float],
    observables: Optional[Sequence[MatrixFreeHamiltonian]] = None,
    n_random: int = 50,
    n_steps: int = 100,
    seed: int = 42
) -> FTLMSweepResult:
    """Memory-bounded two-pass streamed FTLM sweep."""
    res = FTLM(n_random=n_random, n_steps=n_steps, seed=seed, mode="streamed").solve(
        H, betas=betas, observables=observables
    )
    if isinstance(res, FTLMSweepResult):
        return res
    raise TypeError(f"Expected FTLMSweepResult, got {type(res)}")


def ftlm_sample(
    H: MatrixFreeHamiltonian,
    observables: Optional[Sequence[MatrixFreeHamiltonian]] = None,
    n_random: int = 50,
    n_steps: int = 100,
    seed: int = 42
) -> FTLMSamples:
    """Stage 1: Generate Krylov subspace samples and project observables."""
    return FTLM(n_random=n_random, n_steps=n_steps, seed=seed).sample(H, observables=observables)


def ftlm_evaluate_sweep(
    samples: FTLMSamples,
    betas: Union[float, Sequence[float]]
) -> FTLMSweepResult:
    """Stage 2: Evaluate thermodynamic sweep on temperature grid betas with zero SpMV cost."""
    return samples.evaluate_sweep(betas)


class RealTimeResult:
    """Result of real-time pure state quantum evolution."""
    def __init__(self, cpp_res):
        self.time_grid = np.array(cpp_res.time_grid, dtype=float)
        self.survival_probabilities = np.array(cpp_res.survival_probabilities, dtype=complex)
        self.observable_expectations = [np.array(x, dtype=complex) for x in cpp_res.observable_expectations]

    def __repr__(self) -> str:
        return f"RealTimeResult(num_times={len(self.time_grid)}, num_observables={len(self.observable_expectations)})"


class TimeEvolve(Solver):
    """Pure-state real-time quantum dynamics solver: |psi(t)> = exp(-i * H * t) |psi(0)>.

    Computes time evolution by approximating the matrix exponential via the
    **Arnoldi-Krylov** method. Rather than exponentiating the full N×N operator,
    the propagator is projected into a tiny Krylov subspace of dimension
    ``n_steps``:

    .. code-block:: text

        exp(-i H dt) |v> ≈ V_m * exp(-i H_m dt) * e_1

    where V_m is the Krylov basis (m << N), H_m is the m×m projected
    Hamiltonian, and the small exponential is computed exactly via LAPACK.
    This gives **spectral accuracy** in m with no truncation in the physical
    basis.

    The solver marches through the user-supplied ``time_grid`` one step at a
    time, computing ``<O>_t`` for each observable at every time point.

    .. note::
        ``n_steps=30`` is sufficient for typical Heisenberg-scale time
        evolutions. Increase it when ``max(time_grid)`` is large or when
        the Hamiltonian bandwidth is large.

    See Also
    --------
    FTLMDynamics : Finite-temperature dynamical correlators.
    theory: :doc:`/theory/spectral`

    Parameters
    ----------
    time_grid : Sequence[float]
        Sequence of time points t >= 0 at which to evaluate |psi(t)> and observables.
    n_steps : int, default=30
        Krylov subspace dimension per time step. Higher = more accurate but slower.

    Examples
    --------
    >>> import numpy as np
    >>> psi0  = np.zeros(H.dimension, dtype=np.complex128)
    >>> psi0[0] = 1.0  # Neel state
    >>> times  = np.linspace(0, 5.0, 200)
    >>> result = qk.solvers.TimeEvolve(time_grid=times).solve(H, psi0)
    >>> # result.survival_probabilities[t] = |<psi0|psi(t)>|^2
    """
    def __init__(self, time_grid: Sequence[float], n_steps: int = 30):
        self.time_grid = [float(t) for t in time_grid]
        self.n_steps = int(n_steps)

    def solve(
        self,
        H: MatrixFreeHamiltonian,
        psi0: np.ndarray,
        observables: Optional[Sequence[MatrixFreeHamiltonian]] = None
    ) -> RealTimeResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        vec = np.ascontiguousarray(
            psi0,
            dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
        )
        obs_cpp = [obs._cpp_obj for obs in (observables or [])]
        fn = getattr(_cpp, f"time_evolve_{H._backend_suffix}{s_dtype}")
        res = fn(H._cpp_obj, vec, self.time_grid, obs_cpp, self.n_steps)
        return RealTimeResult(res)


def time_evolve(
    H: MatrixFreeHamiltonian,
    psi0: np.ndarray,
    time_grid: Sequence[float],
    observables: Optional[Sequence[MatrixFreeHamiltonian]] = None,
    n_steps: int = 30
) -> RealTimeResult:
    """Compute pure-state real-time evolution: |psi(t)> = exp(-i*H*t) |psi(0)>."""
    return TimeEvolve(time_grid=time_grid, n_steps=n_steps).solve(H, psi0=psi0, observables=observables)


class FTLMDynamicsResult:
    """Result of finite-temperature dynamical correlator calculation: C_AB(t) = <A(t) B(0)>_beta."""
    def __init__(self, cpp_res):
        self.beta = float(cpp_res.beta)
        self.time_grid = np.array(cpp_res.time_grid, dtype=float)
        self.correlations = np.array(cpp_res.correlations, dtype=complex)
        self.correlation_errors = np.array(cpp_res.correlation_errors, dtype=float)

    def __repr__(self) -> str:
        return f"FTLMDynamicsResult(beta={self.beta:.4f}, num_times={len(self.time_grid)})"


class FTLMDynamics(Solver):
    """Finite-temperature real-time dynamical correlator solver: C_AB(t) = <A(t) B(0)>_beta.

    Uses a two-sided Krylov expansion to compute the finite-temperature
    time correlator C_AB(t) = (1/Z) Tr[e^{-beta H} A(t) B(0)] where
    A(t) = e^{iHt} A e^{-iHt}. This is evaluated stochastically:
    for each random vector |r>, a left Krylov chain propagates e^{-beta H/2}|r>
    and a right chain propagates e^{-iHt} B|r>, then their overlap is computed.
    The stochastic average over ``n_random`` such pairs converges to the
    true thermal expectation via the random trace formula.

    See Also
    --------
    FTLM : Single-observable finite-temperature solver.
    theory: :doc:`/theory/spectral`

    Parameters
    ----------
    beta : float, default=1.0
        Inverse temperature beta = 1 / (k_B * T).
    time_grid : Sequence[float], default=(0.0,)
        Evolution time grid points.
    n_random : int, default=50
        Number of stochastic random vectors.
    n_steps : int, default=100
        Lanczos expansion depth per sample.
    seed : int, default=42
        Random seed.
    """
    def __init__(
        self,
        beta: float = 1.0,
        time_grid: Sequence[float] = (0.0,),
        n_random: int = 50,
        n_steps: int = 100,
        seed: int = 42
    ):
        self.beta = float(beta)
        self.time_grid = [float(t) for t in time_grid]
        self.n_random = int(n_random)
        self.n_steps = int(n_steps)
        self.seed = int(seed)

    def solve(
        self,
        H: MatrixFreeHamiltonian,
        A: MatrixFreeHamiltonian,
        B: MatrixFreeHamiltonian
    ) -> FTLMDynamicsResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"ftlm_dynamics_{H._backend_suffix}{s_dtype}")
        res = fn(H._cpp_obj, self.beta, A._cpp_obj, B._cpp_obj, self.time_grid, self.n_random, self.n_steps, self.seed)
        return FTLMDynamicsResult(res)


def ftlm_dynamics(
    H: MatrixFreeHamiltonian,
    A: MatrixFreeHamiltonian,
    B: MatrixFreeHamiltonian,
    beta: float = 1.0,
    time_grid: Sequence[float] = (0.0,),
    n_random: int = 50,
    n_steps: int = 100,
    seed: int = 42
) -> FTLMDynamicsResult:
    """Compute finite-temperature dynamical correlator C_AB(t) = <A(t) B(0)>_beta."""
    return FTLMDynamics(
        beta=beta, time_grid=time_grid, n_random=n_random, n_steps=n_steps, seed=seed
    ).solve(H, A, B)



@dataclass
class CorrectionVectorResult:
    """Result of a correction vector calculation.

    Attributes
    ----------
    correction_vector : np.ndarray
        The computed correction vector.
    spectral_function : float
        The computed spectral function value S(omega).
    iterations : int
        Number of conjugate gradient iterations.
    converged : bool
        Whether the solver converged within tolerance.
    """
    correction_vector: np.ndarray
    spectral_function: float
    iterations: int
    converged: bool

    def __iter__(self):
        return iter((self.correction_vector, self.spectral_function, self.iterations, self.converged))

    def __getitem__(self, idx):
        return (self.correction_vector, self.spectral_function, self.iterations, self.converged)[idx]

    def __len__(self):
        return 4

    def __repr__(self) -> str:
        return (
            f"CorrectionVectorResult(spectral_function={self.spectral_function:.10e}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


class CorrectionVector(Solver):
    """Correction Vector dynamical response solver using Conjugate Gradient.

    Solves the inhomogeneous linear system ((H - E0 - omega)^2 + eta^2) |Y> = eta * Op |psi0>
    and computes spectral response S(omega) = (1/pi) * Re<Op_psi0 | Y>.
    Solves the inhomogeneous linear system

    .. code-block:: text

        ((H - E0 - omega)^2 + eta^2) |Y> = eta * |phi>

    where ``|phi> = Op|psi0>`` using the Conjugate Gradient (CG) method.
    The operator on the left is Hermitian positive definite for any ``eta > 0``,
    guaranteeing CG convergence. The spectral function is then

    .. code-block:: text

        S(omega) = (1/pi) * Re<phi|Y>

    This method gives **exact** (within CG tolerance) spectral weights at any
    frequency, unlike the continued fraction method which requires analytic
    continuation. The cost scales as O(N * max_iter) SpMV operations.

    See Also
    --------
    ContinuedFraction : Cheaper alternative when many omega points are needed.
    theory: :doc:`/theory/spectral`

    Parameters
    ----------
    e0 : float
        Ground state energy E0.
    omega : float
        Probe frequency / energy transfer.
    eta : float, default=0.1
        Lorentzian broadening factor.
    max_iter : int, default=500
        Maximum CG iterations.
    tol : float, default=1e-8
        CG residual convergence tolerance.
    op_psi0 : Optional[np.ndarray], default=None
        Excited state vector Op |psi0>.
    maxiter : Optional[int], default=None
        Alias for max_iter.
    """

    def __init__(
        self,
        e0: float,
        omega: float,
        eta: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-8,
        op_psi0: Optional[np.ndarray] = None,
        maxiter: Optional[int] = None,
    ):
        if maxiter is not None:
            max_iter = maxiter
        self.e0 = float(e0)
        self.omega = float(omega)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.op_psi0 = op_psi0

    def solve(self, H: MatrixFreeHamiltonian, op_psi0: Optional[np.ndarray] = None) -> CorrectionVectorResult:
        vec = op_psi0 if op_psi0 is not None else self.op_psi0
        if vec is None:
            raise ValueError("op_psi0 must be provided to CorrectionVector either at initialization or in solve()")
        vec = np.ascontiguousarray(
            vec,
            dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
        )
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"correction_vector_spectral_{H._backend_suffix}{s_dtype}")
        corr_vec, spec_fn, iters, conv = fn(
            H._cpp_obj, vec, float(self.e0), float(self.omega), float(self.eta), int(self.max_iter), float(self.tol)
        )
        return CorrectionVectorResult(
            correction_vector=corr_vec,
            spectral_function=float(spec_fn),
            iterations=int(iters),
            converged=bool(conv)
        )


def correction_vector(
    H: MatrixFreeHamiltonian,
    op_psi0: np.ndarray,
    E0: float,
    omega: float,
    eta: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8
) -> CorrectionVectorResult:
    """Compute correction vector and spectral function using conjugate gradient."""
    return CorrectionVector(e0=E0, omega=omega, eta=eta, max_iter=max_iter, tol=tol).solve(H, op_psi0)


correction_vector_spectral = correction_vector

