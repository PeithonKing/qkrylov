from typing import Any
from . import _qkrylov_cpp as _cpp

import numpy as np

class Site:
    """Base class for all local site Hilbert spaces and physical algebras.

    Defines local physical degrees of freedom and maps operator symbols to
    sparse matrix elements in the local computational basis.
    The ``Site`` object defines the **local Hilbert space dimension** ``d``
    and the **local operator matrices** (Sz, Sp, etc.) as d×d matrices.
    These are passed to the C++ OpSum builder, which uses them to evaluate
    matrix elements of local terms during basis enumeration. The Site object
    itself holds no state after construction — it is consumed by the C++ backend.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Precision type for matrix elements (np.float32 or np.float64).

    Attributes
    ----------
    dtype : Any
        Floating-point precision for local matrix representations.
    _cpp_obj : Any
        Underlying C++20 Site kernel instance.

    Notes
    -----
    Site objects come in FP32 (``dtype=np.float32``) and FP64
    (``dtype=np.float64``) variants. The coupling coefficients in ``OpSum`` are
    stored at this precision. FP32 uses AVX-512 SIMD 16-wide vectorization
    vs FP64's 8-wide — a 2x throughput advantage on modern Intel/AMD CPUs.
    On NVIDIA GPUs, FP32 is 2x faster than FP64 in pure compute, but memory
    bandwidth is the bottleneck for SpMV, so the advantage is ~1.5x in practice.
    See :doc:`/theory/precision_stability`.
    """
    _cpp_obj: Any
    dtype: Any
    
    def __init__(self, dtype: Any = np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = None
        self.dtype = dtype

class SpinHalfSite(Site):
    """Local site physics for Spin-1/2 quantum degrees of freedom.

    Represents a 2-level quantum system with local basis {|up>, |dn>}.
    Supports Pauli and ladder operators: 'Sz', 'Sp', 'Sm', 'Sx', 'Sy'.
    For a spin-1/2 site (``d=2``), the local operators are 2×2 matrices:

    .. code-block:: text

        Sz = [[+1/2,   0  ], [  0,  -1/2]]
        Sp = [[  0,    1  ], [  0,     0 ]]
        Sm = [[  0,    0  ], [  1,     0 ]]

    These satisfy the SU(2) commutation relations [Sp,Sm] = 2*Sz, [Sz,Sp] = Sp,
    [Sz,Sm] = -Sm. The Heisenberg exchange J*S·S = J*(Sz⊗Sz + (1/2)(Sp⊗Sm + Sm⊗Sp)).

    Parameters
    ----------
    dtype : Any, default=np.float32
        Floating point precision (np.float32 or np.float64).

    Notes
    -----
    Uses the standard physics convention with basis states |↑> = [1,0]^T (m=+1/2)
    and |↓> = [0,1]^T (m=-1/2). Pauli matrices sigma_i = 2*S_i so that
    Sz = sigma_z/2, Sp = sigma_+, Sm = sigma_-.
    """
    def __init__(self, dtype: Any = np.float32):
        super().__init__(dtype=dtype)
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'SpinHalfSite{suffix}')()

    def __repr__(self) -> str:
        return "SpinHalfSite()"

class SpinSSite(Site):
    """Local site physics for arbitrary spin S systems.

    Represents a (2S + 1)-level quantum system with spin quantum number S.
    Supports spin operators: 'Sz', 'Sp', 'Sm'.
    For spin-S (``d = 2S+1``), the ladder operator matrix elements follow
    from the Wigner-Eckart theorem:

    .. code-block:: text

        <m+1|S+|m> = sqrt(S(S+1) - m(m+1))
        <m-1|S-|m> = sqrt(S(S+1) - m(m-1))

    The basis states are ordered from m = +S down to m = -S. This convention
    is consistent with the Clebsch-Gordan tables used in angular momentum theory.

    Parameters
    ----------
    S : float, default=0.5
        Spin quantum number (integer or half-integer, e.g. 0.5, 1.0, 1.5).
    dtype : Any, default=np.float32
        Floating point precision (np.float32 or np.float64).

    Attributes
    ----------
    spin : float
        The total spin quantum number S.
    dimension_per_site : int
        The local Hilbert space dimension (2S + 1).

    Notes
    -----
    For large S (e.g., S=5/2, d=6), the matrix elements of S+ and S- involve
    square roots of numbers up to S(S+1)=8.75. These are computed in FP64 at
    construction time and stored as FP32/FP64 depending on ``dtype``. No
    numerical instability arises from the construction itself — the stability
    concerns are in the iterative solvers (ghost states, loss of orthogonality),
    not in the site representations.
    """
    def __init__(self, S: float = 0.5, dtype: Any = np.float32):
        super().__init__(dtype=dtype)
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'SpinSSite{suffix}')(float(S))
        self._S = float(S)
        self._dtype = dtype

    @property
    def spin(self) -> float:
        """The spin quantum number S."""
        return self._cpp_obj.spin

    @property
    def dimension_per_site(self) -> int:
        """The local Hilbert space dimension 2S + 1."""
        return self._cpp_obj.dimension_per_site

    def __repr__(self) -> str:
        return f"SpinSSite(S={self._S})"

class FermionSite(Site):
    """Local site physics for spinless fermions.

    Represents a 2-level fermionic orbital with occupation numbers n in {0, 1}.
    Supports creation ('Cdag'), annihilation ('C'), and number ('N') operators.
    Spinless fermion operators satisfy ``{c_i, c_j†} = delta_ij`` and
    ``{c_i, c_j} = 0`` (anticommutation). The Jordan-Wigner transformation maps
    these to spin operators:

    .. code-block:: text

        c_i = (prod_{j<i} sigma_z^j) * sigma_-^i

    The JW string ``prod_{j<i} sigma_z^j`` is the non-local phase factor that
    encodes fermionic statistics. qkrylov handles this automatically — you
    write ``Cdag(i)*C(j)`` and the C++ SpMV applies the correct JW phase.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Floating point precision (np.float32 or np.float64).

    Notes
    -----
    The C++ SpMV kernel tracks the Jordan-Wigner phase by counting the
    parity of occupied sites between the operator positions. For ``Cdag(i)*C(j)``
    with i < j, the phase is ``(-1)^(number of occupied sites in [i+1, j-1])``.
    This is computed via bit-population-count (popcount) on the occupation bitstring,
    which is a single CPU instruction on modern hardware.
    """
    def __init__(self, dtype: Any = np.float32):
        super().__init__(dtype=dtype)
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'FermionSite{suffix}')()
        
    def __repr__(self) -> str:
        return "FermionSite()"

class HubbardSite(Site):
    """Local site physics for interacting electrons (spin-1/2 fermions).

    Represents a 4-level electron orbital with basis {|empty>, |up>, |dn>, |up_dn>}.
    Supports operators: 'CdagUp', 'CUp', 'CdagDn', 'CDn', 'Nup', 'Ndn', 'Nupdn'.
    The Hubbard site has 4 local states: |0> (empty), |↑> (spin-up only),
    |↓> (spin-down only), |↑↓> (doubly occupied). The on-site Hubbard
    interaction is ``U * n_up * n_down``, which is diagonal — it equals U
    only for the |↑↓> state. Hopping terms ``-t * Cdag_{i,up} * C_{j,up}``
    include JW phases from all sites between i and j.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Floating point precision (np.float32 or np.float64).

    Notes
    -----
    Two bits per site: bit 0 = spin-up occupation, bit 1 = spin-down occupation.
    This is stored as two separate 32-bit registers for efficient SIMD processing.
    The double-occupancy number operator ``Nupdn`` checks ``(up_bit & dn_bit)``
    at the target site — a single bitwise AND instruction.
    """
    def __init__(self, dtype: Any = np.float32):
        super().__init__(dtype=dtype)
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'HubbardSite{suffix}')()

    def __repr__(self) -> str:
        return "HubbardSite()"

class TJSite(Site):
    """Local site physics for the t-J model with no double occupancy.

    Represents a 3-level restricted Hilbert space {|empty>, |up>, |dn>}
    governing strongly correlated doped antiferromagnets.
    The t-J model emerges from the Hubbard model in the large-U limit via
    a Schrieffer-Wolff transformation. The double-occupancy constraint
    (Gutzwiller projection) eliminates the |↑↓> state from the local Hilbert
    space. The effective Hamiltonian contains only nearest-neighbor hopping
    (with JW phase) and superexchange J*S_i·S_j terms.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Floating point precision (np.float32 or np.float64).

    Notes
    -----
    In the limit U >> t, virtual hopping through doubly-occupied states
    generates an effective antiferromagnetic exchange J = 4t^2/U between
    neighbouring sites. The t-J model directly implements this low-energy
    effective theory. The local Hilbert space dimension drops from 4 (Hubbard)
    to 3 (t-J), reducing the total Hilbert space size by ~(3/4)^L.
    """
    def __init__(self, dtype: Any = np.float32):
        super().__init__(dtype=dtype)
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'TJSite{suffix}')()

    def __repr__(self) -> str:
        return "TJSite()"


# Aliases conforming to API blueprint
SpinHalf = SpinHalfSite
SpinS = SpinSSite
Fermion = FermionSite
Hubbard = HubbardSite
TJ = TJSite
