import numpy as np
from typing import Optional, Any
from . import _qkrylov_cpp as _cpp

class Basis:
    """Base class for many-body Hilbert space bases.

    Provides total state dimension, site count, and state indexing
    abstractions backed by C++20 bit-packed implementations.
    The many-body Hilbert space grows exponentially with system size:
    for L spin-1/2 sites the unconstrained dimension is ``2^L``. At L=30
    this is already ~10^9 complex floats (>8 GB in FP64). Symmetry sectors
    (fixed S_z, particle number, etc.) compress this dramatically — the
    S_z=0 sector of an L=30 spin chain is ~C(30,15) ≈ 155 million states,
    a 7x reduction. See :doc:`/theory/exact_diag`.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Underlying precision type.

    Attributes
    ----------
    size : int
        Total dimension of the many-body basis sector.
    nsites : int
        Number of physical sites in the lattice.
    _cpp_obj : Any
        Underlying C++ Basis instance.

    Notes
    -----
    States are stored as packed integers (one integer per basis state),
    sorted lexicographically for O(log N) binary search. The C++ backend
    exposes them as a raw device pointer — Python wraps this as a zero-copy
    ``np.ndarray`` view with no data duplication.
    """
    _cpp_obj: Any
    
    def __init__(self):
        self._cpp_obj = None

    @property
    def size(self) -> int:
        """The total dimension of this basis."""
        return self._cpp_obj.size()

    @property
    def nsites(self) -> int:
        """The number of sites in this basis."""
        return self._cpp_obj.nsites()

def _build_sector(
    dtype: Any,
    conserve_sz: bool = False, sz: float = 0,
    conserve_nup: bool = False, nup: int = 0,
    conserve_ndn: bool = False, ndn: int = 0,
    conserve_n: bool = False, n: int = 0,
    conserve_nb: bool = False, nb: int = 0
) -> Any:
    """Construct a C++ Sector quantum number constraint object.

    The ``Sector`` object encodes conserved quantum numbers for projection
    onto symmetry subspaces. For spin models this is the total S_z; for
    fermion models it is the total particle number (or spin-up/spin-down
    counts separately). Only basis states satisfying all sector constraints
    are included, reducing Hilbert space dimension and enabling block-diagonal
    Hamiltonian structure.

    Parameters
    ----------
    dtype : Any
        Floating point precision type (np.float32 or np.float64).
    conserve_sz : bool, default=False
        Whether to enforce total magnetization Sz conservation.
    sz : float, default=0
        Target total Sz eigenvalue.
    conserve_nup : bool, default=False
        Whether to conserve spin-up particle count.
    nup : int, default=0
        Target spin-up particle count.
    conserve_ndn : bool, default=False
        Whether to conserve spin-down particle count.
    ndn : int, default=0
        Target spin-down particle count.
    conserve_n : bool, default=False
        Whether to conserve total particle count.
    n : int, default=0
        Target total particle number.
    conserve_nb : bool, default=False
        Whether to conserve total boson number.
    nb : int, default=0
        Target boson number.

    Returns
    -------
    Any
        Constructed C++ Sector object.
    """
    suffix = "_FP64" if dtype == np.float64 else "_FP32"
    sec = getattr(_cpp, f"Sector{suffix}")()
    if conserve_sz or sz != 0:
        sec.use_sz = True
        sec.sz2 = int(round(sz * 2))  # C++ uses 2*Sz
    if conserve_nup or nup != 0:
        sec.use_nup = True
        sec.nup = nup
    if conserve_ndn or ndn != 0:
        sec.use_ndn = True
        sec.ndn = ndn
    if conserve_n or n != 0:
        sec.use_n = True
        sec.n = n
    if conserve_nb or nb != 0:
        sec.use_nb = True
        sec.nb = nb
    return sec


class SpinHalfBasis(Basis):
    """Many-body basis for Spin-1/2 lattice systems.

    Encodes states as bitstrings with optional total Sz abelian symmetry projection.
    States are 16-bit integers where bit ``i`` = 1 means site ``i`` has
    spin-up. The Sz-conserved sector with ``n_up`` up-spins is enumerated
    as all L-bit integers with exactly ``n_up`` bits set — a combinatorial
    set of size C(L, n_up). The C++ backend generates these via the
    Gosper's hack bit-twiddling algorithm in O(C(L,n_up)) time.

    Parameters
    ----------
    N : int
        Total number of spin sites.
    conserve_sz : bool, default=False
        Enable total Sz conservation.
    sz : Optional[float], default=None
        Target Sz sector (0, +/-0.5, +/-1, etc.).
    dtype : Any, default=np.float32
        Precision type (np.float32 or np.float64).

    Notes
    -----
    Given a state integer, its index in the basis is computed via a rank
    query using a pre-computed prefix-sum table over the sorted state list.
    Binary search locates the state in O(log N); the prefix table converts
    this to a rank in O(1). Total query time: O(log N).
    """
    
    def __init__(self, N: int, conserve_sz: bool = False, sz: Optional[float] = None, dtype: Any = np.float32):
        # If sz is explicitly provided, it implies conservation
        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0  # default sector when conserve_sz=True but no sz given
        else:
            sz = 0  # no conservation, sz value doesn't matter

        # C++ Sector takes sz2 (which is 2 * sz)
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(2 * sz)

        self._cpp_obj = getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec)
        self._conserve_sz = conserve_sz
        self._sz = sz

    def __repr__(self) -> str:
        sec_str = f", sz={self._sz}" if self._conserve_sz else ""
        return f"SpinHalfBasis(N={self.nsites}, dim={self.size}{sec_str})"


class SpinSBasis(Basis):
    """Many-body basis for arbitrary spin S systems.

    Represents systems with local dimension 2S + 1 using mixed-radix integers.
    For spin-S sites with ``d = 2S+1`` local states, each site's state
    is an integer in [0, d). A basis state is a mixed-radix number in base d
    across L digits. The Sz sector constrains the sum of local m_z values.
    Enumeration uses a recursive combinatorial generation algorithm.

    Parameters
    ----------
    N : int
        Total number of lattice sites.
    S : float, default=0.5
        Spin quantum number.
    conserve_sz : bool, default=False
        Enable total Sz conservation.
    sz : Optional[float], default=None
        Target total Sz value.
    dtype : Any, default=np.float32
        Precision type (np.float32 or np.float64).

    Attributes
    ----------
    spin : float
        Spin quantum number S.
    dimension_per_site : int
        Local site Hilbert space dimension 2S + 1.

    Notes
    -----
    The C++ backend stores basis states as packed integers using the
    minimum necessary bits per site (ceil(log2(d)) bits). For spin-1
    (d=3) this is 2 bits/site, allowing 32 sites per 64-bit word.
    Index lookup uses binary search on the sorted state array: O(log N).
    """
    
    def __init__(
        self,
        N: int,
        S: float = 0.5,
        conserve_sz: bool = False,
        sz: Optional[float] = None,
        dtype: Any = np.float32
    ):
        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0
        else:
            sz = 0

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(round(2 * sz))

        self._cpp_obj = getattr(_cpp, f"SpinSBasis{suffix}")(N, float(S), sec)
        self._S = float(S)
        self._conserve_sz = conserve_sz
        self._sz = sz

    @property
    def spin(self) -> float:
        """The spin quantum number S."""
        return self._cpp_obj.spin

    @property
    def dimension_per_site(self) -> int:
        """The local Hilbert space dimension 2S + 1."""
        return self._cpp_obj.dimension_per_site

    def state(self, i: int) -> int:
        """Get the basis state at index i."""
        return self._cpp_obj.state(i)

    def index(self, s: int) -> int:
        """Find the index of basis state s."""
        return self._cpp_obj.index(s)

    def contains(self, s: int) -> bool:
        """Check if basis state s is in this basis."""
        return self._cpp_obj.contains(s)

    def __repr__(self) -> str:
        sec_str = f", sz={self._sz}" if self._conserve_sz else ""
        return f"SpinSBasis(N={self.nsites}, S={self._S}, dim={self.size}{sec_str})"


class FermionBasis(Basis):
    """Many-body basis for spinless fermions with particle number conservation.

    Fermionic states are represented as bitstrings where bit ``i`` = 1
    means site ``i`` is occupied. The Jordan-Wigner transformation maps
    fermion creation operators to spin operators with non-local phase strings.
    The C++ SpMV kernel tracks the JW phase automatically during the
    matrix-vector product — no explicit JW string is stored.

    Parameters
    ----------
    N : int
        Number of fermionic lattice sites.
    conserve_n : bool, default=False
        Whether to conserve total particle number.
    n : int, default=0
        Total particle count sector.
    dtype : Any, default=np.float32
        Precision type.
    """
    def __init__(self, N: int, conserve_n: bool = False, n: int = 0, dtype: Any = np.float32):
        if n != 0:
            conserve_n = True
        
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_n:
            sec.use_n = True
            sec.n = n

        self._cpp_obj = getattr(_cpp, f"FermionBasis{suffix}")(N, sec)
        self._conserve_n = conserve_n
        self._n = n

    def __repr__(self) -> str:
        sec_str = f", n={self._n}" if self._conserve_n else ""
        return f"FermionBasis(N={self.nsites}, dim={self.size}{sec_str})"


class HubbardBasis(Basis):
    """Many-body basis for interacting spinful Fermi-Hubbard electrons.

    Represents spin-up and spin-down configurations in separate bit-registers.
    Hubbard states use two separate 32-bit registers: one for spin-up
    occupations and one for spin-down, bit-interleaved into a single 64-bit
    integer for efficient SIMD processing. Particle conservation is enforced
    separately for each spin species, enabling projection onto sectors
    with fixed (N_up, N_down) simultaneously.

    Parameters
    ----------
    N : int
        Number of spatial sites.
    conserve_nup : bool, default=False
        Conserve spin-up electron count.
    nup : int, default=0
        Target spin-up electron count.
    conserve_ndn : bool, default=False
        Conserve spin-down electron count.
    ndn : int, default=0
        Target spin-down electron count.
    dtype : Any, default=np.float32
        Precision type.
    """
    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype: Any = np.float32):
        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = getattr(_cpp, f"HubbardBasis{suffix}")(N, sec)
        self._conserve_nup = conserve_nup
        self._conserve_ndn = conserve_ndn
        self._nup = nup
        self._ndn = ndn

    def __repr__(self) -> str:
        sec_strs = []
        if self._conserve_nup: sec_strs.append(f"nup={self._nup}")
        if self._conserve_ndn: sec_strs.append(f"ndn={self._ndn}")
        sec_str = ", " + ", ".join(sec_strs) if sec_strs else ""
        return f"HubbardBasis(N={self.nsites}, dim={self.size}{sec_str})"


class TJBasis(Basis):
    """Many-body basis for the t-J model on arbitrary lattices.

    Restricts configurations to those without doubly-occupied sites.
    t-J basis states are Hubbard states with the double-occupancy
    constraint (no site can have both spin-up and spin-down). States are
    generated by filtering the Hubbard Fock space: a state is valid if
    ``(up_bits & dn_bits) == 0``. The C++ enumerator uses a precomputed
    bitmask to skip invalid states in O(C(L, N_up) * C(L - N_up, N_dn)) time.

    Parameters
    ----------
    N : int
        Number of spatial sites.
    conserve_nup : bool, default=False
        Conserve spin-up electron count.
    nup : int, default=0
        Target spin-up electron count.
    conserve_ndn : bool, default=False
        Conserve spin-down electron count.
    ndn : int, default=0
        Target spin-down electron count.
    dtype : Any, default=np.float32
        Precision type.
    """
    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype: Any = np.float32):
        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = getattr(_cpp, f"TJBasis{suffix}")(N, sec)
        self._conserve_nup = conserve_nup
        self._conserve_ndn = conserve_ndn
        self._nup = nup
        self._ndn = ndn

    def __repr__(self) -> str:
        sec_strs = []
        if self._conserve_nup: sec_strs.append(f"nup={self._nup}")
        if self._conserve_ndn: sec_strs.append(f"ndn={self._ndn}")
        sec_str = ", " + ", ".join(sec_strs) if sec_strs else ""
        return f"TJBasis(N={self.nsites}, dim={self.size}{sec_str})"


# Aliases conforming to API blueprint
SpinHalf = SpinHalfBasis
SpinS = SpinSBasis
Fermion = FermionBasis
Hubbard = HubbardBasis
TJ = TJBasis
