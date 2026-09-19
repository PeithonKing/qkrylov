import enum
import numpy as np
from typing import List, Tuple, Union, Sequence, Any
from . import _qkrylov_cpp as _cpp

class Op(str, enum.Enum):
    """Enumeration of quantum operator symbols to prevent string typos.

    Each symbol acts locally on a single site:

    - **Spin operators:** ``Sz`` (diagonal, returns ±1/2), ``Sp``/``Sm``
      (raising/lowering, flip one spin and multiply by sqrt(S(S+1)-m(m±1))),
      ``Sx = (Sp+Sm)/2``, ``Sy = -i(Sp-Sm)/2``.
    - **Fermionic operators:** ``CdagUp``/``CUp`` (creation/annihilation, spin-up),
      ``CdagDn``/``CDn`` (spin-down), with Jordan-Wigner phase tracked
      automatically by the C++ kernel.
    - **Number operators:** ``Nup``, ``Ndn``, ``Nupdn`` (double occupancy).
    - **Bosonic operators:** ``Bdag``/``B`` (creation/annihilation), ``N`` (number).
    """
    Sz = "Sz"
    Sp = "Sp"
    Sm = "Sm"
    Sx = "Sx"
    Sy = "Sy"
    CdagUp = "CdagUp"
    CUp = "CUp"
    CdagDn = "CdagDn"
    CDn = "CDn"
    Nup = "Nup"
    Ndn = "Ndn"
    Nupdn = "Nupdn"
    Bdag = "Bdag"
    B = "B"
    N = "N"


class LocalOpExpr:
    """Symbolic representation of a local quantum operator acting at a specific site.

    ``LocalOpExpr`` nodes are lightweight Python objects that defer all
    evaluation to C++. Multiplying two ``LocalOpExpr`` instances returns a
    ``TermExpr`` — a symbolic product. Adding ``TermExpr`` instances returns
    an ``OpSumExpr``. The full expression tree is then compiled into a C++
    operator sum list when passed to :class:`~qkrylov.operators.OpSum`.

    Parameters
    ----------
    name : str
        Operator symbol (e.g., 'Sz', 'Sp', 'CdagUp').
    site : int
        Zero-indexed site index.
    """
    def __init__(self, name: str, site: int):
        self.name = name
        self.site = site

    def __mul__(self, other):
        if isinstance(other, LocalOpExpr):
            return TermExpr(1.0, [self, other])
        elif isinstance(other, (int, float, complex)):
            return TermExpr(complex(other), [self])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return TermExpr(complex(other), [self])
        return NotImplemented


class TermExpr:
    """Symbolic representation of a scalar-weighted product of local operators.

    The term ``TermExpr(coeff, [op1, op2, ...])`` preserves the user's
    operator ordering exactly as written. The C++ backend handles the physical
    ordering (normal order, Jordan-Wigner phase) internally during the SpMV.
    Users should not reorder operators manually; the backend is correct for
    any ordering of creation/annihilation operators on distinct sites.

    Parameters
    ----------
    coeff : complex
        Scalar coupling coefficient.
    ops : List[LocalOpExpr]
        List of local operator factors acting on specific sites.
    """
    def __init__(self, coeff: complex, ops: List[LocalOpExpr]):
        self.coeff = coeff
        self.ops = ops

    def __mul__(self, other):
        if isinstance(other, LocalOpExpr):
            return TermExpr(self.coeff, self.ops + [other])
        elif isinstance(other, TermExpr):
            return TermExpr(self.coeff * other.coeff, self.ops + other.ops)
        elif isinstance(other, (int, float, complex)):
            return TermExpr(self.coeff * complex(other), self.ops)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return TermExpr(self.coeff * complex(other), self.ops)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, TermExpr):
            return OpSumExpr([self, other])
        elif isinstance(other, OpSumExpr):
            return OpSumExpr([self] + other.terms)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TermExpr):
            neg_other = TermExpr(-other.coeff, other.ops)
            return OpSumExpr([self, neg_other])
        return NotImplemented


class OpSumExpr:
    """Symbolic sum of many-body interaction terms.

    ``OpSumExpr`` accumulates terms as a flat list. Duplicate terms (same
    sites, same operators, different coefficients) are **not** automatically
    consolidated at the Python level — this happens inside the C++ ``OpSum``
    builder when the expression is compiled. This makes Python-level construction
    O(number of terms) with no quadratic merging overhead.

    Parameters
    ----------
    terms : List[TermExpr]
        Sequence of interaction terms in the sum.
    """
    def __init__(self, terms: List[TermExpr]):
        self.terms = terms

    def __add__(self, other):
        if isinstance(other, TermExpr):
            return OpSumExpr(self.terms + [other])
        elif isinstance(other, OpSumExpr):
            return OpSumExpr(self.terms + other.terms)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return OpSumExpr([t * other for t in self.terms])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return OpSumExpr([t * other for t in self.terms])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TermExpr):
            neg_other = TermExpr(-other.coeff, other.ops)
            return OpSumExpr(self.terms + [neg_other])
        elif isinstance(other, OpSumExpr):
            neg_terms = [TermExpr(-t.coeff, t.ops) for t in other.terms]
            return OpSumExpr(self.terms + neg_terms)
        return NotImplemented


# Helper functions to instantiate LocalOpExpr easily
def Sz(i: int) -> LocalOpExpr: return LocalOpExpr("Sz", i)
def Sp(i: int) -> LocalOpExpr: return LocalOpExpr("Sp", i)
def Sm(i: int) -> LocalOpExpr: return LocalOpExpr("Sm", i)
def Sx(i: int) -> LocalOpExpr: return LocalOpExpr("Sx", i)
def Sy(i: int) -> LocalOpExpr: return LocalOpExpr("Sy", i)

def CdagUp(i: int) -> LocalOpExpr: return LocalOpExpr("CdagUp", i)
def CUp(i: int) -> LocalOpExpr: return LocalOpExpr("CUp", i)
def CdagDn(i: int) -> LocalOpExpr: return LocalOpExpr("CdagDn", i)
def CDn(i: int) -> LocalOpExpr: return LocalOpExpr("CDn", i)

def Nup(i: int) -> LocalOpExpr: return LocalOpExpr("Nup", i)
def Ndn(i: int) -> LocalOpExpr: return LocalOpExpr("Ndn", i)
def Nupdn(i: int) -> LocalOpExpr: return LocalOpExpr("Nupdn", i)

def Bdag(i: int) -> LocalOpExpr: return LocalOpExpr("Bdag", i)
def B(i: int) -> LocalOpExpr: return LocalOpExpr("B", i)
def N(i: int) -> LocalOpExpr: return LocalOpExpr("N", i)


class OpSum:
    """High-performance symbolic expression builder for Hamiltonian interaction terms.

    Binds Python operator definitions into efficient C++ interaction lists.
    When you call ``ops += term``, the Python ``TermExpr`` is immediately
    serialized into the C++ ``OpSum`` object as a packed struct containing
    (coefficient, [(operator_enum, site_index), ...]). During Hamiltonian
    construction, the C++ backend iterates over all basis states, evaluates
    each term's action, and builds the CSR connectivity table in one pass.
    This one-time O(N * terms) cost eliminates any per-SpMV interpretation.

    Parameters
    ----------
    dtype : Any, default=np.float32
        Floating point precision for coupling coefficients.

    Attributes
    ----------
    dtype : Any
        Floating point precision.
    size : int
        Number of interaction terms in the sum.

    Methods
    -------
    add_term(coeff, *ops)
        Add a multi-site interaction term with scalar coefficient.
    clear()
        Remove all terms from the builder.
    """
    dtype: Any
    _cpp_obj: Any

    def __init__(self, dtype: Any = np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self.dtype = dtype
        self._cpp_obj = getattr(_cpp, f'OpSum{suffix}')()

    def add_term(self, coeff: complex, *ops: Union[str, Op, int]):
        if len(ops) % 2 != 0:
            raise ValueError("Operators must be provided in (name, site) pairs.")
        processed_ops: List[Union[str, int]] = []
        for i in range(0, len(ops), 2):
            op_item = ops[i]
            op_name = op_item.value if isinstance(op_item, Op) else str(op_item)
            site = int(ops[i+1])
            processed_ops.extend([op_name, site])
        tup = (coeff,) + tuple(processed_ops)
        self._cpp_obj.__iadd__(tup)

    def __iadd__(self, term: Union[Tuple[Any, ...], TermExpr, OpSumExpr]) -> "OpSum":
        if isinstance(term, tuple):
            if len(term) < 3 or len(term) % 2 == 0:
                raise ValueError("Term must be a tuple of (coeff, op1, site1, ...)")
            self.add_term(term[0], *term[1:])
        elif isinstance(term, TermExpr):
            ops_flat: List[Union[str, Op, int]] = []
            for op in term.ops:
                ops_flat.extend([op.name, op.site])
            self.add_term(term.coeff, *ops_flat)
        elif isinstance(term, OpSumExpr):
            for t in term.terms:
                self.__iadd__(t)
        else:
            raise ValueError("Unsupported type for += on OpSum")
        return self

    def clear(self):
        self._cpp_obj.clear()

    @property
    def size(self) -> int:
        return self._cpp_obj.size()

    def __repr__(self) -> str:
        return f"OpSum(terms={self.size})"
