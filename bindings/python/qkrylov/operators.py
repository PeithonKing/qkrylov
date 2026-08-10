import enum
import numpy as np
from typing import List, Tuple, Union, Sequence
from . import _qkrylov_cpp as _cpp

class Op(str, enum.Enum):
    """Enumeration of standard quantum operators to prevent typos."""
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
    """Represents a local operator acting on a site, e.g. Sz(0)."""
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
    """Represents a term in the Hamiltonian, e.g. 1.0 * Sz(0) * Sz(1)."""
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
    """Represents a sum of terms, e.g. 1.0*Sz(0)*Sz(1) + 0.5*Sp(0)*Sm(1)."""
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
    """Symbolic expression builder for quantum interactions."""

    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self.dtype = dtype
        self._cpp_obj = getattr(_cpp, f'OpSum{suffix}')()

    def add_term(self, coeff: complex, *ops: Union[str, Op, int]):
        if len(ops) % 2 != 0:
            raise ValueError("Operators must be provided in (name, site) pairs.")
        processed_ops = []
        for i in range(0, len(ops), 2):
            op_name = ops[i].value if isinstance(ops[i], Op) else str(ops[i])
            site = int(ops[i+1])
            processed_ops.extend([op_name, site])
        tup = (coeff,) + tuple(processed_ops)
        self._cpp_obj.__iadd__(tup)

    def __iadd__(self, term: Union[Tuple, TermExpr, OpSumExpr]):
        if isinstance(term, tuple):
            if len(term) < 3 or len(term) % 2 == 0:
                raise ValueError("Term must be a tuple of (coeff, op1, site1, ...)")
            self.add_term(term[0], *term[1:])
        elif isinstance(term, TermExpr):
            ops_flat = []
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
