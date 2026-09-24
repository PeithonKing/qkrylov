from typing import Any
import numpy as np
from . import _qkrylov_cpp as _cpp
from .operators import OpSum

class Heisenberg1D(OpSum):
    def __init__(self, L: int, J: float = 1.0, Jz: float = 1.0, pbc: bool = False, dtype: Any = np.float64):
        self.dtype = dtype
        self._cpp_obj = getattr(_cpp.models, "Heisenberg1D")(L, float(J), float(Jz), bool(pbc))

class TransverseFieldIsing1D(OpSum):
    def __init__(self, L: int, J: float = 1.0, h: float = 1.0, pbc: bool = False, dtype: Any = np.float64):
        self.dtype = dtype
        self._cpp_obj = getattr(_cpp.models, "TransverseFieldIsing1D")(L, float(J), float(h), bool(pbc))

