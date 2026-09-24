import re

with open("bindings/python/qkrylov/basis.py", "r") as f:
    code = f.read()

# Replace dtype
code = code.replace("dtype: Any = np.float32", "dtype: Any = np.float64")
code = code.replace("dtype = np.float32", "dtype = np.float64")

# Helper to fix lists
list_fix_str = """
def _to_list(val, type_cast):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [type_cast(x) for x in val]
    return [type_cast(val)]
"""

code = "from typing import Union, List\n" + list_fix_str + code

# SpinHalfBasis
shb_old = """    def __init__(self, N: int, conserve_sz: bool = False, sz: Optional[float] = None, dtype: Any = np.float64):
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

        self._cpp_obj = getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec)
        self._conserve_sz = conserve_sz
        self._sz = sz"""

shb_new = """    def __init__(self, N: int, sz: Optional[Union[float, List[float]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if sz is not None:
            sec.sz = _to_list(sz, float)
            
        self._cpp_obj = getattr(_cpp, "SpinHalfBasis")(N, sec) if hasattr(_cpp, "SpinHalfBasis") else getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec)
        self._conserve_sz = (sz is not None)
        self._sz = sz"""

code = code.replace(shb_old, shb_new)

# SpinSBasis
ssb_old = """    def __init__(
        self,
        N: int,
        S: float = 0.5,
        conserve_sz: bool = False,
        sz: Optional[float] = None,
        dtype: Any = np.float64
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
        self._sz = sz"""

ssb_new = """    def __init__(
        self,
        N: int,
        S: float = 0.5,
        sz: Optional[Union[float, List[float]]] = None,
        dtype: Any = np.float64
    ):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if sz is not None:
            sec.sz = _to_list(sz, float)

        self._cpp_obj = getattr(_cpp, "SpinSBasis")(N, float(S), sec) if hasattr(_cpp, "SpinSBasis") else getattr(_cpp, f"SpinSBasis{suffix}")(N, float(S), sec)
        self._S = float(S)
        self._conserve_sz = (sz is not None)
        self._sz = sz"""

code = code.replace(ssb_old, ssb_new)

# FermionBasis
fb_old = """    def __init__(self, N: int, conserve_n: bool = False, n: int = 0, dtype: Any = np.float64):
        if n != 0:
            conserve_n = True
        
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_n:
            sec.use_n = True
            sec.n = n

        self._cpp_obj = getattr(_cpp, f"FermionBasis{suffix}")(N, sec)
        self._conserve_n = conserve_n
        self._n = n"""

fb_new = """    def __init__(self, N: int, n: Optional[Union[int, List[int]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if n is not None:
            sec.n = _to_list(n, int)

        self._cpp_obj = getattr(_cpp, "FermionBasis")(N, sec) if hasattr(_cpp, "FermionBasis") else getattr(_cpp, f"FermionBasis{suffix}")(N, sec)
        self._conserve_n = (n is not None)
        self._n = n"""
code = code.replace(fb_old, fb_new)

# HubbardBasis
hb_old = """    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype: Any = np.float64):
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
        self._ndn = ndn"""

hb_new = """    def __init__(self, N: int, nup: Optional[Union[int, List[int]]] = None, 
                 ndn: Optional[Union[int, List[int]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if nup is not None:
            sec.nup = _to_list(nup, int)
        if ndn is not None:
            sec.ndn = _to_list(ndn, int)

        self._cpp_obj = getattr(_cpp, "HubbardBasis")(N, sec) if hasattr(_cpp, "HubbardBasis") else getattr(_cpp, f"HubbardBasis{suffix}")(N, sec)
        self._conserve_nup = (nup is not None)
        self._conserve_ndn = (ndn is not None)
        self._nup = nup
        self._ndn = ndn"""

code = code.replace(hb_old, hb_new)

# TJBasis
tjb_old = """    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype: Any = np.float64):
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
        self._ndn = ndn"""
        
tjb_new = """    def __init__(self, N: int, nup: Optional[Union[int, List[int]]] = None, 
                 ndn: Optional[Union[int, List[int]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if nup is not None:
            sec.nup = _to_list(nup, int)
        if ndn is not None:
            sec.ndn = _to_list(ndn, int)

        self._cpp_obj = getattr(_cpp, "TJBasis")(N, sec) if hasattr(_cpp, "TJBasis") else getattr(_cpp, f"TJBasis{suffix}")(N, sec)
        self._conserve_nup = (nup is not None)
        self._conserve_ndn = (ndn is not None)
        self._nup = nup
        self._ndn = ndn"""

code = code.replace(tjb_old, tjb_new)

with open("bindings/python/qkrylov/basis.py", "w") as f:
    f.write(code)
