import re

with open("bindings/python/qkrylov/basis.py", "r") as f:
    code = f.read()

# Change all np.float32 to np.float64
code = code.replace("np.float32", "np.float64")

def listify(var_name):
    return f"[{var_name}] if isinstance({var_name}, (int, float)) else list({var_name})"

# Replace SpinHalfBasis init
spinhalf_old = """    def __init__(self, N: int, conserve_sz: bool = False, sz: Optional[float] = None, dtype: Any = np.float64):
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

spinhalf_new = """    def __init__(self, N: int, sz: Optional[Union[float, List[float]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")() if hasattr(_cpp, f"Sector{suffix}") else _cpp.Sector()
        if sz is not None:
            sec.sz = [float(x) for x in (sz if isinstance(sz, (list, tuple)) else [sz])]

        self._cpp_obj = getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec) if hasattr(_cpp, f"SpinHalfBasis{suffix}") else getattr(_cpp, "SpinHalfBasis")(N, sec)
        self._conserve_sz = (sz is not None)
        self._sz = sz"""

code = code.replace(spinhalf_old, spinhalf_new)

# Repeat for others manually via sed or just write a script
