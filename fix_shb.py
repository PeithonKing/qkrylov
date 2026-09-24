import re

with open("bindings/python/qkrylov/basis.py", "r") as f:
    code = f.read()

shb_old_pattern = r"    def __init__\(self, N: int, conserve_sz: bool = False, sz: Optional\[float\] = None, dtype: Any = np\.float64\):.*?self\._sz = sz"

shb_new = """    def __init__(self, N: int, sz: Optional[Union[float, List[float]]] = None, dtype: Any = np.float64):
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, "Sector")() if hasattr(_cpp, "Sector") else getattr(_cpp, f"Sector{suffix}")()
        
        if sz is not None:
            sec.sz = _to_list(sz, float)
            
        self._cpp_obj = getattr(_cpp, "SpinHalfBasis")(N, sec) if hasattr(_cpp, "SpinHalfBasis") else getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec)
        self._conserve_sz = (sz is not None)
        self._sz = sz"""

code = re.sub(shb_old_pattern, shb_new, code, flags=re.DOTALL)

with open("bindings/python/qkrylov/basis.py", "w") as f:
    f.write(code)
