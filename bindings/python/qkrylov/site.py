from . import _qkrylov_cpp as _cpp

class Site:
    """Base class for all local site physics."""
    
    def __init__(self):
        self._cpp_obj = None

class SpinHalfSite(Site):
    """Local site physics for Spin-1/2 systems.
    
    Supports operators: 'Sz', 'Sp', 'Sm', 'Sx', 'Sy'
    """
    def __init__(self):
        self._cpp_obj = _cpp.SpinHalfSite()

    def __repr__(self) -> str:
        return "SpinHalfSite()"

class FermionSite(Site):
    """Local site physics for spinless fermions."""
    def __init__(self):
        self._cpp_obj = _cpp.FermionSite()
        
    def __repr__(self) -> str:
        return "FermionSite()"

class HubbardSite(Site):
    """Local site physics for interacting electrons (spin-1/2 fermions)."""
    def __init__(self):
        self._cpp_obj = _cpp.HubbardSite()

    def __repr__(self) -> str:
        return "HubbardSite()"

class TJSite(Site):
    """Local site physics for t-J model (doped antiferromagnets)."""
    def __init__(self):
        self._cpp_obj = _cpp.TJSite()

    def __repr__(self) -> str:
        return "TJSite()"
