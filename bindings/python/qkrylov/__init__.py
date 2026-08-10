"""Python interface for qkrylov.

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics,
now with a Pythonic wrapper layer.
"""

from .operators import (
    Op, OpSum,
    Sz, Sp, Sm, Sx, Sy,
    CdagUp, CUp, CdagDn, CDn,
    Nup, Ndn, Nupdn,
    Bdag, B, N
)
from .site import Site, SpinHalfSite, FermionSite, HubbardSite, TJSite
from .basis import Basis, SpinHalfBasis, FermionBasis, HubbardBasis, TJBasis
from .hamiltonian import MatrixFreeHamiltonian
from .solvers import (
    LanczosResult,
    lanczos_ground_state,
    DavidsonResult,
    davidson_lowest,
    DynamicsResult,
    continued_fraction_coeffs,
    evaluate_spectral_function,
    FTLMResult,
    ftlm,
)

from ._qkrylov_cpp import Device_FP32 as Device

def find_gpu():
    """Return the name of the GPU backend ('cuda', 'hip', 'sycl') if available, else None."""
    if Device.is_gpu_build():
        return Device.backend_name()
    return None

def gpu_count():
    """Return the number of available GPUs."""
    return Device.gpu_count()


try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("qkrylov")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    # Operators
    "Op",
    "OpSum",
    
    # Operator Generators
    "Sz", "Sp", "Sm", "Sx", "Sy",
    "CdagUp", "CUp", "CdagDn", "CDn",
    "Nup", "Ndn", "Nupdn",
    "Bdag", "B", "N",
    
    # Sites
    "Site",
    "SpinHalfSite",
    "FermionSite",
    "HubbardSite",
    "TJSite",
    
    # Bases
    "Basis",
    "SpinHalfBasis",
    "FermionBasis",
    "HubbardBasis",
    "TJBasis",
    
    # Hamiltonian
    "MatrixFreeHamiltonian",
    
    # Solvers
    "LanczosResult",
    "lanczos_ground_state",
    "DavidsonResult",
    "davidson_lowest",
    "DynamicsResult",
    "continued_fraction_coeffs",
    "evaluate_spectral_function",
    "FTLMResult",
    "ftlm",
    
    # Utilities
    "find_gpu",
    "gpu_count",
]
