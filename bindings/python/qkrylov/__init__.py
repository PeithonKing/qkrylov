"""Python interface for qkrylov.

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics,
now with a Pythonic wrapper layer.
"""

import ctypes as _ctypes
import glob as _glob
import os as _os
import sys as _sys


def _preload_cuda_runtime():
    """Load libcudart from the ``nvidia-cuda-runtime-cu12`` pip package, if installed.

    The CUDA wheel does not bundle the CUDA runtime. ``pip install qkrylov[cuda]`` puts it under
    ``site-packages/nvidia/cuda_runtime/lib``, which the dynamic loader does not search, so it has to be loaded
    before the compiled extension is imported. A system CUDA installation needs nothing here.
    """
    for entry in _sys.path:
        for lib in _glob.glob(_os.path.join(entry, "nvidia", "cuda_runtime", "lib", "libcudart.so*")):
            try:
                _ctypes.CDLL(lib, mode=_os.RTLD_GLOBAL)
                return
            except OSError:
                pass


_preload_cuda_runtime()

from .operators import (
    Op, OpSum,
    Sz, Sp, Sm, Sx, Sy,
    CdagUp, CUp, CdagDn, CDn,
    Nup, Ndn, Nupdn,
    Bdag, B, N
)
from . import operators
from . import site
from . import basis
from . import hamiltonian
from . import solvers

from .site import Site, SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite
from .basis import Basis, SpinHalfBasis, SpinSBasis, FermionBasis, HubbardBasis, TJBasis
from .hamiltonian import MatrixFreeHamiltonian, Hamiltonian
from .solvers import (
    Solver,
    Lanczos,
    LanczosTwoPass,
    Davidson,
    FTLM,
    ContinuedFraction,
    CorrectionVector,
    LanczosResult,
    lanczos_ground_state,
    lanczos_two_pass,
    DavidsonResult,
    davidson_lowest,
    DynamicsResult,
    continued_fraction_coeffs,
    evaluate_spectral_function,
    FTLMResult,
    FTLMSweepResult,
    FTLMSamples,
    ftlm,
    ftlm_sweep_streamed,
    ftlm_sample,
    ftlm_evaluate_sweep,
    RealTimeResult,
    TimeEvolve,
    time_evolve,
    FTLMDynamicsResult,
    FTLMDynamics,
    ftlm_dynamics,
    CorrectionVectorResult,
    correction_vector,
)

from typing import Optional
from ._qkrylov_cpp import Device_FP32 as Device

def find_gpu() -> Optional[str]:
    """Detect available hardware GPU acceleration backend.

    Inspects compiled C++ bindings to determine if CUDA, ROCm/HIP, or Intel SYCL
    acceleration runtimes are active and functional.

    **Prerequisites by backend:**

    - **CUDA:** NVIDIA driver >= 525.60, CUDA runtime >= 12.0, any Ampere/Ada GPU.
    - **HIP/ROCm:** AMD ROCm >= 5.6, gfx90a (MI250) or gfx1100 (RX 7900) GPU.
    - **SYCL:** Intel oneAPI >= 2024.0, Intel Arc or Intel Data Center GPU Max.

    The library is compiled for exactly one backend at build time. A CPU-only
    build (``Kokkos_ENABLE_CUDA=OFF``) will always return ``None`` here.

    Returns
    -------
    Optional[str]
        The active GPU backend name ('cuda', 'hip', 'sycl') or None if CPU only.

    Notes
    -----
    At runtime, Kokkos probes the system's dynamic library loader for the
    relevant GPU runtime (libcuda.so, libamdhip64.so, or libsycl.so).
    If the compiled backend library is not found at runtime — e.g., running a
    CUDA-compiled binary on a machine without a GPU driver — Kokkos falls back
    to a CPU serial execution space with a runtime warning.

    Examples
    --------
    >>> import qkrylov as qk
    >>> if qk.find_gpu():
    ...     H = qk.Hamiltonian(basis, site, ops).to("cuda")
    ... else:
    ...     H = qk.Hamiltonian(basis, site, ops)
    """
    if Device.is_gpu_build():
        return Device.backend_name()
    return None

def gpu_count() -> int:
    """Return the number of accessible GPU accelerators.

    Queries the active GPU runtime (CUDA, ROCm, or SYCL) to enumerate
    the physical compute devices available to this process.

    Returns
    -------
    int
        Total number of detected GPU devices. Returns 0 if no GPU runtime
        is active or the library was compiled for CPU only.

    Notes
    -----
    On multi-GPU nodes, this returns the total visible device count. Use
    the environment variable ``CUDA_VISIBLE_DEVICES`` (CUDA) or
    ``HIP_VISIBLE_DEVICES`` (ROCm) to restrict which devices are visible
    before importing qkrylov. Kokkos picks device 0 by default; to target
    a specific GPU, set ``Kokkos::InitializationSettings.device_id`` at
    the C++ level or set ``CUDA_VISIBLE_DEVICES=2`` before launching Python.

    Examples
    --------
    >>> print(f"Running on {qk.gpu_count()} GPU(s)")
    Running on 1 GPU(s)
    """
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
    "SpinSSite",
    "FermionSite",
    "HubbardSite",
    "TJSite",
    
    # Bases
    "Basis",
    "SpinHalfBasis",
    "SpinSBasis",
    "FermionBasis",
    "HubbardBasis",
    "TJBasis",
    
    # Submodules
    "operators",
    "site",
    "basis",
    "hamiltonian",
    "models",
    "solvers",

    # Hamiltonian
    "MatrixFreeHamiltonian",
    "Hamiltonian",
    
    # Solvers
    "Solver",
    "Lanczos",
    "LanczosTwoPass",
    "Davidson",
    "FTLM",
    "ContinuedFraction",
    "CorrectionVector",
    "LanczosResult",
    "lanczos_ground_state",
    "lanczos_two_pass",
    "DavidsonResult",
    "davidson_lowest",
    "DynamicsResult",
    "continued_fraction_coeffs",
    "evaluate_spectral_function",
    "FTLMResult",
    "FTLMSweepResult",
    "FTLMSamples",
    "ftlm",
    "ftlm_sweep_streamed",
    "ftlm_sample",
    "ftlm_evaluate_sweep",
    "RealTimeResult",
    "TimeEvolve",
    "time_evolve",
    "FTLMDynamicsResult",
    "FTLMDynamics",
    "ftlm_dynamics",
    "CorrectionVectorResult",
    "correction_vector",
    
    # Utilities
    "find_gpu",
    "gpu_count",
]
from . import models
