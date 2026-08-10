import numpy as np
from typing import Union
from . import _qkrylov_cpp as _cpp
from .basis import Basis
from .site import Site
from .operators import OpSum

class MatrixFreeHamiltonian:
    """A matrix-free operator that applies the Hamiltonian to a state vector.
    
    This class binds a physical Hilbert space (Basis), local physical 
    rules (Site), and interaction terms (OpSum) into a single callable operator 
    capable of computing `y = H.apply(x)`.
    
    Parameters
    ----------
    basis : Basis
        The Hilbert space basis.
    site : Site
        The local site physics.
    ops : OpSum
        The interaction terms.
    """
    
    def __init__(self, basis: Basis, site: Site, ops: OpSum, device: str = "cpu", dtype=np.float32):
        self.basis = basis
        self.site = site
        self.ops = ops
        self.device = device
        self.dtype = dtype
        s_dtype = "_FP64" if dtype == np.float64 else "_FP32"
        
        dev_obj = getattr(_cpp, f"Device{s_dtype}")(device)
        d_lower = device.lower()
        if "cuda" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianCUDA{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianCUDA{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "CUDA"
            else:
                raise ValueError("QKrylov was not built with CUDA support. Install the CUDA wheel: pip install qkrylov --extra-index-url .../cuda")
        elif "hip" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianHIP{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianHIP{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "HIP"
            else:
                raise ValueError("QKrylov was not built with HIP support. Install the ROCm wheel: pip install qkrylov --extra-index-url .../rocm")
        elif "sycl" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianSYCL{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianSYCL{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "SYCL"
            else:
                raise ValueError("QKrylov was not built with SYCL support. Install the SYCL wheel: pip install qkrylov --extra-index-url .../sycl")
        elif d_lower.startswith("gpu"):
            # Generic "gpu" alias: try CUDA -> HIP -> SYCL in priority order
            for suffix in ("CUDA", "HIP", "SYCL"):
                cls_name = f"MatrixFreeHamiltonian{suffix}{s_dtype}"
                if hasattr(_cpp, cls_name):
                    self._cpp_obj = getattr(_cpp, cls_name)(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                    self._backend_suffix = suffix
                    break
            else:
                raise ValueError("No GPU backend available. This wheel was built for CPU only. Install a GPU wheel via --extra-index-url.")
        else:
            # Default: CPU backends (OpenMP -> Threads -> Serial)
            if hasattr(_cpp, f"MatrixFreeHamiltonianCPU{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianCPU{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "CPU"
            elif hasattr(_cpp, f"MatrixFreeHamiltonianThreads{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianThreads{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "Threads"
            elif hasattr(_cpp, f"MatrixFreeHamiltonianSerial{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianSerial{s_dtype}")(basis._cpp_obj, site._cpp_obj, ops._cpp_obj, dev_obj)
                self._backend_suffix = "Serial"
            else:
                raise ValueError("No CPU backend available. This wheel may be a GPU-only build. Try: pip install qkrylov (from PyPI) for the CPU version.")

    @property
    def dimension(self) -> int:
        """The total dimension of this Hamiltonian (size of the basis)."""
        return self._cpp_obj.dimension()

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the Hamiltonian to a state vector.
        
        Parameters
        ----------
        x : np.ndarray
            Input state vector of size `dimension`. Must be complex128 and C-contiguous.
            
        Returns
        -------
        np.ndarray
            The resulting state vector `y = H(x)`. Zero-copy: backed by C++ memory.
        """
        x = np.ascontiguousarray(x, dtype=np.complex128 if self.dtype == np.float64 else np.complex64)
        return self._cpp_obj.apply(x)

    def diagonal(self) -> np.ndarray:
        """Compute the diagonal of the Hamiltonian.
        
        Returns
        -------
        np.ndarray
            The diagonal elements. Zero-copy: backed by C++ memory.
        """
        return self._cpp_obj.diagonal()

    def __repr__(self) -> str:
        return f"MatrixFreeHamiltonian(dim={self.dimension})"

    def to(self, device: str = None, dtype = None):
        """Return a new MatrixFreeHamiltonian with the specified device and dtype."""
        new_device = device if device is not None else self.device
        new_dtype = dtype if dtype is not None else self.dtype
        return MatrixFreeHamiltonian(self.basis, self.site, self.ops, device=new_device, dtype=new_dtype)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def aslinearoperator(self):
        """Convert this Hamiltonian into a SciPy LinearOperator.
        
        This allows you to use the Hamiltonian directly with SciPy's sparse 
        solvers (e.g., `scipy.sparse.linalg.eigsh`) without ever explicitly 
        building the matrix in memory!
        
        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            A matrix-free operator compatible with SciPy.
        """
        try:
            import scipy.sparse.linalg as sla
        except ImportError:
            raise ImportError("scipy is required to use aslinearoperator(). Install it with `pip install scipy`.")
            
        def matvec(x):
            return self.apply(x)
            
        return sla.LinearOperator(
            shape=(self.dimension, self.dimension),
            matvec=matvec,
            dtype=np.complex128 if self.dtype == np.float64 else np.complex64
        )
        
    def to_sparse(self):
        """Construct the explicit sparse matrix in SciPy CSR format.
        
        Warning: This evaluates the full matrix explicitly. For large systems, 
        this will consume massive amounts of memory and time. This is primarily 
        intended for debugging small systems.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            The Hamiltonian as a full SciPy sparse matrix.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy is required to use to_sparse(). Install it with `pip install scipy`.")
            
        dim = self.dimension
        rows, cols, data = [], [], []
        
        for i in range(dim):
            x = np.zeros(dim, dtype=np.complex128 if self.dtype == np.float64 else np.complex64)
            x[i] = 1.0
            y = self.apply(x)
            
            non_zeros = np.nonzero(y)[0]
            if len(non_zeros) > 0:
                rows.extend(non_zeros)
                cols.extend([i] * len(non_zeros))
                data.extend(y[non_zeros])
                
        return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128 if self.dtype == np.float64 else np.complex64)
