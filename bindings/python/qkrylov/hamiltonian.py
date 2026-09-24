import numpy as np
from typing import Union, Optional, Any, cast
from . import _qkrylov_cpp as _cpp
from .basis import Basis
from .site import Site
from .operators import OpSum

class MatrixFreeHamiltonian:
    """Matrix-free Hamiltonian operator executing on-the-fly matrix-vector products.

    Binds a physical Hilbert space (Basis), local physical rules (Site), and
    interaction terms (OpSum) into a callable operator capable of computing
    y = H @ x with zero matrix storage overhead.
    Traditional sparse matrix storage (CSR/COO) requires O(N * nnz_per_row)
    memory — for a Heisenberg chain with L=24 this is ~3 GB. qkrylov avoids
    this entirely. Instead of storing the matrix, we store the **operator sum**
    (a short list of interaction terms) and reconstruct matrix elements on the
    fly during each SpMV. This reduces memory to O(N) for the state vector plus
    O(terms) for the operator definition — a 100-1000x reduction.

    Hardware dispatch is compile-time via Kokkos execution space tags:
    ``device::cpu`` → OpenMP, ``device::gpu`` → CUDA, ``device::serial`` →
    scalar fallback. See :doc:`/theory/matrix_free_spmv`.

    Parameters
    ----------
    basis : Basis
        The many-body Hilbert space basis.
    site_or_ops : Union[Site, OpSum]
        Local site physics or interaction terms if site is omitted.
    ops : Optional[OpSum], default=None
        Interaction terms (if site is provided as second argument).
    device : str, default="cpu"
        Target execution device ("cpu", "cuda", "hip", "sycl").
    dtype : Any, default=None
        Precision (np.float32 or np.float64, inferred from basis if None).

    Attributes
    ----------
    dimension : int
        Total dimension of the many-body basis.
    device : str
        Active execution hardware device.
    dtype : Any
        Numerical precision.

    Methods
    -------
    apply(x)
        Apply Hamiltonian to state vector x, returning y = H * x.
    diagonal()
        Compute the diagonal elements of H in the basis.
    to(device=None, dtype=None)
        Create a clone targeted to a different device or precision.
    aslinearoperator()
        Convert to a scipy.sparse.linalg.LinearOperator.
    to_sparse()
        Construct explicit SciPy CSR matrix (for small test systems).

    Notes
    -----
    The SpMV kernel parallelizes over output state indices using Kokkos
    parallel_reduce. Each thread atomically accumulates contributions from
    the input vector into the output vector via a CSR-structured loop over
    non-zero connections. On GPU, warp-level reductions minimize global
    memory traffic. The one-time CSR construction at ``__init__`` time
    computes all off-diagonal connections and sorts them for coalesced
    memory access patterns on NVIDIA hardware.
    """

    basis: Basis
    site: Site
    ops: OpSum
    device: str
    dtype: Any
    _cpp_obj: Any
    _backend_suffix: str

    def __init__(
        self,
        basis: Basis,
        site_or_ops: Union[Site, OpSum],
        ops: Optional[OpSum] = None,
        device: str = "cpu",
        dtype: Any = None
    ):
        if dtype is None:
            if hasattr(basis, "dtype") and basis.dtype is not None:
                dtype = basis.dtype
            elif hasattr(basis, "_dtype") and basis._dtype is not None:
                dtype = basis._dtype
            elif hasattr(basis, "_cpp_obj") and basis._cpp_obj is not None and "FP64" in type(basis._cpp_obj).__name__:
                dtype = np.float64
            else:
                dtype = np.float64

        actual_site: Site
        actual_ops: OpSum
        if ops is None:
            # 2-argument invocation: Hamiltonian(basis, ops) -> infer site from basis
            if isinstance(site_or_ops, OpSum):
                actual_ops = site_or_ops
            else:
                raise TypeError("When ops is not provided, site_or_ops must be an OpSum")
            from .basis import SpinHalfBasis, SpinSBasis, FermionBasis, HubbardBasis, TJBasis
            from .site import SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite
            if isinstance(basis, SpinHalfBasis):
                actual_site = SpinHalfSite(dtype=dtype)
            elif isinstance(basis, SpinSBasis):
                actual_site = SpinSSite(S=basis.spin, dtype=dtype)
            elif isinstance(basis, FermionBasis):
                actual_site = FermionSite(dtype=dtype)
            elif isinstance(basis, HubbardBasis):
                actual_site = HubbardSite(dtype=dtype)
            elif isinstance(basis, TJBasis):
                actual_site = TJSite(dtype=dtype)
            else:
                actual_site = SpinHalfSite(dtype=dtype)
        else:
            if isinstance(site_or_ops, Site):
                actual_site = site_or_ops
            else:
                raise TypeError("When ops is provided, site_or_ops must be a Site")
            actual_ops = ops

        self.site = actual_site
        self.ops = actual_ops
        self.basis = basis
        self.device = device
        self.dtype = dtype
        s_dtype = "_FP64" if dtype == np.float64 else "_FP32"
        site: Site = self.site
        ops_val: OpSum = self.ops

        # Ensure site matches precision
        if getattr(site, "dtype", None) is not None and site.dtype != dtype:
            if hasattr(site, "spin"):
                from .site import SpinSSite
                self.site = SpinSSite(S=getattr(site, "spin"), dtype=dtype)
            else:
                site_cls: Any = site.__class__
                self.site = site_cls(dtype=dtype)
            site = self.site

        # Ensure ops matches precision
        if getattr(ops_val, "dtype", None) is not None and ops_val.dtype != dtype:
            converted_ops = OpSum(dtype=dtype)
            for t in ops_val._cpp_obj.terms():
                items = [t.coeff]
                for f in t.factors:
                    items.extend([f.op, f.site])
                converted_ops._cpp_obj.__iadd__(tuple(items))
            self.ops = converted_ops
            ops_val = self.ops

        dev_obj = getattr(_cpp, f"Device{s_dtype}")(device)
        d_lower = device.lower()
        if "cuda" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianCUDA{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianCUDA{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                self._backend_suffix = "CUDA"
            else:
                raise ValueError("QKrylov was not built with CUDA support. Install the CUDA wheel: pip install qkrylov --extra-index-url .../cuda")
        elif "hip" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianHIP{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianHIP{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                self._backend_suffix = "HIP"
            else:
                raise ValueError("QKrylov was not built with HIP support. Install the ROCm wheel: pip install qkrylov --extra-index-url .../rocm")
        elif "sycl" in d_lower:
            if hasattr(_cpp, f"MatrixFreeHamiltonianSYCL{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianSYCL{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                self._backend_suffix = "SYCL"
            else:
                raise ValueError("QKrylov was not built with SYCL support. Install the SYCL wheel: pip install qkrylov --extra-index-url .../sycl")
        elif d_lower.startswith("gpu"):
            # Generic "gpu" alias: try CUDA -> HIP -> SYCL in priority order
            for suffix in ("CUDA", "HIP", "SYCL"):
                cls_name = f"MatrixFreeHamiltonian{suffix}{s_dtype}"
                if hasattr(_cpp, cls_name):
                    self._cpp_obj = getattr(_cpp, cls_name)(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                    self._backend_suffix = suffix
                    break
            else:
                raise ValueError("No GPU backend available. This wheel was built for CPU only. Install a GPU wheel via --extra-index-url.")
        else:
            # Default: CPU backends (OpenMP -> Threads -> Serial)
            if hasattr(_cpp, f"MatrixFreeHamiltonianCPU{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianCPU{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                self._backend_suffix = "CPU"
            elif hasattr(_cpp, f"MatrixFreeHamiltonianThreads{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianThreads{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
                self._backend_suffix = "Threads"
            elif hasattr(_cpp, f"MatrixFreeHamiltonianSerial{s_dtype}"):
                self._cpp_obj = getattr(_cpp, f"MatrixFreeHamiltonianSerial{s_dtype}")(basis._cpp_obj, site._cpp_obj, self.ops._cpp_obj, dev_obj)
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

    def to(self, device: Optional[str] = None, dtype: Any = None) -> "MatrixFreeHamiltonian":
        """Return a new MatrixFreeHamiltonian with the specified device and dtype."""
        new_device = device if device is not None else self.device
        new_dtype = dtype if dtype is not None else self.dtype
        return MatrixFreeHamiltonian(self.basis, self.site, self.ops, device=new_device, dtype=new_dtype)

    def __matmul__(self, x: Union[np.ndarray, Any]) -> Any:
        """Matrix-free matrix-vector multiplication `y = H @ x`."""
        is_torch = False
        if hasattr(x, "__class__") and x.__class__.__name__ == "Tensor" and hasattr(x, "numpy"):
            is_torch = True
            dev = getattr(x, "device", None)
            x_arr = cast(Any, x).detach().cpu().numpy()
        else:
            x_arr = x
        y = self.apply(x_arr)
        if is_torch:
            try:
                import torch
                t = torch.from_numpy(y)
                return t.to(dev) if dev is not None else t
            except ImportError:
                pass
        return y

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


# Alias conforming to API blueprint
Hamiltonian = MatrixFreeHamiltonian
