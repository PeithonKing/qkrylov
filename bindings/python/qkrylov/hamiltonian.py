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
    
    def __init__(self, basis: Basis, site: Site, ops: OpSum):
        self.basis = basis
        self.site = site
        self.ops = ops
        
        # Instantiate the underlying C++ Hamiltonian
        self._cpp_obj = _cpp.MatrixFreeHamiltonian(
            basis._cpp_obj, site._cpp_obj, ops._cpp_obj
        )

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
        x = np.ascontiguousarray(x, dtype=np.complex128)
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
            dtype=np.complex128
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
            x = np.zeros(dim, dtype=np.complex128)
            x[i] = 1.0
            y = self.apply(x)
            
            non_zeros = np.nonzero(y)[0]
            if len(non_zeros) > 0:
                rows.extend(non_zeros)
                cols.extend([i] * len(non_zeros))
                data.extend(y[non_zeros])
                
        return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
