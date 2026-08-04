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
