import qkrylov as qk
import numpy as np

def test_symbolic_operator_and_scipy():
    # 2 sites, Sz=0
    basis = qk.SpinHalfBasis(2, sz=0)
    site = qk.SpinHalfSite()
    os = qk.OpSum()
    os += 1.0 * qk.Sz(0) * qk.Sz(1) + 0.5 * (qk.Sp(0) * qk.Sm(1) + qk.Sm(0) * qk.Sp(1))

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    assert H.dimension > 0

    H_op = H.aslinearoperator()
    assert H_op.shape == (H.dimension, H.dimension)

    H_mat = H.to_sparse()
    assert H_mat.shape == (H.dimension, H.dimension)

    # test to() method and __matmul__
    H2 = H.to(device="cpu")
    assert H2.dimension == H.dimension
    x = np.random.rand(H.dimension)
    y1 = H.apply(x)
    y2 = H2 @ x
    np.testing.assert_allclose(y1, y2)

