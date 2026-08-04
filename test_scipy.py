import qkrylov as qk
import numpy as np

# 2 sites, Sz=0
basis = qk.SpinHalfBasis(2, sz=0)
site = qk.SpinHalfSite()
os = qk.OpSum()
os += 1.0 * qk.Sz(0) * qk.Sz(1) + 0.5 * (qk.Sp(0) * qk.Sm(1) + qk.Sm(0) * qk.Sp(1))

H = qk.MatrixFreeHamiltonian(basis, site, os)
print("H dim:", H.dimension)

H_op = H.aslinearoperator()
print("LinearOperator shape:", H_op.shape)

H_mat = H.to_sparse()
print("Dense matrix:\n", H_mat.toarray())
