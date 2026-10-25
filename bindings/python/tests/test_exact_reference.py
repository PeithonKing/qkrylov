"""Exact reference tests.

Every Hamiltonian here is also built independently in pure Python/SciPy from
textbook operator definitions, and the library's H @ x and Lanczos energies
are compared against it. Products follow the standard convention: in A * B,
B acts first.
"""
import itertools
import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import qkrylov as qk
from qkrylov.operators import LocalOpExpr

DTYPES = [np.float64, np.float32]


def _tol(dtype):
    return 1e-8 if dtype == np.float64 else 2e-4


# --------------------------------------------------------------------------
# Reference spin-1/2 operators (bit = 1 means spin up)
# --------------------------------------------------------------------------
def _spin_action(op, site, state):
    """Return (new_state, amplitude) or None for a single-site spin operator."""
    up = (state >> site) & 1
    bit = 1 << site
    if op == "Sz":
        return state, (0.5 if up else -0.5)
    if op == "Sp":
        return (state | bit, 1.0) if not up else None
    if op == "Sm":
        return (state & ~bit, 1.0) if up else None
    if op == "Sx":
        return state ^ bit, 0.5
    if op == "Sy":
        # Sy = (Sp - Sm) / 2i :  Sy|up> = +i/2 |dn>,  Sy|dn> = -i/2 |up>
        return state ^ bit, (0.5j if up else -0.5j)
    raise ValueError(op)


def _reference_matrix(states, terms):
    """Sparse reference H on the given basis states for spin-1/2 terms."""
    index = {s: i for i, s in enumerate(states)}
    rows, cols, vals = [], [], []
    for coeff, factors in terms:
        for col, s in enumerate(states):
            amp = complex(coeff)
            cur = s
            for op, site in reversed(factors):  # rightmost factor acts first
                res = _spin_action(op, site, cur)
                if res is None:
                    amp = 0
                    break
                cur, a = res
                amp *= a
            if amp != 0 and cur in index:
                rows.append(index[cur])
                cols.append(col)
                vals.append(amp)
    n = len(states)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128)


def _build_opsum(terms):
    ops = qk.OpSum()
    for coeff, factors in terms:
        expr = coeff
        for op, site in factors:
            expr = expr * LocalOpExpr(op, site)
        ops += expr
    return ops


def _basis_states(basis):
    return [basis._cpp_obj.state(i) for i in range(basis.size)]


def _check_apply(basis, terms, dtype, site=None):
    site = site or qk.SpinHalfSite()
    H = qk.MatrixFreeHamiltonian(basis, site, _build_opsum(terms), dtype=dtype)
    ref = _reference_matrix(_basis_states(basis), terms)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(basis.size) + 1j * rng.standard_normal(basis.size)
    y = np.asarray(H @ x)
    y_ref = ref @ x
    err = np.max(np.abs(y - y_ref)) / max(1.0, np.max(np.abs(y_ref)))
    assert err < _tol(dtype), f"H @ x differs from reference (rel err {err:.3e})"
    return H, ref


def _xxz_terms(L, J=1.0, delta=0.7, h=0.3, pbc=True):
    bonds = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if pbc else [])
    terms = []
    for i, j in bonds:
        terms.append((J * delta, [("Sz", i), ("Sz", j)]))
        terms.append((0.5 * J, [("Sp", i), ("Sm", j)]))
        terms.append((0.5 * J, [("Sm", i), ("Sp", j)]))
    for i in range(L):
        terms.append((h * (1 + 0.1 * i), [("Sz", i)]))
    return terms


# --------------------------------------------------------------------------
# Spin-1/2: H @ x against the reference, including sites >= 16
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sz", [0, 1])
def test_xxz_sector_L18_apply(dtype, sz):
    L = 18
    basis = qk.SpinHalfBasis(L, sz=sz)
    assert basis.size == math.comb(L, L // 2 + sz)
    _check_apply(basis, _xxz_terms(L), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_xxz_sector_L18_energy(dtype):
    L = 18
    basis = qk.SpinHalfBasis(L, sz=0)
    terms = _xxz_terms(L)
    H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(terms), dtype=dtype)
    ref = _reference_matrix(_basis_states(basis), terms)
    e_ref = spla.eigsh(ref.real, k=1, which="SA")[0][0]
    res = qk.solvers.Lanczos(maxiter=400, tol=1e-12).solve(H)
    assert abs(res.energy - e_ref) < (1e-7 if dtype == np.float64 else 1e-3) * L


@pytest.mark.parametrize("dtype", DTYPES)
def test_heisenberg_long_range_bonds_L20(dtype):
    """Bonds touching both low sites (< 16) and high sites (>= 16)."""
    L = 20
    pairs = [(0, 19), (3, 17), (15, 16), (14, 15), (7, 18), (16, 19)]
    terms = []
    for i, j in pairs:
        terms.append((1.0, [("Sz", i), ("Sz", j)]))
        terms.append((0.5, [("Sp", i), ("Sm", j)]))
        terms.append((0.5, [("Sm", i), ("Sp", j)]))
    _check_apply(qk.SpinHalfBasis(L, sz=0), terms, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_tfim_full_space_apply_and_energy(dtype):
    L = 10
    terms = [(-1.0, [("Sz", i), ("Sz", (i + 1) % L)]) for i in range(L)]
    terms += [(-0.8, [("Sx", i)]) for i in range(L)]
    basis = qk.SpinHalfBasis(L)
    assert basis.size == 2 ** L
    H, ref = _check_apply(basis, terms, dtype)
    e_ref = np.linalg.eigvalsh(ref.toarray().real)[0]
    res = qk.solvers.Lanczos(maxiter=400, tol=1e-12).solve(H)
    assert abs(res.energy - e_ref) < (1e-7 if dtype == np.float64 else 1e-3) * L


@pytest.mark.parametrize("dtype", DTYPES)
def test_complex_sy_terms_apply(dtype):
    """Single Sy and mixed Sx*Sy terms (Dzyaloshinskii-Moriya-like)."""
    L = 8
    terms = []
    for i in range(L - 1):
        terms.append((0.4, [("Sx", i), ("Sy", i + 1)]))
        terms.append((-0.4, [("Sy", i), ("Sx", i + 1)]))
    terms += [(0.3, [("Sy", i)]) for i in range(L)]
    _check_apply(qk.SpinHalfBasis(L), terms, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_spin_operator_identities(dtype):
    """Sx = (Sp + Sm)/2 and Sy = (Sp - Sm)/2i, using only the library."""
    L = 6
    basis = qk.SpinHalfBasis(L)
    site = qk.SpinHalfSite()
    rng = np.random.default_rng(3)
    x = rng.standard_normal(basis.size) + 1j * rng.standard_normal(basis.size)
    for i in (0, 3, 5):
        def H(terms):
            return np.asarray(qk.MatrixFreeHamiltonian(basis, site, _build_opsum(terms), dtype=dtype) @ x)
        sx = H([(1.0, [("Sx", i)])])
        sy = H([(1.0, [("Sy", i)])])
        sp_ = H([(1.0, [("Sp", i)])])
        sm_ = H([(1.0, [("Sm", i)])])
        assert np.allclose(sx, 0.5 * (sp_ + sm_), atol=_tol(dtype))
        assert np.allclose(sy, (sp_ - sm_) / 2j, atol=_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("factors", [
    [("Sz", 2), ("Sp", 2)],
    [("Sp", 2), ("Sz", 2)],
    [("Sp", 2), ("Sm", 2)],
    [("Sm", 2), ("Sp", 2)],
    [("Sz", 2), ("Sz", 2)],
    [("Sx", 2), ("Sz", 2)],
    [("Sp", 2), ("Sm", 2), ("Sz", 4)],
])
def test_same_site_products(dtype, factors):
    """Products acting twice on the same site; B acts first in A * B."""
    _check_apply(qk.SpinHalfBasis(6), [(1.0, factors)], dtype)


# --------------------------------------------------------------------------
# Fermions: free-fermion energies are independent of sign conventions
# --------------------------------------------------------------------------
def _hopping_matrix(L, seed=11):
    rng = np.random.default_rng(seed)
    h = np.zeros((L, L))
    for i, j in itertools.combinations(range(L), 2):
        if j - i <= 3:  # includes non-nearest-neighbour hops -> Jordan-Wigner strings matter
            h[i, j] = h[j, i] = rng.uniform(-1, 1)
    h += np.diag(rng.uniform(-0.5, 0.5, L))
    return h


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 4])
def test_free_spinless_fermions_energy(dtype, n):
    L = 10
    h = _hopping_matrix(L)
    ops = qk.OpSum()
    for i in range(L):
        ops += h[i, i] * LocalOpExpr("N", i)
        for j in range(L):
            if i != j and h[i, j] != 0:
                ops += h[i, j] * LocalOpExpr("Cdag", i) * LocalOpExpr("C", j)
    basis = qk.FermionBasis(L, n=n)
    assert basis.size == math.comb(L, n)
    H = qk.MatrixFreeHamiltonian(basis, qk.FermionSite(), ops, dtype=dtype)
    e_exact = np.sort(np.linalg.eigvalsh(h))[:n].sum()
    res = qk.solvers.Lanczos(maxiter=400, tol=1e-12).solve(H)
    assert abs(res.energy - e_exact) < (1e-7 if dtype == np.float64 else 1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fermion_number_operator_same_site(dtype):
    """Cdag(i) * C(i) must equal N(i)."""
    L = 8
    basis = qk.FermionBasis(L, n=3)
    site = qk.FermionSite()
    rng = np.random.default_rng(5)
    x = rng.standard_normal(basis.size) + 1j * rng.standard_normal(basis.size)
    for i in (0, 4, 7):
        a = qk.OpSum(); a += 1.0 * LocalOpExpr("Cdag", i) * LocalOpExpr("C", i)
        b = qk.OpSum(); b += 1.0 * LocalOpExpr("N", i)
        ya = np.asarray(qk.MatrixFreeHamiltonian(basis, site, a, dtype=dtype) @ x)
        yb = np.asarray(qk.MatrixFreeHamiltonian(basis, site, b, dtype=dtype) @ x)
        assert np.allclose(ya, yb, atol=_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_free_hubbard_energy(dtype):
    L, nup, ndn = 6, 3, 2
    h = _hopping_matrix(L, seed=4)
    np.fill_diagonal(h, 0.0)
    ops = qk.OpSum()
    for i in range(L):
        for j in range(L):
            if i != j and h[i, j] != 0:
                ops += h[i, j] * qk.CdagUp(i) * qk.CUp(j)
                ops += h[i, j] * qk.CdagDn(i) * qk.CDn(j)
    basis = qk.HubbardBasis(L, nup=nup, ndn=ndn)
    H = qk.MatrixFreeHamiltonian(basis, qk.HubbardSite(), ops, dtype=dtype)
    eps = np.sort(np.linalg.eigvalsh(h))
    e_exact = eps[:nup].sum() + eps[:ndn].sum()
    res = qk.solvers.Lanczos(maxiter=400, tol=1e-12).solve(H)
    assert abs(res.energy - e_exact) < (1e-7 if dtype == np.float64 else 1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("U", [0.0, 2.0, 8.0])
def test_two_site_hubbard_energy(dtype, U):
    """Half-filled 2-site Hubbard: E0 = (U - sqrt(U^2 + 16 t^2)) / 2."""
    t = 1.0
    ops = qk.OpSum()
    for i, j in ((0, 1), (1, 0)):
        ops += -t * qk.CdagUp(i) * qk.CUp(j)
        ops += -t * qk.CdagDn(i) * qk.CDn(j)
    for i in range(2):
        ops += U * qk.Nupdn(i)
    basis = qk.HubbardBasis(2, nup=1, ndn=1)
    H = qk.MatrixFreeHamiltonian(basis, qk.HubbardSite(), ops, dtype=dtype)
    res = qk.solvers.Lanczos(maxiter=50, tol=1e-12).solve(H)
    assert abs(res.energy - (U - math.sqrt(U * U + 16 * t * t)) / 2) < _tol(dtype) * 10


# --------------------------------------------------------------------------
# Python API: site inference (Change 6)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_site_inference_warns_and_works(dtype):
    L = 6
    basis = qk.SpinHalfBasis(L, sz=0)
    terms = _xxz_terms(L, pbc=False)
    with pytest.warns(UserWarning, match="No Site provided"):
        H = qk.MatrixFreeHamiltonian(basis, _build_opsum(terms), dtype=dtype)
    assert H.dtype == dtype
    H_explicit = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(terms), dtype=dtype)
    x = np.random.default_rng(0).standard_normal(basis.size).astype(np.complex128)
    assert np.allclose(np.asarray(H @ x), np.asarray(H_explicit @ x), atol=_tol(dtype))


# --------------------------------------------------------------------------
# t-J: projected hopping. Its basis excludes double occupancy, so a hop can
# come from a source state outside the basis; those terms must be dropped.
# --------------------------------------------------------------------------
def _tj_reference(states, L, t, J=0.0):
    """Dense t-J matrix. Mode order: bit 2i = (i, up), bit 2i+1 = (i, dn); c acts first in c^dag c."""
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    H = np.zeros((n, n))
    def bit(i, spin): return 1 << (2 * i + spin)
    def sign(state, b): return -1.0 if bin(state & (b - 1)).count("1") & 1 else 1.0
    for col, s0 in enumerate(states):
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                for spin in (0, 1):
                    other = 1 - spin
                    bj, bi = bit(j, spin), bit(i, spin)
                    if not (s0 & bj):                    # c_j needs the mode occupied
                        continue
                    amp = sign(s0, bj); s1 = s0 & ~bj
                    if s1 & bi:                          # c^dag_i needs the mode empty
                        continue
                    if s1 & bit(i, other):               # projector: no double occupancy
                        continue
                    amp *= sign(s1, bi); s2 = s1 | bi
                    if s2 in index:
                        H[index[s2], col] += -t * amp
    return H


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,nup,ndn", [(5, 2, 2), (6, 3, 1), (6, 2, 2)])
def test_tj_hopping_matches_dense_reference(dtype, L, nup, ndn):
    basis = qk.TJBasis(L, nup=nup, ndn=ndn)
    states = [basis._cpp_obj.state(i) for i in range(basis.size)]
    assert all(((s >> (2 * i)) & 1) + ((s >> (2 * i + 1)) & 1) <= 1 for s in states for i in range(L))
    ops = qk.OpSum()
    for i in range(L):
        for j in range(L):
            if i != j:
                ops += -1.0 * LocalOpExpr("CdagUp", i) * LocalOpExpr("CUp", j)
                ops += -1.0 * LocalOpExpr("CdagDn", i) * LocalOpExpr("CDn", j)
    H = qk.MatrixFreeHamiltonian(basis, qk.TJSite(), ops, dtype=dtype)
    ref = _tj_reference(states, L, 1.0, 0.0)
    assert np.allclose(ref, ref.T), "reference must be Hermitian"
    rng = np.random.default_rng(1)
    x = rng.standard_normal(basis.size) + 1j * rng.standard_normal(basis.size)
    y = np.asarray(H @ x)
    assert np.max(np.abs(y - ref @ x)) < _tol(dtype) * max(1.0, np.max(np.abs(ref @ x)))


# --------------------------------------------------------------------------
# Kernel structure: >64 instructions (several chunks) and hops whose bits
# straddle byte boundaries of the state.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sz", [0, 1])
def test_all_to_all_xxz_many_instructions(dtype, sz):
    L = 12
    rng = np.random.default_rng(21)
    terms = []
    for i, j in itertools.combinations(range(L), 2):   # 66 pairs: more than one 64-instruction chunk
        terms.append((rng.uniform(0.2, 1.0), [("Sp", i), ("Sm", j)]))
        terms.append((rng.uniform(0.2, 1.0), [("Sm", i), ("Sp", j)]))
        terms.append((rng.uniform(-1.0, 1.0), [("Sz", i), ("Sz", j)]))
    basis = qk.SpinHalfBasis(L, sz=sz)
    H, ref = _check_apply(basis, terms, dtype)
    assert H.dimension == math.comb(L, L // 2 + sz)


@pytest.mark.parametrize("dtype", DTYPES)
def test_byte_straddling_hops_hubbard(dtype):
    """Hubbard modes 2i, 2i+1: hopping across sites 3/4 puts bits on both sides of a byte edge."""
    L, nup, ndn = 7, 3, 2
    ops = qk.OpSum()
    pairs = [(3, 4), (0, 6), (2, 5), (4, 3)]
    for i, j in pairs:
        ops += -1.0 * qk.CdagUp(i) * qk.CUp(j) + -1.0 * qk.CdagUp(j) * qk.CUp(i)
        ops += -1.0 * qk.CdagDn(i) * qk.CDn(j) + -1.0 * qk.CdagDn(j) * qk.CDn(i)
    for i in range(L):
        ops += 1.5 * qk.Nupdn(i)
    basis = qk.HubbardBasis(L, nup=nup, ndn=ndn)
    H = qk.MatrixFreeHamiltonian(basis, qk.HubbardSite(), ops, dtype=dtype)
    eye = np.eye(basis.size, dtype=np.complex128)
    Hd = np.column_stack([np.asarray(H @ eye[:, c]) for c in range(basis.size)])
    assert np.allclose(Hd, Hd.conj().T, atol=_tol(dtype)), "Hubbard hopping must be Hermitian"
    # independent check on the spectrum: with U=0 the ground energy is the free-fermion one
    ops0 = qk.OpSum()
    h = np.zeros((L, L))
    for i, j in pairs:
        h[i, j] = h[j, i] = -1.0
        ops0 += -1.0 * qk.CdagUp(i) * qk.CUp(j) + -1.0 * qk.CdagUp(j) * qk.CUp(i)
        ops0 += -1.0 * qk.CdagDn(i) * qk.CDn(j) + -1.0 * qk.CdagDn(j) * qk.CDn(i)
    # duplicate pair (3,4)/(4,3) means the hop amplitude is doubled in the operator
    h[3, 4] = h[4, 3] = -2.0
    eps = np.sort(np.linalg.eigvalsh(h))
    H0 = qk.MatrixFreeHamiltonian(basis, qk.HubbardSite(), ops0, dtype=dtype)
    res = qk.solvers.Lanczos(maxiter=500, tol=1e-12).solve(H0)
    assert abs(res.energy - (eps[:nup].sum() + eps[:ndn].sum())) < (1e-7 if dtype == np.float64 else 1e-3)


# --------------------------------------------------------------------------
# Real Hamiltonians give real results (API Change 7)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", [qk.solvers.Lanczos, qk.solvers.LanczosTwoPass])
def test_real_hamiltonian_returns_real_eigenvector(dtype, solver):
    L = 8
    basis = qk.SpinHalfBasis(L, sz=0)
    H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(_xxz_terms(L)), dtype=dtype)
    assert H.is_real
    res = solver(maxiter=300, tol=1e-12).solve(H)
    real_t = np.float64 if dtype == np.float64 else np.float32
    assert res.eigenvector.dtype == real_t
    ref = _reference_matrix(_basis_states(basis), _xxz_terms(L)).real
    resid = np.linalg.norm(ref @ res.eigenvector - res.energy * res.eigenvector)
    assert resid < (1e-5 if dtype == np.float64 else 1e-3)  # vector error ~ sqrt(energy error)
    sparse = H.to_sparse()
    assert sparse.dtype == real_t
    assert abs(sparse - sp.csr_matrix(ref)).max() < _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_complex_hamiltonian_stays_complex(dtype):
    L = 6
    terms = [(0.4, [("Sx", i), ("Sy", i + 1)]) for i in range(L - 1)] + \
            [(-0.4, [("Sy", i), ("Sx", i + 1)]) for i in range(L - 1)] + \
            [(1.0, [("Sz", i), ("Sz", i + 1)]) for i in range(L - 1)]
    H = qk.MatrixFreeHamiltonian(qk.SpinHalfBasis(L), qk.SpinHalfSite(), _build_opsum(terms), dtype=dtype)
    assert not H.is_real
    res = qk.solvers.Lanczos(maxiter=300, tol=1e-12).solve(H)
    assert np.iscomplexobj(res.eigenvector)
    assert np.iscomplexobj(H.to_sparse().data)


# --------------------------------------------------------------------------
# Spin algebra: the library's Sx, Sy, Sz must satisfy [Sx, Sy] = i Sz (cyclic).
# This pins the sign convention of Sy independently of any hand-written matrix.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("site", [0, 2])
def test_spin_commutation_relations(dtype, site):
    L = 4
    basis = qk.SpinHalfBasis(L)
    rng = np.random.default_rng(9)
    x = rng.standard_normal(basis.size) + 1j * rng.standard_normal(basis.size)

    def act(terms):
        H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(terms), dtype=dtype)
        return np.asarray(H @ x)

    i = site
    for (a, b, c) in [("Sx", "Sy", "Sz"), ("Sy", "Sz", "Sx"), ("Sz", "Sx", "Sy")]:
        comm = act([(1.0, [(a, i), (b, i)]), (-1.0, [(b, i), (a, i)])])
        assert np.allclose(comm, act([(1j, [(c, i)])]), atol=_tol(dtype)), f"[{a},{b}] != i {c}"
    # and Sy is (Sp - Sm)/2i
    sy = act([(1.0, [("Sy", i)])])
    assert np.allclose(sy, act([(1 / 2j, [("Sp", i)]), (-1 / 2j, [("Sm", i)])]), atol=_tol(dtype))


# --------------------------------------------------------------------------
# early_stopping=False: fixed work
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", [qk.solvers.Lanczos, qk.solvers.LanczosTwoPass])
def test_early_stopping_flag(dtype, solver):
    L = 10
    basis = qk.SpinHalfBasis(L, sz=0)
    H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(_xxz_terms(L)), dtype=dtype)
    tol = 1e-10 if dtype == np.float64 else 1e-4
    early = solver(maxiter=60, tol=tol).solve(H)
    full = solver(maxiter=60, tol=tol, early_stopping=False).solve(H)
    assert early.converged and early.iterations < 60
    assert full.iterations == 60 and not full.converged
    assert abs(full.energy - early.energy) < (1e-8 if dtype == np.float64 else 2e-3)
    # the functional wrappers forward the flag too
    assert qk.lanczos_ground_state(H, maxiter=25, tol=tol, early_stopping=False).iterations == 25
    assert qk.solvers.lanczos_two_pass(H, maxiter=25, tol=tol, early_stopping=False).iterations == 25


# --------------------------------------------------------------------------
# H @ x runs on the caller's arrays on the CPU: it must not modify x, must give a fresh
# result array, and must still be right when called repeatedly / with x == previous y.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_matvec_does_not_touch_input_and_returns_fresh_array(dtype):
    L = 8
    basis = qk.SpinHalfBasis(L, sz=0)
    terms = _xxz_terms(L)
    H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(terms), dtype=dtype)
    ref = _reference_matrix(_basis_states(basis), terms)
    cdt = np.complex128 if dtype == np.float64 else np.complex64
    x = (np.random.default_rng(2).standard_normal(basis.size) + 1j * np.random.default_rng(3).standard_normal(basis.size)).astype(cdt)
    x0 = x.copy()
    y1 = np.asarray(H @ x)
    assert np.array_equal(x, x0), "H @ x modified its input"
    assert not np.shares_memory(x, y1)
    y2 = np.asarray(H @ y1)                      # feed the previous result back in
    assert not np.shares_memory(y1, y2)
    assert np.allclose(y2, ref @ (ref @ x0), atol=_tol(dtype) * 20)
    ro = x.copy(); ro.setflags(write=False)      # read-only input is fine
    assert np.allclose(np.asarray(H @ ro), ref @ x0, atol=_tol(dtype) * 5)


# --------------------------------------------------------------------------
# return_state=False: energy only
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", [qk.solvers.Lanczos, qk.solvers.LanczosTwoPass])
def test_energy_only(dtype, solver):
    L = 10
    basis = qk.SpinHalfBasis(L, sz=0)
    H = qk.MatrixFreeHamiltonian(basis, qk.SpinHalfSite(), _build_opsum(_xxz_terms(L)), dtype=dtype)
    full = solver(maxiter=200, tol=1e-12).solve(H)
    only = solver(maxiter=200, tol=1e-12, return_state=False).solve(H)
    assert only.eigenvector is None
    assert abs(only.energy - full.energy) < (1e-9 if dtype == np.float64 else 1e-3)
    energy, psi = qk.lanczos_ground_state(H, return_state=False)
    assert psi is None and abs(energy - full.energy) < (1e-9 if dtype == np.float64 else 1e-3)
    assert qk.lanczos_two_pass(H, return_state=False).eigenvector is None


# --------------------------------------------------------------------------
# Public basis accessors: state / index / contains
# --------------------------------------------------------------------------
@pytest.mark.parametrize("L,sz", [(6, 0), (7, 0.5), (8, 1)])
def test_spinhalf_basis_public_accessors(L, sz):
    basis = qk.basis.SpinHalf(L, sz=sz)
    states = [basis.state(i) for i in range(basis.size)]
    k = int(round(sz + L / 2))
    assert states == sorted(s for s in range(2 ** L) if bin(s).count("1") == k)
    assert [basis.index(s) for s in states] == list(range(basis.size))
    assert all(basis.contains(s) for s in states)
    assert not basis.contains(2 ** L - 1 if k != L else 0)
    with pytest.raises(RuntimeError):
        basis.index(2 ** L - 1 if k != L else 0)
    with pytest.raises(Exception):
        basis.state(basis.size)


def test_large_basis_state_is_cheap_and_exact():
    """L=40, Sz=0 has 1.4e11 states: state(i) / index(s) must not enumerate anything."""
    from math import comb
    basis = qk.basis.SpinHalf(40, sz=0)
    assert basis.size == comb(40, 20)
    for i in (0, 1, 12345, basis.size // 2, basis.size - 1):
        s = basis.state(i)
        assert bin(s).count("1") == 20 and basis.index(s) == i
