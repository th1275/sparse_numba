"""
API test helpers: the 100x100 matrix of test_prefactorize_slu.py :22, the
69-bus vehicle (through _sn1_common), each backend's reference ordering, and
the backend availability probe. The API is imported inside the cells.
"""

import importlib

import numpy as np

import _sn1_common as common

BACKEND_NAMES = ("superlu", "umfpack")

# the reference ordering per backend: SuperLU NATURAL (the oracle), UMFPACK AMD
# (its 0.1.11 default under STRATEGY_SYMMETRIC, umfpack.h :308)
SN2_ORDERING = {"superlu": "NATURAL", "umfpack": "AMD"}


CY_MODULE = {
    "superlu": "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "umfpack": "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
    "eigen_lu": "sparse_numba.sparse_eigen.cy_eigen_wrapper",
    "eigen_ldlt": "sparse_numba.sparse_eigen.cy_eigen_wrapper",
}

# the Eigen kinds' ordering in the cache child (sn2_cache_child.py);
# they are not in BACKEND_NAMES (the API cells compare with the 0.1.11
# oracle, which has no Eigen x)
SN2_ORDERING.update({"eigen_lu": "COLAMD", "eigen_ldlt": "AMD"})


def fixed_spd_system():
    """The fixed 60x60's pattern made symmetric positive definite: A + A^T
    off the diagonal, the diagonal the row's absolute sum + 1; dense A, b."""
    a, b = common.fixed_system()
    s = a + a.T
    np.fill_diagonal(s, 0.0)
    np.fill_diagonal(s, np.abs(s).sum(axis=1) + 1.0)
    return s, b


def system_for(backend):
    """The cache child's system: the fixed 60x60, or its SPD form for LDLT."""
    if backend == "eigen_ldlt":
        return fixed_spd_system()
    return common.fixed_system()


def require_backend(backend):
    import pytest
    try:
        importlib.import_module(CY_MODULE[backend])
    except ImportError as e:
        pytest.skip(f"{backend} extension absent: {e}")


def api_ids(api, backend):
    """(backend id, ordering id) of the new API for a backend name."""
    return getattr(api, backend.upper()), getattr(api, SN2_ORDERING[backend])


def make_test_matrix(n=100, density=0.05, seed=42):
    """test_prefactorize_slu.py :22's well-conditioned sparse matrix (COO)."""
    import scipy.sparse as sp
    np.random.seed(seed)
    A_coo = sp.random(n, n, density=density, format='coo', dtype=np.float64)
    diag = np.arange(n)
    A_coo.data = np.concatenate([A_coo.data, np.ones(n) * 10.0])
    A_coo.row = np.concatenate([A_coo.row, diag])
    A_coo.col = np.concatenate([A_coo.col, diag])
    return A_coo


def m100():
    """The 100x100 as CSC arrays (float64, int32, int32), n and b = A @ ones."""
    a = make_test_matrix().tocsc()
    a.sort_indices()
    data = np.ascontiguousarray(a.data, dtype=np.float64)
    indices = np.ascontiguousarray(a.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(a.indptr, dtype=np.int32)
    n = a.shape[0]
    b = a @ np.ones(n)
    return data, indices, indptr, n, np.ascontiguousarray(b, dtype=np.float64)


def n69():
    data, indices, indptr, b = common.read_n69()
    return data, indices, indptr, len(b), b


MATRICES = {"m100": m100, "n69": n69}


def new_values(data):
    """The same pattern, new values: every value scaled, deterministically."""
    return np.ascontiguousarray(data * 1.25 + 0.0, dtype=np.float64)


def csc_matvec(data, indices, indptr, x):
    y = np.zeros(len(indptr) - 1, dtype=np.float64)
    for j in range(len(indptr) - 1):
        for p in range(indptr[j], indptr[j + 1]):
            y[indices[p]] += data[p] * x[j]
    return y
