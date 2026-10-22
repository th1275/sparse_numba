"""
Test helpers: the fixed 60x60 system of the cache round trip, the
69-bus vehicle reader, and one pass through every public function of a
backend. Shared by make_oracle_0111.py, sn1_cache_child.py and
test_sn1_cache.py; numpy only (no scipy), so the child stays light.
"""

import importlib
import importlib.util
import os

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# THE SOURCE TREE the source-reading cells read (setup.py, the .pyx / .c
# sources, third_party/, vendor/): the checkout beside tests/, or
# SPARSE_NUMBA_SOURCE_DIR when tests/ runs beside an installed wheel
# (the tests run against a wheel). THE PACKAGE is imported from wherever Python
# finds it; IMPORT_ROOT, the directory holding it, is what the children are
# given (PYTHONPATH, cwd), so they import the same package as the cells.
REPO_DIR = os.path.abspath(os.environ.get("SPARSE_NUMBA_SOURCE_DIR")
                           or os.path.dirname(TESTS_DIR))
IMPORT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(importlib.util.find_spec("sparse_numba").origin)))
DATA_DIR = os.path.join(TESTS_DIR, "data")
ORACLE_PATH = os.path.join(DATA_DIR, "oracle_0111.npz")
# the 69-bus and 207-bus vehicles' files the cells read, copied byte for
# byte (tests/data/SHA256SUMS)
N69_DIR = os.path.join(DATA_DIR, "n69")
N207_DIR = os.path.join(DATA_DIR, "n207")

BACKENDS = {
    "superlu": "sparse_numba.sparse_superlu.superlu_numba_interface",
    "umfpack": "sparse_numba.sparse_umfpack.umfpack_numba_interface",
}

N_FIXED = 60
SEED_FIXED = 20261021


def fixed_system():
    """The fixed 60x60 diagonally dominant, unsymmetric system; dense A, b."""
    rng = np.random.default_rng(SEED_FIXED)
    n = N_FIXED
    a = np.zeros((n, n), dtype=np.float64)
    mask = rng.random((n, n)) < 0.08
    a[mask] = rng.uniform(-1.0, 1.0, size=int(mask.sum()))
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(a, np.abs(a).sum(axis=1) + 1.0 + rng.random(n))
    b = rng.uniform(-1.0, 1.0, size=n)
    return a, b


def dense_to_csc(a):
    n_rows, n_cols = a.shape
    data, indices, indptr = [], [], [0]
    for j in range(n_cols):
        for i in range(n_rows):
            if a[i, j] != 0.0:
                data.append(a[i, j])
                indices.append(i)
        indptr.append(len(data))
    return (np.array(data, dtype=np.float64), np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32))


def dense_to_csr(a):
    data, indices, indptr = dense_to_csc(a.T)
    return data, indices, indptr


def dense_to_coo(a):
    """Row-major COO (row, col, data), so the COO paths run their conversion."""
    rows, cols = np.nonzero(a)
    return (rows.astype(np.int32), cols.astype(np.int32),
            a[rows, cols].astype(np.float64))


def _read_mtx_lines(path):
    with open(path, "r", encoding="ascii") as f:
        lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("%")]
    return lines


def read_n69():
    """tests/data/n69 G.mtx (coordinate, explicit zeros kept) as CSC, and b.mtx."""
    lines = _read_mtx_lines(os.path.join(N69_DIR, "G.mtx"))
    n_rows, n_cols, nnz = (int(t) for t in lines[0].split())
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    vals = np.empty(nnz, dtype=np.float64)
    for k, ln in enumerate(lines[1:1 + nnz]):
        i, j, v = ln.split()
        rows[k], cols[k], vals[k] = int(i) - 1, int(j) - 1, float(v)
    order = np.lexsort((rows, cols))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    np.add.at(indptr, cols + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)
    blines = _read_mtx_lines(os.path.join(N69_DIR, "b.mtx"))
    b = np.array([float(t) for t in blines[1:]], dtype=np.float64)
    assert n_rows == n_cols == len(b)
    return vals, rows.astype(np.int32), indptr, b


def run_all(backend, a, b):
    """x from every public function of the backend on (a, b); every info 0."""
    mod = importlib.import_module(BACKENDS[backend])
    f = {name: getattr(mod, backend + "_" + name) for name in (
        "solve_csc", "solve_coo", "solve_csr",
        "factorize_csc", "factorize_coo", "factorize_csr",
        "solve_factored", "free_factors")}
    csc = dense_to_csc(a)
    csr = dense_to_csr(a)
    row, col, val = dense_to_coo(a)
    shape = (a.shape[0], a.shape[1])
    out = {}

    def keep(key, x, info):
        if info != 0:
            raise RuntimeError(f"{backend} {key}: info {info}")
        out[key] = np.asarray(x)

    keep("solve_csc", *f["solve_csc"](csc[0], csc[1], csc[2], b))
    keep("solve_coo", *f["solve_coo"](row, col, val, shape, b))
    keep("solve_csr", *f["solve_csr"](csr[0], csr[1], csr[2], b))
    for fmt, args in (("csc", csc), ("coo", (row, col, val, shape)), ("csr", csr)):
        handle, info = f["factorize_" + fmt](*args)
        if info != 0:
            raise RuntimeError(f"{backend} factorize_{fmt}: info {info}")
        keep("factored_" + fmt, *f["solve_factored"](handle, b))
        info = f["free_factors"](handle)
        if info != 0:
            raise RuntimeError(f"{backend} free_factors after {fmt}: info {info}")
    return out


def run_n69_superlu():
    """The 69-bus G at SuperLU NATURAL (the wrapper's fixed ColPerm)."""
    mod = importlib.import_module(BACKENDS["superlu"])
    data, indices, indptr, b = read_n69()
    out = {}
    x, info = mod.superlu_solve_csc(data, indices, indptr, b)
    if info != 0:
        raise RuntimeError(f"n69 superlu_solve_csc: info {info}")
    out["solve_csc"] = np.asarray(x)
    handle, info = mod.superlu_factorize_csc(data, indices, indptr)
    if info != 0:
        raise RuntimeError(f"n69 superlu_factorize_csc: info {info}")
    x, info = mod.superlu_solve_factored(handle, b)
    if info != 0:
        raise RuntimeError(f"n69 superlu_solve_factored: info {info}")
    out["factored_csc"] = np.asarray(x)
    mod.superlu_free_factors(handle)
    return out


def bits_equal(x, y):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return x.shape == y.shape and x.tobytes() == y.tobytes()
