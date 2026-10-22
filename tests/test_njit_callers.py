"""
NJIT CALLER cells: the user's own njit function calling the interface.

- one cell per backend: a @njit(nogil=True) caller defined here calls
  factorize, handle_info, solve_into, solve_many_into (k = 3), refactorize
  (values x 1.5), solve_into, free, then solve_into after the free; each
  info, handle_info's n / backend / ordering and each relative residual
  (<= RESIDUAL_TOL) asserted;
- the cache cell: njit_callers_child.py's @njit(cache=True) caller makes the
  same calls on SUPERLU in two child interpreters on one fresh
  NUMBA_CACHE_DIR; the second loads it (no cache file added or rewritten)
  and the x are bit-equal.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from numba import njit

import sparse_numba as sn
from sparse_numba.errors import HANDLE_FREED, OK

import _sn1_common as common
import _sn2_common as c2

RESIDUAL_TOL = 1e-10
K_RHS = 3
REFACTOR_SCALE = 1.5
CHILD_SCRIPT = os.path.join(common.TESTS_DIR, "njit_callers_child.py")


@njit(nogil=True)
def user_caller(data, indices, indptr, n, backend, ordering, b, B, data2):
    infos = np.zeros(8, dtype=np.int64)
    x1 = np.empty(n)
    X = np.empty_like(B)
    x2 = np.empty(n)
    x3 = np.empty(n)
    handle, info = sn.factorize(data, indices, indptr, n, backend, ordering)
    infos[0] = info
    n_, nnz_l, nnz_u, backend_, ordering_, last_status, pivot_tol, info = sn.handle_info(handle)
    infos[1] = info
    infos[2] = sn.solve_into(handle, b, x1)
    infos[3] = sn.solve_many_into(handle, B, X)
    infos[4] = sn.refactorize(handle, data2)
    infos[5] = sn.solve_into(handle, b, x2)
    infos[6] = sn.free(handle)
    infos[7] = sn.solve_into(handle, b, x3)
    meta = np.array([n_, backend_, ordering_], dtype=np.int64)
    return infos, meta, x1, X, x2


def _residual(a, x, b):
    return np.linalg.norm(a @ x - b, np.inf) / np.linalg.norm(b, np.inf)


CASES = {
    "superlu": (sn.SUPERLU, sn.COLAMD),
    "umfpack": (sn.UMFPACK, sn.AMD),
    "eigen_lu": (sn.EIGEN_LU, sn.COLAMD),
    "eigen_ldlt": (sn.EIGEN_LDLT, sn.AMD),
}


@pytest.mark.parametrize("backend", list(CASES))
def test_njit_caller(backend):
    c2.require_backend(backend)
    backend_id, ordering_id = CASES[backend]
    a, b = c2.system_for(backend)
    n = a.shape[0]
    data, indices, indptr = common.dense_to_csc(a)
    rng = np.random.default_rng(common.SEED_FIXED + 1)
    B = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n, K_RHS)))
    data2 = data * REFACTOR_SCALE

    infos, meta, x1, X, x2 = user_caller(data, indices, indptr, n, backend_id, ordering_id,
                                         b, B, data2)

    assert list(infos[:7]) == [OK] * 7, infos
    assert infos[7] == HANDLE_FREED, infos
    assert list(meta) == [n, backend_id, ordering_id], meta
    assert _residual(a, x1, b) <= RESIDUAL_TOL
    for j in range(K_RHS):
        assert _residual(a, X[:, j], B[:, j]) <= RESIDUAL_TOL, j
    assert _residual(a * REFACTOR_SCALE, x2, b) <= RESIDUAL_TOL


def _run_child(cache_dir, out):
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = (common.IMPORT_ROOT + os.pathsep + common.TESTS_DIR + os.pathsep
                         + env.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, CHILD_SCRIPT, str(out)], cwd=common.IMPORT_ROOT,
                          env=env, capture_output=True, text=True, timeout=600)


def _snapshot(cache_dir):
    snap = {}
    for root, _, files in os.walk(cache_dir):
        for name in files:
            if name.endswith((".nbi", ".nbc")):
                path = os.path.join(root, name)
                snap[os.path.relpath(path, cache_dir)] = os.stat(path).st_mtime_ns
    return snap


def test_njit_caller_cache_two_processes(tmp_path):
    c2.require_backend("superlu")
    cache_dir = tmp_path / "numba_cache"
    cache_dir.mkdir()
    proc1 = _run_child(cache_dir, tmp_path / "child1.npz")
    assert proc1.returncode == 0, proc1.stderr[-4000:]
    snap1 = _snapshot(cache_dir)
    proc2 = _run_child(cache_dir, tmp_path / "child2.npz")
    assert proc2.returncode == 0, proc2.stderr[-4000:]
    snap2 = _snapshot(cache_dir)

    assert "Cannot cache" not in proc1.stderr + proc2.stderr
    assert any(k.endswith(".nbi") and "njit_callers_child" in k for k in snap1), sorted(snap1)
    assert snap2 == snap1
    r1 = np.load(tmp_path / "child1.npz")
    r2 = np.load(tmp_path / "child2.npz")
    assert list(r1["infos"]) == [OK] * 7 + [HANDLE_FREED], r1["infos"]
    assert list(r2["infos"]) == list(r1["infos"])
    for k in ("x1", "X", "x2"):
        assert common.bits_equal(r1[k], r2[k]), k
