"""
API cells (the backend-neutral API, the handle registry, the KEPT names).

- (a) the round trip per backend at its reference ordering (SuperLU NATURAL,
  UMFPACK AMD): factorize / solve_into / solve_many_into / handle_info /
  refactorize / free on the 100x100 of test_prefactorize_slu.py :22 and on
  the 69-bus G; solve_many_into against k solve_into's; refactorize against
  a fresh factorize on the new values (bit-equal at SuperLU NATURAL);
- (a) (f) the kept 0.1.11 names bit-equal to the new calls and to
  tests/data/oracle_0111.npz; on n69 every such backend within 1e-10
  relative of the 0.1.11 SuperLU NATURAL x;
- (b) the cache round trip of the new API in two children (sn2_cache_child.py);
- the python-level surface: solve, Factors, explain, SparseNumbaError,
  require, backends, is_eigen_available, the enums.

No cell carries an xfail marker.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import _sn1_common as common
import _sn2_common as c2


CHILD2_FRACTION = 0.5
CHILD_SCRIPT = os.path.join(common.TESTS_DIR, "sn2_cache_child.py")


def _api():
    import sparse_numba.api as api
    return api


def _oracle():
    return np.load(common.ORACLE_PATH)


def _rel(x, ref):
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


# ------------------------------------------------------------- (a) round trip

@pytest.mark.parametrize("matrix", ["m100", "n69"])
@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_round_trip(backend, matrix):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK and handle > 0, info
    try:
        x = np.empty(n)
        assert api.solve_into(handle, b, x) == api.OK
        if matrix == "m100":
            r = c2.csc_matvec(data, indices, indptr, x) - b
            assert np.max(np.abs(r)) <= 1e-10 * np.max(np.abs(b))
        else:
            assert _rel(x, _oracle()["n69_superlu__factored_csc"]) <= 1e-10

        # solve_many_into against k solve_into's
        k = 3
        B = np.ascontiguousarray(np.stack([b, 2.0 * b, b + 1.0], axis=1))
        OUT = np.empty((n, k))
        assert api.solve_many_into(handle, B, OUT) == api.OK
        for j in range(k):
            xj = np.empty(n)
            assert api.solve_into(handle, np.ascontiguousarray(B[:, j]), xj) == api.OK
            assert _rel(OUT[:, j], xj) <= 1e-12, j

        # handle_info: (n, nnz_L, nnz_U, backend_id, ordering_id, last_status,
        #               pivot_tolerance, info)
        hi = api.handle_info(handle)
        assert hi[-1] == api.OK
        assert hi[0] == n and hi[1] > 0 and hi[2] > 0
        assert hi[3] == be and hi[4] == order and hi[5] == 0
        assert hi[6] == 1.0

        # refactorize against a fresh factorize on the new values
        new = c2.new_values(data)
        assert api.refactorize(handle, new) == api.OK
        x_re = np.empty(n)
        assert api.solve_into(handle, b, x_re) == api.OK
        h2, info2 = api.factorize(new, indices, indptr, n, be, order)
        assert info2 == api.OK
        x_fresh = np.empty(n)
        assert api.solve_into(h2, b, x_fresh) == api.OK
        assert api.free(h2) == api.OK
        if backend == "superlu":
            assert common.bits_equal(x_re, x_fresh)
        else:
            assert _rel(x_re, x_fresh) <= 1e-12
    finally:
        assert api.free(handle) == api.OK
    assert api.free(handle) == api.HANDLE_FREED


# --------------------------------------- (a) (f) the kept names and the oracle

@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_new_calls_equal_oracle_fixed_system(backend):
    """The fixed 60x60 through the new calls, bit-equal to the 0.1.11 x."""
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    a, b = common.fixed_system()
    data, indices, indptr = common.dense_to_csc(a)
    n = a.shape[0]
    ref = _oracle()[f"{backend}__factored_csc"]

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    x = np.empty(n)
    assert api.solve_into(handle, b, x) == api.OK
    assert api.free(handle) == api.OK
    assert common.bits_equal(x, ref)

    x1, info = api.solve(data, indices, indptr, n, b, be, order)
    assert info == api.OK
    assert common.bits_equal(x1, ref)


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
@pytest.mark.parametrize("matrix", ["m100", "n69"])
def test_kept_names_equal_new_calls(backend, matrix):
    """superlu_* / umfpack_* (0.1.11 names) bit-equal to the new calls."""
    c2.require_backend(backend)
    import importlib
    api = _api()
    mod = importlib.import_module(common.BACKENDS[backend])
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    x_new = np.empty(n)
    assert api.solve_into(handle, b, x_new) == api.OK
    assert api.free(handle) == api.OK

    h_old, info = getattr(mod, backend + "_factorize_csc")(data, indices, indptr)
    assert info == 0
    x_old, info = getattr(mod, backend + "_solve_factored")(h_old, b)
    assert info == 0
    assert getattr(mod, backend + "_free_factors")(h_old) == 0
    assert common.bits_equal(x_old, x_new)

    x_one, info = getattr(mod, backend + "_solve_csc")(data, indices, indptr, b)
    assert info == 0
    assert common.bits_equal(x_one, x_new)
    if matrix == "n69" and backend == "superlu":
        assert common.bits_equal(x_new, _oracle()["n69_superlu__factored_csc"])


# ------------------------------------------------- (b) the new API's cache trip

def _run_child(backend, cache_dir, out):
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = common.IMPORT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, CHILD_SCRIPT, backend, str(out)],
                          cwd=common.IMPORT_ROOT, env=env, capture_output=True, text=True,
                          timeout=600)
    return proc, time.perf_counter() - t0


def _snapshot(cache_dir):
    snap = {}
    for root, _, files in os.walk(cache_dir):
        for name in files:
            if name.endswith((".nbi", ".nbc")):
                path = os.path.join(root, name)
                snap[os.path.relpath(path, cache_dir)] = os.stat(path).st_mtime_ns
    return snap


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_api_cache_round_trip(backend, tmp_path):
    c2.require_backend(backend)
    cache_dir = tmp_path / "numba_cache"
    cache_dir.mkdir()
    proc1, proc_wall1 = _run_child(backend, cache_dir, tmp_path / "child1.npz")
    snap1 = _snapshot(cache_dir)
    proc2, proc_wall2 = _run_child(backend, cache_dir, tmp_path / "child2.npz")
    snap2 = _snapshot(cache_dir)

    record = {"backend": backend, "returncode": [proc1.returncode, proc2.returncode],
              "process_wall_s": [proc_wall1, proc_wall2], "cache_files": snap1,
              "cache_files_after_child2": snap2,
              "stderr": [proc1.stderr[-4000:], proc2.stderr[-4000:]],
              "stdout": [proc1.stdout[-4000:], proc2.stdout[-4000:]]}
    if proc1.returncode == 0 and proc2.returncode == 0:
        c1 = np.load(tmp_path / "child1.npz")
        c2_ = np.load(tmp_path / "child2.npz")
        record["calls_wall_s"] = [float(c1["wall_calls"]), float(c2_["wall_calls"])]
        record["import_wall_s"] = [float(c1["wall_import"]), float(c2_["wall_import"])]
    with open(tmp_path / "round_trip_record.json", "w", encoding="ascii") as f:
        json.dump(record, f, indent=1)

    assert proc1.returncode == 0, proc1.stderr[-4000:]
    assert proc2.returncode == 0, proc2.stderr[-4000:]
    assert "Cannot cache" not in proc1.stderr, proc1.stderr[-4000:]
    assert "Cannot cache" not in proc2.stderr, proc2.stderr[-4000:]
    names = [os.path.basename(k) for k in snap1]
    assert any(k.endswith(".nbi") and k.startswith("api.") for k in names), sorted(names)
    assert any(k.endswith(".nbi") and k.startswith("sn2_cache_child.") for k in names), \
        sorted(names)
    assert snap2 == snap1
    wall1, wall2 = record["calls_wall_s"]
    assert wall2 < CHILD2_FRACTION * wall1, (wall1, wall2)
    # x bit-equal across the children; the factored x's bit-equal to the
    # 0.1.11 oracle (UMFPACK's refactorized x within 1e-12, as in (a))
    ref = _oracle()[f"{backend}__factored_csc"]
    keys = sorted(k for k in c1.files if k.startswith(("x__", "m__")))
    assert keys == ["m__solve_many_into", "x__caller", "x__refactorized", "x__solve",
                    "x__solve_into"], keys
    for k in keys:
        assert common.bits_equal(c1[k], c2_[k]), k
    for k in ("x__caller", "x__solve", "x__solve_into"):
        assert common.bits_equal(c1[k], ref), k
    if backend == "superlu":
        assert common.bits_equal(c1["x__refactorized"], ref)
    else:
        assert _rel(c1["x__refactorized"], ref) <= 1e-12


# ---------------------------------------------------- the python-level surface

def test_enums_and_backends():
    api = _api()
    assert (api.SUPERLU, api.UMFPACK, api.EIGEN_LU, api.EIGEN_LDLT) == (1, 2, 3, 4)
    assert (api.NATURAL, api.COLAMD, api.AMD, api.MMD_AT_PLUS_A) == (0, 1, 2, 3)
    ids = api.backends()
    assert api.SUPERLU in ids
    # the Eigen kinds: tests/test_sn4_eigen.py; an unknown id
    # stays absent
    assert 99 not in ids
    with pytest.raises(api.SparseNumbaError) as e:
        api.require(99)
    assert e.value.info == api.BACKEND_ABSENT
    api.require(api.SUPERLU)


def test_explain_every_code():
    api = _api()
    codes = [api.OK, api.BAD_DTYPE, api.BAD_SHAPE, api.BAD_INDPTR, api.BAD_INDEX,
             api.NOT_SQUARE, api.HANDLE_INVALID, api.HANDLE_FREED,
             api.HANDLE_WRONG_BACKEND, api.BACKEND_ABSENT, api.ORDERING_UNSUPPORTED,
             api.ALLOC_FAILED, api.BACKEND_ERROR, 7]
    texts = [api.explain(c) for c in codes]
    assert all(isinstance(t, str) and t for t in texts)
    assert len(set(texts)) == len(texts)
    assert "7" in api.explain(7)
    err = api.SparseNumbaError(api.HANDLE_FREED)
    assert err.info == api.HANDLE_FREED and api.explain(api.HANDLE_FREED) in str(err)


def test_factors_context_manager():
    api = _api()
    data, indices, indptr, n, b = c2.m100()
    with api.Factors(data, indices, indptr, n, api.SUPERLU, api.NATURAL) as f:
        x = f.solve(b)
        handle = f.handle
    ref, info = api.solve(data, indices, indptr, n, b, api.SUPERLU, api.NATURAL)
    assert info == api.OK and common.bits_equal(x, ref)
    assert api.free(handle) == api.HANDLE_FREED
    with pytest.raises(api.SparseNumbaError) as e:
        api.Factors(data, indices, indptr, n + 1, api.SUPERLU, api.NATURAL)
    assert e.value.info == api.NOT_SQUARE


def test_symbols_public_module():
    api = _api()
    from sparse_numba import symbols, _symbols
    assert symbols.TABLE == _symbols.TABLE
    for name, _, _, sig in symbols.TABLE:
        assert str(symbols.declared(name)) == sig
    assert api.cache_key() == _symbols.cache_key()
