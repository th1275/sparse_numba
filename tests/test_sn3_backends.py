"""
Backend cells (SuperLU, UMFPACK: the orderings, the build facts, UMFPACK's runtime load).

- the supported orderings per backend on the 100x100 and the 69-bus G:
  SuperLU NATURAL / COLAMD / MMD_AT_PLUS_A (AMD refused), UMFPACK
  NATURAL (ORDERING_NONE) / COLAMD / AMD (MMD_AT_PLUS_A refused); every x
  within 1e-10 relative of the 0.1.11 SuperLU NATURAL x;
  handle_info names the ordering and the pivot tolerance 1.0;
- refactorize per ordering against a fresh factorize on the new values;
- solve_many_into widening the held buffer (k = 1, then 3, then 1);
- SINGULAR = k > 0 at every ordering; the codes contract (OK 0, every
  refusal negative) and its statement in errors.py, api.py and both kept
  modules;
- cache_key() naming the SuperLU version and build, and the UMFPACK
  library's version (or absent);
- UMFPACK interface-only: the extension imports without
  SuiteSparse; the library not found -> BACKEND_ABSENT with a silent
  import; found through SPARSE_NUMBA_UMFPACK_LIBRARY; a bad path there ->
  BACKEND_ABSENT (children, sn3_umfpack_child.py).

No cell carries an xfail marker.
"""

import importlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import _sn1_common as common
import _sn2_common as c2


UMFPACK_CHILD = os.path.join(common.TESTS_DIR, "sn3_umfpack_child.py")

# the supported orderings per backend and the development part each opened
# at (None: open from the start; no marker is applied any more)
SUPPORTED = {
    "superlu": {"NATURAL": None, "COLAMD": None, "MMD_AT_PLUS_A": None},
    "umfpack": {"NATURAL": None, "COLAMD": None, "AMD": None},
}
REFUSED = {"superlu": ("AMD",), "umfpack": ("MMD_AT_PLUS_A",)}


def _part(n):
    return pytest.mark.xfail(strict=True, reason=f"SN-3 part {n}")


def _params(matrices=None):
    out = []
    for backend, orderings in SUPPORTED.items():
        for ordering, part in orderings.items():
            for matrix in (matrices or [None]):
                values = (backend, ordering) if matrix is None else (backend, ordering, matrix)
                marks = [] if part is None else [_part(part)]
                out.append(pytest.param(*values, marks=marks))
    return out


def _api():
    import sparse_numba.api as api
    return api


def _rel(x, ref):
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def _reference_x(matrix, data, indices, indptr, b):
    """The 0.1.11 SuperLU NATURAL x: the oracle's on n69, the kept
    superlu_solve_csc's on the 100x100 (bit-equal to 0.1.11 by the oracle
    cells)."""
    if matrix == "n69":
        return np.load(common.ORACLE_PATH)["n69_superlu__factored_csc"]
    mod = importlib.import_module(common.BACKENDS["superlu"])
    x, info = mod.superlu_solve_csc(data, indices, indptr, b)
    assert info == 0
    return x


def _child_env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = common.IMPORT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


# ------------------------------------------------------------- the orderings

@pytest.mark.parametrize("backend, ordering, matrix", _params(["m100", "n69"]))
def test_ordering_x(backend, ordering, matrix):
    c2.require_backend(backend)
    api = _api()
    be, order = getattr(api, backend.upper()), getattr(api, ordering)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()
    ref = _reference_x(matrix, data, indices, indptr, b)

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK and handle > 0, info
    try:
        x = np.empty(n)
        assert api.solve_into(handle, b, x) == api.OK
        assert _rel(x, ref) <= 1e-10
        hi = api.handle_info(handle)
        assert hi[-1] == api.OK
        assert hi[0] == n and hi[1] > 0 and hi[2] > 0
        assert hi[3] == be and hi[4] == order and hi[5] == 0
        assert hi[6] == 1.0
    finally:
        assert api.free(handle) == api.OK
    x1, info = api.solve(data, indices, indptr, n, b, be, order)
    assert info == api.OK and common.bits_equal(x1, x)


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_orderings_refused(backend):
    c2.require_backend(backend)
    api = _api()
    be = getattr(api, backend.upper())
    data, indices, indptr, n, b = c2.m100()
    for ordering in REFUSED[backend]:
        assert api.factorize(data, indices, indptr, n, be, getattr(api, ordering)) \
            == (0, api.ORDERING_UNSUPPORTED), ordering


@pytest.mark.parametrize("backend, ordering, matrix", _params(["m100", "n69"]))
def test_refactorize_per_ordering(backend, ordering, matrix):
    c2.require_backend(backend)
    api = _api()
    be, order = getattr(api, backend.upper()), getattr(api, ordering)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()
    new = c2.new_values(data)

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    h2, info2 = api.factorize(new, indices, indptr, n, be, order)
    assert info2 == api.OK
    try:
        assert api.refactorize(handle, new) == api.OK
        x_re = np.empty(n)
        x_fresh = np.empty(n)
        assert api.solve_into(handle, b, x_re) == api.OK
        assert api.solve_into(h2, b, x_fresh) == api.OK
        if backend == "superlu" and ordering == "NATURAL":
            assert common.bits_equal(x_re, x_fresh)
        else:
            assert _rel(x_re, x_fresh) <= 1e-12
        assert api.handle_info(handle)[4] == order
    finally:
        assert api.free(handle) == api.OK
        assert api.free(h2) == api.OK


@pytest.mark.parametrize("backend, ordering", _params())
def test_solve_many_widening(backend, ordering):
    """k = 1, then 3, then 1 on one handle: the held right-hand side grows
    to n * 3 and back to one column without moving a solution."""
    c2.require_backend(backend)
    api = _api()
    be, order = getattr(api, backend.upper()), getattr(api, ordering)
    data, indices, indptr, n, b = c2.n69()

    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    try:
        x0 = np.empty(n)
        assert api.solve_into(handle, b, x0) == api.OK
        B1 = np.ascontiguousarray(b.reshape(n, 1))
        OUT1 = np.empty((n, 1))
        assert api.solve_many_into(handle, B1, OUT1) == api.OK
        assert common.bits_equal(OUT1[:, 0], x0)

        B3 = np.ascontiguousarray(np.stack([b, 2.0 * b, b + 1.0], axis=1))
        OUT3 = np.empty((n, 3))
        assert api.solve_many_into(handle, B3, OUT3) == api.OK
        for j in range(3):
            xj = np.empty(n)
            assert api.solve_into(handle, np.ascontiguousarray(B3[:, j]), xj) == api.OK
            assert _rel(OUT3[:, j], xj) <= 1e-12, j

        OUT1b = np.empty((n, 1))
        assert api.solve_many_into(handle, B1, OUT1b) == api.OK
        assert common.bits_equal(OUT1b[:, 0], x0)
        x1 = np.empty(n)
        assert api.solve_into(handle, b, x1) == api.OK
        assert common.bits_equal(x1, x0)
    finally:
        assert api.free(handle) == api.OK


# ------------------------------------------------------------- the codes

def _singular_3x3():
    data = np.array([1.0, 1.0, 0.0])
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    return data, indices, indptr, 3


@pytest.mark.parametrize("backend, ordering", _params())
def test_singular_per_ordering(backend, ordering):
    c2.require_backend(backend)
    api = _api()
    be, order = getattr(api, backend.upper()), getattr(api, ordering)
    data, indices, indptr, n = _singular_3x3()
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert handle == 0
    assert 1 <= info <= n                               # SINGULAR = k > 0


def test_codes_contract_values():
    from sparse_numba import errors
    assert errors.OK == 0
    names = ("BAD_DTYPE", "BAD_SHAPE", "BAD_INDPTR", "BAD_INDEX", "NOT_SQUARE",
             "HANDLE_INVALID", "HANDLE_FREED", "HANDLE_WRONG_BACKEND", "BACKEND_ABSENT",
             "ORDERING_UNSUPPORTED", "ALLOC_FAILED", "BACKEND_ERROR")
    for name in names:
        assert getattr(errors, name) < 0, name
    assert "SINGULAR" in errors.explain(1)


@pytest.mark.parametrize("module", [
    "sparse_numba.errors", "sparse_numba.api",
    "sparse_numba.sparse_superlu.superlu_numba_interface",
    "sparse_numba.sparse_umfpack.umfpack_numba_interface"])
def test_codes_contract_stated(module):
    """The codes contract: 0 success, a negative code a refusal or a
    backend error, a positive k SINGULAR; `info != 0` is failure."""
    doc = " ".join(importlib.import_module(module).__doc__.split())
    assert "0 success" in doc, module
    assert "negative code" in doc, module
    assert "positive k SINGULAR" in doc, module
    assert "info != 0" in doc, module


# ------------------------------------------------------------- cache_key()

def test_cache_key_names_superlu():
    from sparse_numba import cache_key
    key = cache_key()
    assert re.search(r"(^|;)superlu=\d+\.\d+\.\d+,", key), key
    assert re.search(r"(^|;)superlu=[^;]*cc=", key), key


def test_cache_key_names_suitesparse():
    from sparse_numba import cache_key
    key = cache_key()
    assert re.search(r"(^|;)umfpack=(\d+\.\d+\.\d+|unknown)(;|$)", key), key


# ------------------------------------------------------------- 7.2 runtime load

def _run_umfpack_child(mode, tmp_path, **env):
    out = tmp_path / f"umfpack_child_{mode}.json"
    proc = subprocess.run([sys.executable, UMFPACK_CHILD, mode, str(out)],
                          cwd=common.IMPORT_ROOT, env=_child_env(**env),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-4000:])
    assert proc.stdout == ""
    assert proc.stderr == ""
    with open(out, "r", encoding="ascii") as f:
        return json.load(f)


def test_umfpack_library_not_found(tmp_path):
    if sys.platform != "win32":
        pytest.skip("the hidden-library child is the Windows layout")
    facts = _run_umfpack_child("hidden", tmp_path)
    assert facts["umfpack_module_imported"] is True
    assert facts["is_umf_available"] is False
    # UMFPACK absent, SuperLU present (the Eigen kinds are 3 and 4)
    assert 1 in facts["backends"] and 2 not in facts["backends"]
    assert facts["umfpack_factorize_info"] == -20
    assert facts["kept_umfpack_factorize_info"] == -20
    assert facts["superlu_factorize_info"] == 0
    assert "umfpack=absent" in facts["cache_key"]


def test_umfpack_library_by_env_path(tmp_path):
    if sys.platform != "win32":
        pytest.skip("the vendored test library is the Windows layout")
    facts = _run_umfpack_child("envvar", tmp_path)
    assert facts["umfpack_module_imported"] is True
    assert facts["is_umf_available"] is True
    assert 1 in facts["backends"] and 2 in facts["backends"]
    assert facts["umfpack_factorize_info"] == 0
    ref = np.load(common.ORACLE_PATH)["umfpack__factored_csc"]
    assert common.bits_equal(np.array(facts["x_fixed"]), ref)
    assert re.search(r"umfpack=\d+\.\d+\.\d+", facts["cache_key"]), facts["cache_key"]


def test_umfpack_library_bad_env_path(tmp_path):
    facts = _run_umfpack_child("badpath", tmp_path)
    assert facts["umfpack_module_imported"] is True
    assert facts["is_umf_available"] is False
    assert facts["umfpack_factorize_info"] == -20
    assert facts["superlu_factorize_info"] == 0
