"""
Eigen cells (EIGEN_LU and EIGEN_LDLT; the MPL2-only build; Eigen's
licence).

- part 2: one registry header shared by the shims (sparse_numba_registry.h);
  setup.py's compile flags as one list, their text compiled into the
  SuperLU shim (-DSN_BUILD_FLAGS) and named by cache_key();
- part 3: third_party/eigen-3.4.0/ (Eigen/, COPYING.*, README.md; no
  unsupported/) with a SHA256SUMS of every copied file, the tarball's
  sha256 at its head;
- part 4: EIGEN_LU at NATURAL / COLAMD / AMD (MMD_AT_PLUS_A refused) on the
  100x100 and the 69-bus G, every x within 1e-10 relative of the 0.1.11
  SuperLU NATURAL x; refactorize; solve_many_into (k = 1, 3, 1); SINGULAR
  k > 0; EIGEN_LDLT at AMD only on tests/data/n207 S.mtx with b_s.mtx,
  x within 1e-10 of x_C5.txt (solve_gap_0's LDLT x) and of SuperLU
  NATURAL; an unsymmetric pattern -> BAD_SHAPE, a symmetric matrix with a
  zero pivot -> SINGULAR, the other orderings refused; handle_info per
  kind (the nonzeros handle_info reports); the handle guards; the Eigen rows of the
  symbol table; the cache round trip of the new API per Eigen kind
  (sn2_cache_child.py); cache_key() naming eigen=;
  is_eigen_available() / backends(); no printf outside SPARSE_NUMBA_DEBUG
  in the new C / C++ sources.

No cell carries an xfail marker.
"""

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

import _sn1_common as common
import _sn2_common as c2


PKG = os.path.join(common.REPO_DIR, "sparse_numba")
REGISTRY_HEADER = os.path.join(PKG, "sparse_numba_registry.h")
EIGEN_DIR = os.path.join(PKG, "sparse_eigen")
THIRD_PARTY = os.path.join(common.REPO_DIR, "third_party", "eigen-3.4.0")
TARBALL_SHA256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
N207_DIR = common.N207_DIR
CHILD_SCRIPT = os.path.join(common.TESTS_DIR, "sn2_cache_child.py")
CHILD2_FRACTION = 0.5

LU_ORDERINGS = ("NATURAL", "COLAMD", "AMD")
EIGEN_ROWS = (
    "sparse_numba_eigen_lu_factorize_ordered",
    "sparse_numba_eigen_ldlt_factorize_ordered",
    "sparse_numba_eigen_refactorize",
    "sparse_numba_eigen_solve_into",
    "sparse_numba_eigen_handle_info",
    "sparse_numba_eigen_free",
)


def _api():
    import sparse_numba.api as api
    return api


def _read(path):
    with open(path, "r", encoding="ascii") as f:
        return f.read()


def _rel(x, ref):
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def _reference_x(matrix, data, indices, indptr, b):
    """The 0.1.11 SuperLU NATURAL x: the oracle's on n69, the kept
    superlu_solve_csc's on the 100x100 (as tests/test_sn3_backends.py)."""
    if matrix == "n69":
        return np.load(common.ORACLE_PATH)["n69_superlu__factored_csc"]
    mod = importlib.import_module(common.BACKENDS["superlu"])
    x, info = mod.superlu_solve_csc(data, indices, indptr, b)
    assert info == 0
    return x


def _read_mtx_csc(path):
    """A MatrixMarket coordinate file (explicit zeros kept) as sorted CSC."""
    lines = [ln for ln in _read(path).splitlines() if ln and not ln.startswith("%")]
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
    assert n_rows == n_cols
    return (np.ascontiguousarray(vals), np.ascontiguousarray(rows.astype(np.int32)),
            np.ascontiguousarray(indptr), n_cols)


def _read_mtx_vector(path):
    lines = [ln for ln in _read(path).splitlines() if ln and not ln.startswith("%")]
    return np.array([float(t) for t in lines[1:]], dtype=np.float64)


def s207():
    """The n207 nodal S (621 x 621, symmetric positive definite), b_s and
    solve_gap_0's C5 (Eigen SimplicialLDLT) x."""
    data, indices, indptr, n = _read_mtx_csc(os.path.join(N207_DIR, "S.mtx"))
    b = _read_mtx_vector(os.path.join(N207_DIR, "b_s.mtx"))
    x_c5 = np.array([float(t) for t in _read(os.path.join(N207_DIR, "x_C5.txt")).split()])
    assert len(b) == n == len(x_c5)
    return data, indices, indptr, n, b, x_c5


# ================================================================ part 2

def test_registry_header_shared():
    """One registry header shared by the shims:
    the slot table is defined there and nowhere else."""
    text = _read(REGISTRY_HEADER)
    assert "sn_register" in text and "sn_lookup" in text and "sn_release" in text
    for path in (os.path.join(PKG, "sparse_superlu", "superlu_wrapper.c"),
                 os.path.join(PKG, "sparse_umfpack", "umfpack_wrapper.c")):
        src = _read(path)
        assert '#include "../sparse_numba_registry.h"' in src, path
        assert "static sn_slot_t *sn_chunks" not in src, path
        assert "static int sn_register(" not in src, path


def test_cache_key_build_flags():
    """setup.py's compile flags, their text compiled in (-DSN_BUILD_FLAGS),
    named in cache_key()'s superlu=."""
    from sparse_numba import cache_key
    key = cache_key()
    superlu = re.search(r"(^|;)superlu=([^;]*)", key).group(2)
    assert re.search(r",flags=[^;]*-O3", superlu), key
    setup_text = _read(os.path.join(common.REPO_DIR, "setup.py"))
    assert "SN_BUILD_FLAGS" in setup_text


def test_registry_header_no_printf():
    assert _printf_hits(REGISTRY_HEADER) == []


def _printf_hits(path):
    """printf outside #ifdef SPARSE_NUMBA_DEBUG (tests/test_sn2_contract.py's scan)."""
    debug_if = re.compile(r"#\s*(ifdef\s+SPARSE_NUMBA_DEBUG\b|"
                          r"if\s+defined\s*\(\s*SPARSE_NUMBA_DEBUG\s*\))")
    hits = []
    stack = []
    for k, line in enumerate(_read(path).splitlines(), 1):
        s = line.strip()
        if re.match(r"#\s*if", s):
            stack.append(bool(debug_if.match(s)))
            continue
        if re.match(r"#\s*(else|elif)\b", s):
            if stack:
                stack[-1] = False
            continue
        if re.match(r"#\s*endif\b", s):
            if stack:
                stack.pop()
            continue
        if ("printf" in line or "std::cout" in line or "std::cerr" in line) \
                and not any(stack):
            hits.append(f"{os.path.basename(path)}:{k}: {s}")
    return hits


# ================================================================ part 3

def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_third_party_manifest():
    """third_party/eigen-3.4.0/: Eigen/, COPYING.*, README.md copied from the
    sha-checked tarball, no unsupported/; SHA256SUMS lists every copied
    file, the tarball's sha256 at its head."""
    sums_path = os.path.join(THIRD_PARTY, "SHA256SUMS")
    with open(sums_path, "rb") as f:
        raw = f.read()
    assert b"\r" not in raw
    lines = raw.decode("ascii").splitlines()
    assert TARBALL_SHA256 in lines[0], lines[0]
    listed = {}
    for ln in lines[1:]:
        if not ln or ln.startswith("#"):
            continue
        sha, rel = ln.split(None, 1)
        listed[rel.strip()] = sha
    on_disk = []
    for root, dirs, files in os.walk(THIRD_PARTY):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), THIRD_PARTY).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                on_disk.append(rel)
    assert sorted(on_disk) == sorted(listed)
    for rel, sha in listed.items():
        assert _sha256(os.path.join(THIRD_PARTY, rel)) == sha, rel
    tops = {rel.split("/", 1)[0] for rel in on_disk}
    assert "unsupported" not in tops
    assert tops <= {"Eigen", "README.md"} | {t for t in tops if t.startswith("COPYING.")}
    for rel in ("Eigen/SparseLU", "Eigen/SparseCholesky", "Eigen/OrderingMethods",
                "COPYING.MPL2", "COPYING.README", "README.md"):
        assert rel in listed, rel


# ================================================================ part 4: EIGEN_LU

def _lu_params(matrices=None):
    out = []
    for ordering in LU_ORDERINGS:
        for matrix in (matrices or [None]):
            values = (ordering,) if matrix is None else (ordering, matrix)
            out.append(pytest.param(*values))
    return out


@pytest.mark.parametrize("ordering, matrix", _lu_params(["m100", "n69"]))
def test_lu_ordering_x(ordering, matrix):
    api = _api()
    order = getattr(api, ordering)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()
    ref = _reference_x(matrix, data, indices, indptr, b)

    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU, order)
    assert info == api.OK and handle > 0, info
    try:
        x = np.empty(n)
        assert api.solve_into(handle, b, x) == api.OK
        assert _rel(x, ref) <= 1e-10
        hi = api.handle_info(handle)
        assert hi[-1] == api.OK
        assert hi[0] == n and hi[1] > 0 and hi[2] > 0       # nnzL(), nnzU()
        assert hi[3] == api.EIGEN_LU and hi[4] == order and hi[5] == 0
        assert hi[6] == 1.0
    finally:
        assert api.free(handle) == api.OK
    x1, info = api.solve(data, indices, indptr, n, b, api.EIGEN_LU, order)
    assert info == api.OK and common.bits_equal(x1, x)


def test_lu_orderings_refused():
    api = _api()
    data, indices, indptr, n, b = c2.m100()
    for ordering in (api.MMD_AT_PLUS_A, 99):
        assert api.factorize(data, indices, indptr, n, api.EIGEN_LU, ordering) \
            == (0, api.ORDERING_UNSUPPORTED), ordering


@pytest.mark.parametrize("ordering, matrix", _lu_params(["m100", "n69"]))
def test_lu_refactorize(ordering, matrix):
    api = _api()
    order = getattr(api, ordering)
    data, indices, indptr, n, b = c2.MATRICES[matrix]()
    new = c2.new_values(data)

    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU, order)
    assert info == api.OK
    h2, info2 = api.factorize(new, indices, indptr, n, api.EIGEN_LU, order)
    assert info2 == api.OK
    try:
        assert api.refactorize(handle, new) == api.OK
        x_re = np.empty(n)
        x_fresh = np.empty(n)
        assert api.solve_into(handle, b, x_re) == api.OK
        assert api.solve_into(h2, b, x_fresh) == api.OK
        assert _rel(x_re, x_fresh) <= 1e-12
        assert api.handle_info(handle)[4] == order
    finally:
        assert api.free(handle) == api.OK
        assert api.free(h2) == api.OK


def _widening(api, handle, b, n):
    """k = 1, then 3, then 1 on one handle (tests/test_sn3_backends.py's cell)."""
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


@pytest.mark.parametrize("ordering", _lu_params())
def test_lu_solve_many_widening(ordering):
    api = _api()
    data, indices, indptr, n, b = c2.n69()
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU,
                                 getattr(api, ordering))
    assert info == api.OK
    try:
        _widening(api, handle, b, n)
    finally:
        assert api.free(handle) == api.OK


def _singular_3x3():
    """diag(1, 1, 0) with the (2, 2) entry an explicit zero."""
    data = np.array([1.0, 1.0, 0.0])
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    return data, indices, indptr, 3


@pytest.mark.parametrize("ordering", _lu_params())
def test_lu_singular(ordering):
    api = _api()
    data, indices, indptr, n = _singular_3x3()
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU,
                                 getattr(api, ordering))
    assert handle == 0
    assert 1 <= info <= n                               # SINGULAR = k > 0


# ================================================================ part 4: EIGEN_LDLT

def test_ldlt_s207():
    api = _api()
    data, indices, indptr, n, b, x_c5 = s207()
    x_slu, info = api.solve(data, indices, indptr, n, b, api.SUPERLU, api.NATURAL)
    assert info == api.OK

    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, api.AMD)
    assert info == api.OK and handle > 0, info
    try:
        x = np.empty(n)
        assert api.solve_into(handle, b, x) == api.OK
        assert _rel(x, x_c5) <= 1e-10
        assert _rel(x, x_slu) <= 1e-10
        hi = api.handle_info(handle)
        assert hi[-1] == api.OK
        assert hi[0] == n and hi[1] > 0                    # the L nonzeros
        assert hi[3] == api.EIGEN_LDLT and hi[4] == api.AMD and hi[5] == 0
        _widening(api, handle, b, n)
    finally:
        assert api.free(handle) == api.OK
    x1, info = api.solve(data, indices, indptr, n, b, api.EIGEN_LDLT, api.AMD)
    assert info == api.OK and common.bits_equal(x1, x)


def test_ldlt_refactorize():
    api = _api()
    data, indices, indptr, n, b, _ = s207()
    new = c2.new_values(data)
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, api.AMD)
    assert info == api.OK
    h2, info2 = api.factorize(new, indices, indptr, n, api.EIGEN_LDLT, api.AMD)
    assert info2 == api.OK
    try:
        assert api.refactorize(handle, new) == api.OK
        x_re = np.empty(n)
        x_fresh = np.empty(n)
        assert api.solve_into(handle, b, x_re) == api.OK
        assert api.solve_into(h2, b, x_fresh) == api.OK
        assert _rel(x_re, x_fresh) <= 1e-12
    finally:
        assert api.free(handle) == api.OK
        assert api.free(h2) == api.OK


def test_ldlt_unsymmetric_pattern_bad_shape():
    api = _api()
    data, indices, indptr, n, b = c2.m100()            # the pattern is unsymmetric
    assert api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, api.AMD) \
        == (0, api.BAD_SHAPE)
    x, info = api.solve(data, indices, indptr, n, b, api.EIGEN_LDLT, api.AMD)
    assert info == api.BAD_SHAPE


def test_ldlt_zero_pivot_singular():
    """A symmetric pattern, not positive definite: [[0, 1], [1, 0]] (both
    diagonal entries explicit zeros) -> info() != Success -> SINGULAR."""
    api = _api()
    data = np.array([0.0, 1.0, 1.0, 0.0])
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 4], dtype=np.int32)
    handle, info = api.factorize(data, indices, indptr, 2, api.EIGEN_LDLT, api.AMD)
    assert handle == 0
    assert info >= 1                                    # SINGULAR = k > 0


def test_ldlt_orderings_refused():
    api = _api()
    data, indices, indptr, n, b, _ = s207()
    for ordering in (api.NATURAL, api.COLAMD, api.MMD_AT_PLUS_A, 99):
        assert api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, ordering) \
            == (0, api.ORDERING_UNSUPPORTED), ordering


# ================================================================ part 4: guards, rows

def test_eigen_handle_guards():
    """A double free and a use after free -> HANDLE_FREED; an Eigen handle in
    a kept SuperLU / UMFPACK name -> HANDLE_WRONG_BACKEND; the interpreter
    alive after every one."""
    api = _api()
    slu = importlib.import_module(common.BACKENDS["superlu"])
    data, indices, indptr, n, b = c2.m100()
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU, api.COLAMD)
    assert info == api.OK
    assert (handle >> 24) & 0xFF == api.EIGEN_LU
    x, info = slu.superlu_solve_factored(handle, b)
    assert info == api.HANDLE_WRONG_BACKEND
    assert slu.superlu_free_factors(handle) == api.HANDLE_WRONG_BACKEND
    assert api.free(handle) == api.OK
    assert api.free(handle) == api.HANDLE_FREED
    out = np.empty(n)
    assert api.solve_into(handle, b, out) == api.HANDLE_FREED
    assert api.refactorize(handle, data) == api.HANDLE_FREED
    assert api.handle_info(handle)[-1] == api.HANDLE_FREED
    sd, si, sp, sn, sb, _ = s207()
    handle, info = api.factorize(sd, si, sp, sn, api.EIGEN_LDLT, api.AMD)
    assert info == api.OK
    assert (handle >> 24) & 0xFF == api.EIGEN_LDLT
    assert api.refactorize(handle, sd[:-1].copy()) == api.BAD_SHAPE
    assert api.free(handle) == api.OK
    assert api.free(handle) == api.HANDLE_FREED


def test_eigen_rows_registered():
    import llvmlite.binding as llb
    import sparse_numba.api  # noqa: F401  registers every row
    from sparse_numba import symbols
    rows = [r for r in symbols.TABLE if r[2] == "sparse_numba.sparse_eigen.cy_eigen_wrapper"]
    assert sorted(r[0] for r in rows) == sorted(EIGEN_ROWS)
    for name in EIGEN_ROWS:
        assert llb.address_of_symbol(name) is not None, name


def test_is_eigen_available_and_backends():
    api = _api()
    import sparse_numba
    assert api.is_eigen_available() is True
    assert sparse_numba.is_eigen_available() is True
    ids = api.backends()
    assert api.EIGEN_LU in ids and api.EIGEN_LDLT in ids and api.SUPERLU in ids
    api.require(api.EIGEN_LU)
    api.require(api.EIGEN_LDLT)


def test_cache_key_names_eigen():
    from sparse_numba import cache_key
    key = cache_key()
    m = re.search(r"(^|;)eigen=([^;]*)", key)
    assert m, key
    eigen = m.group(2)
    assert eigen.startswith("3.4.0,"), eigen
    assert "cc=" in eigen and "flags=" in eigen, eigen
    assert "-DEIGEN_MPL2_ONLY" in eigen and "-DEIGEN_DONT_PARALLELIZE" in eigen, eigen
    assert re.search(r"(^|;)cy_eigen_wrapper=[0-9a-f]{16}(;|$)", key), key


@pytest.mark.parametrize("name", ["eigen_wrapper.cpp", "eigen_wrapper.h"])
def test_eigen_sources_no_printf(name):
    assert _printf_hits(os.path.join(EIGEN_DIR, name)) == []


def test_eigen_interface_no_print():
    path = os.path.join(EIGEN_DIR, "eigen_numba_interface.py")
    assert [ln for ln in _read(path).splitlines() if "print(" in ln] == []


# ================================================================ part 4: the cache trip

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


@pytest.mark.parametrize("backend", ["eigen_lu", "eigen_ldlt"])
def test_eigen_api_cache_round_trip(backend, tmp_path):
    """The cache round trip for the Eigen kinds: two children on one NUMBA_CACHE_DIR."""
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
        r1 = np.load(tmp_path / "child1.npz")
        r2 = np.load(tmp_path / "child2.npz")
        record["calls_wall_s"] = [float(r1["wall_calls"]), float(r2["wall_calls"])]
        record["import_wall_s"] = [float(r1["wall_import"]), float(r2["wall_import"])]
    with open(tmp_path / "round_trip_record.json", "w", encoding="ascii") as f:
        json.dump(record, f, indent=1)

    assert proc1.returncode == 0, proc1.stderr[-4000:]
    assert proc2.returncode == 0, proc2.stderr[-4000:]
    assert "Cannot cache" not in proc1.stderr, proc1.stderr[-4000:]
    assert "Cannot cache" not in proc2.stderr, proc2.stderr[-4000:]
    assert proc1.stderr == "" and proc2.stderr == ""
    names = [os.path.basename(k) for k in snap1]
    assert any(k.endswith(".nbi") and k.startswith("api.") for k in names), sorted(names)
    assert any(k.endswith(".nbi") and k.startswith("sn2_cache_child.") for k in names), \
        sorted(names)
    assert snap2 == snap1
    wall1, wall2 = record["calls_wall_s"]
    assert wall2 < CHILD2_FRACTION * wall1, (wall1, wall2)
    keys = sorted(k for k in r1.files if k.startswith(("x__", "m__")))
    assert keys == ["m__solve_many_into", "x__caller", "x__refactorized", "x__solve",
                    "x__solve_into"], keys
    for k in keys:
        assert common.bits_equal(r1[k], r2[k]), k
    # every x within 1e-10 relative of SuperLU NATURAL on the child's system
    api = _api()
    a, b = c2.system_for(backend)
    data, indices, indptr = common.dense_to_csc(a)
    ref, info = api.solve(data, indices, indptr, a.shape[0], b, api.SUPERLU, api.NATURAL)
    assert info == api.OK
    if backend == "eigen_lu":
        assert common.bits_equal(ref, np.load(common.ORACLE_PATH)["superlu__factored_csc"])
    for k in ("x__caller", "x__solve", "x__solve_into", "x__refactorized"):
        assert _rel(r1[k], ref) <= 1e-10, k
    for k in ("x__caller", "x__solve"):
        assert common.bits_equal(r1[k], r1["x__solve_into"]), k
