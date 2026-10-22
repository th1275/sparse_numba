"""
CACHE cells (the cache round trip across processes; the declared signatures).

- the child round trip per backend (steps (1)-(4)): two child
  interpreters on one fresh NUMBA_CACHE_DIR; (2)'s stated fraction is
  CHILD2_FRACTION of child 1's wall over the calls (the author's, flagged);
- the declared signature per backend: sparse_numba._symbols against
  the .pyx `cdef api` declarations, and the registered addresses;
- the oracle cells: x bit-equal to tests/data/oracle_0111.npz, the 0.1.11
  ctypes path's x (tests/make_oracle_0111.py).

No cell carries an xfail marker.
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

import _sn1_common as common

CHILD2_FRACTION = 0.5
CHILD_SCRIPT = os.path.join(common.TESTS_DIR, "sn1_cache_child.py")


PYX = {
    "superlu": os.path.join(common.REPO_DIR, "sparse_numba", "sparse_superlu",
                            "cy_superlu_wrapper.pyx"),
    "umfpack": os.path.join(common.REPO_DIR, "sparse_numba", "sparse_umfpack",
                            "cy_umfpack_wrapper.pyx"),
}
CY_MODULE = {
    "superlu": "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "umfpack": "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
}


def _require(backend):
    try:
        __import__(CY_MODULE[backend])
    except ImportError as e:
        pytest.skip(f"{backend} extension absent: {e}")


def _oracle():
    return np.load(common.ORACLE_PATH)


# ---------------------------------------------------------------- the round trip

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


@pytest.mark.parametrize("backend", [
    "superlu",
    "umfpack",
])
def test_cache_round_trip(backend, tmp_path):
    _require(backend)
    cache_dir = tmp_path / "numba_cache"
    cache_dir.mkdir()
    proc1, proc_wall1 = _run_child(backend, cache_dir, tmp_path / "child1.npz")
    snap1 = _snapshot(cache_dir)
    proc2, proc_wall2 = _run_child(backend, cache_dir, tmp_path / "child2.npz")
    snap2 = _snapshot(cache_dir)

    record = {"backend": backend, "returncode": [proc1.returncode, proc2.returncode],
              "process_wall_s": [proc_wall1, proc_wall2], "cache_files": snap1,
              "cache_files_after_child2": snap2,
              "stderr": [proc1.stderr[-4000:], proc2.stderr[-4000:]]}
    if proc1.returncode == 0 and proc2.returncode == 0:
        c1 = np.load(tmp_path / "child1.npz")
        c2 = np.load(tmp_path / "child2.npz")
        record["calls_wall_s"] = [float(c1["wall_calls"]), float(c2["wall_calls"])]
        record["import_wall_s"] = [float(c1["wall_import"]), float(c2["wall_import"])]
    with open(tmp_path / "round_trip_record.json", "w", encoding="ascii") as f:
        json.dump(record, f, indent=1)

    assert proc1.returncode == 0, proc1.stderr[-4000:]
    assert proc2.returncode == 0, proc2.stderr[-4000:]
    # (4) no "Cannot cache" in either child
    assert "Cannot cache" not in proc1.stderr, proc1.stderr[-4000:]
    assert "Cannot cache" not in proc2.stderr, proc2.stderr[-4000:]
    # (1) the index files exist after child 1 (the library's and the caller's),
    #     unchanged after child 2
    stem = os.path.basename(common.BACKENDS[backend].replace(".", os.sep))
    assert any(k.endswith(".nbi") and stem in k for k in snap1), sorted(snap1)
    assert any(k.endswith(".nbi") and "sn1_cache_child" in k for k in snap1), sorted(snap1)
    assert snap2 == snap1
    # (2) child 2 under the stated fraction of child 1
    wall1, wall2 = record["calls_wall_s"]
    assert wall2 < CHILD2_FRACTION * wall1, (wall1, wall2)
    # (3) x bit-equal across the children and to the 0.1.11 oracle
    oracle = _oracle()
    keys = [k for k in c1.files if k.startswith("x__")]
    assert len(keys) == 7, keys
    for k in keys:
        assert common.bits_equal(c1[k], c2[k]), k
        name = k[3:]
        ref = "factored_csc" if name == "caller_factor_solve_free" else name
        assert common.bits_equal(c1[k], oracle[f"{backend}__{ref}"]), k


# ------------------------------------------------------- the declared signature

_PYX_API = re.compile(r"cdef\s+api\s+(\w+)\s+(\w+)\s*\(([^)]*)\)", re.S)


def _pyx_declarations(path):
    """{cdef api name: (return C type, [argument C types])} from the .pyx."""
    with open(path, "r", encoding="ascii") as f:
        text = f.read()
    decls = {}
    for ret, name, args in _PYX_API.findall(text):
        types_ = []
        for arg in args.split(","):
            arg = " ".join(arg.split())
            ptr = "*" in arg
            ctype = arg.replace("*", " ").split()[0]
            types_.append(ctype + "*" if ptr else ctype)
        decls[name] = (ret, types_)
    return decls


def _numba_signature(ret, args):
    from numba import types
    ctypes_map = {"int": types.intc, "int64_t": types.int64}
    conv = lambda t: types.voidptr if t.endswith("*") else ctypes_map[t]
    return conv(ret)(*[conv(a) for a in args])


@pytest.mark.parametrize("backend", [
    "superlu",
    "umfpack",
])
def test_declared_signature(backend):
    _require(backend)
    import importlib
    import llvmlite.binding as llb
    from numba.extending import get_cython_function_address
    from sparse_numba import _symbols

    importlib.import_module(common.BACKENDS[backend])
    decls = _pyx_declarations(PYX[backend])
    rows = [r for r in _symbols.TABLE if r[2] == CY_MODULE[backend]]
    assert sorted(r[1] for r in rows) == sorted(decls)
    for symbol, cdef_name, module, sig_text in rows:
        assert symbol.startswith(f"sparse_numba_{backend}_"), symbol
        declared = _symbols.declared(symbol)
        assert declared == _numba_signature(*decls[cdef_name]), (symbol, declared)
        assert str(declared) == sig_text, (symbol, sig_text)
        assert llb.address_of_symbol(symbol) == get_cython_function_address(module, cdef_name)


# ------------------------------------------------------------------ the oracle

@pytest.mark.parametrize("backend", ["superlu", "umfpack"])
def test_oracle_fixed_system(backend):
    _require(backend)
    a, b = common.fixed_system()
    xs = common.run_all(backend, a, b)
    oracle = _oracle()
    assert len(xs) == 6
    for key, x in xs.items():
        assert common.bits_equal(x, oracle[f"{backend}__{key}"]), key


def test_oracle_n69_superlu_natural():
    _require("superlu")
    xs = common.run_n69_superlu()
    oracle = _oracle()
    for key, x in xs.items():
        assert common.bits_equal(x, oracle[f"n69_superlu__{key}"]), key
