"""
The child of the UMFPACK runtime-load cells (the user's library; run by
test_sn3_backends.py; never collected by pytest).

    python sn3_umfpack_child.py <mode> <out.json>

mode:
  hidden   the UMFPACK library hidden: PATH without SuiteSparse, the
           package's add_dll_directory of a SuiteSparse directory refused,
           SPARSE_NUMBA_UMFPACK_LIBRARY and SUITESPARSE_BIN unset;
  envvar   hidden as above, with SPARSE_NUMBA_UMFPACK_LIBRARY naming the
           private tree's test library (vendor/suitesparse/bin);
  badpath  nothing hidden, SPARSE_NUMBA_UMFPACK_LIBRARY naming a file that
           does not exist.
Writes what it found to <out.json>; nothing to stdout or stderr itself.
"""

import json
import os
import sys

MODE, OUT = sys.argv[1], sys.argv[2]
# the source tree (its vendor/suitesparse holds the test UMFPACK library):
# beside tests/, or SPARSE_NUMBA_SOURCE_DIR (tests/_sn1_common.py)
REPO_DIR = os.path.abspath(os.environ.get("SPARSE_NUMBA_SOURCE_DIR")
                           or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for name in ("SPARSE_NUMBA_UMFPACK_LIBRARY", "SUITESPARSE_BIN"):
    os.environ.pop(name, None)

if MODE in ("hidden", "envvar"):
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if "suitesparse" not in p.lower())
    if hasattr(os, "add_dll_directory"):
        _add_dll_directory = os.add_dll_directory

        def _hiding_add_dll_directory(path):
            if "suitesparse" in str(path).lower():
                return None
            return _add_dll_directory(path)

        os.add_dll_directory = _hiding_add_dll_directory

if MODE == "envvar":
    os.environ["SPARSE_NUMBA_UMFPACK_LIBRARY"] = os.path.join(
        REPO_DIR, "vendor", "suitesparse", "bin", "libumfpack.dll")
elif MODE == "badpath":
    os.environ["SPARSE_NUMBA_UMFPACK_LIBRARY"] = os.path.join(
        REPO_DIR, "vendor", "no_such_dir", "no_such_umfpack_library")

import numpy as np

import sparse_numba

if MODE in ("hidden", "envvar"):
    # the package prepends its library directories to PATH at import; strip
    # them again so only SPARSE_NUMBA_UMFPACK_LIBRARY can name the library
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if "suitesparse" not in p.lower())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sn1_common as common

facts = {"mode": MODE}
try:
    import sparse_numba.sparse_umfpack.cy_umfpack_wrapper  # noqa: F401
    facts["umfpack_module_imported"] = True
except ImportError:
    facts["umfpack_module_imported"] = False
facts["is_umf_available"] = sparse_numba.is_umf_available()
facts["is_slu_available"] = sparse_numba.is_slu_available()
facts["backends"] = [int(b) for b in sparse_numba.backends()]

a, b = common.fixed_system()
data, indices, indptr = common.dense_to_csc(a)
n = a.shape[0]
handle, info = sparse_numba.factorize(data, indices, indptr, n,
                                      sparse_numba.UMFPACK, sparse_numba.AMD)
facts["umfpack_factorize_info"] = int(info)
if info == 0:
    x = np.empty(n)
    facts["umfpack_solve_info"] = int(sparse_numba.solve_into(handle, b, x))
    facts["x_fixed"] = [float(v) for v in x]
    sparse_numba.free(handle)
handle, info = sparse_numba.umfpack_factorize_csc(data, indices, indptr)
facts["kept_umfpack_factorize_info"] = int(info)
if info == 0:
    sparse_numba.umfpack_free_factors(handle)
handle, info = sparse_numba.factorize(data, indices, indptr, n,
                                      sparse_numba.SUPERLU, sparse_numba.NATURAL)
facts["superlu_factorize_info"] = int(info)
if info == 0:
    sparse_numba.free(handle)
facts["cache_key"] = sparse_numba.cache_key()

with open(OUT, "w", encoding="ascii") as f:
    json.dump(facts, f)
