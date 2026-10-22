"""
The child of the silent-import cell (run by
test_sn2_contract.py; never collected by pytest).

    python sn2_import_child.py <out.json>

Hides the UMFPACK DLLs, imports sparse_numba, and writes what it found to
<out.json>; it writes nothing to stdout or stderr itself. The DLLs are
hidden by PATH and by refusing the package's own add_dll_directory of a
SuiteSparse directory: since Python 3.8 Windows resolves an extension's
DLL dependencies through add_dll_directory, not through PATH.
"""

import json
import os
import sys

OUT = sys.argv[1]

# the UMFPACK library is also named by these (the user's); unset them
for _name in ("SPARSE_NUMBA_UMFPACK_LIBRARY", "SUITESPARSE_BIN"):
    os.environ.pop(_name, None)

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if "suitesparse" not in p.lower())

if hasattr(os, "add_dll_directory"):
    _add_dll_directory = os.add_dll_directory

    def _hiding_add_dll_directory(path):
        if "suitesparse" in str(path).lower():
            return None
        return _add_dll_directory(path)

    os.add_dll_directory = _hiding_add_dll_directory

import numpy as np

import sparse_numba

facts = {}
try:
    import sparse_numba.sparse_umfpack.cy_umfpack_wrapper  # noqa: F401
    facts["umfpack_module_failed"] = False
except ImportError:
    facts["umfpack_module_failed"] = True
facts["is_umf_available"] = sparse_numba.is_umf_available()
facts["is_slu_available"] = sparse_numba.is_slu_available()
facts["backends"] = [int(b) for b in sparse_numba.backends()]

data = np.array([4.0, 1.0, 1.0, 4.0])
indices = np.array([0, 1, 0, 1], dtype=np.int32)
indptr = np.array([0, 2, 4], dtype=np.int32)
handle, info = sparse_numba.factorize(data, indices, indptr, 2,
                                      sparse_numba.UMFPACK, sparse_numba.AMD)
facts["umfpack_factorize_info"] = int(info)
handle, info = sparse_numba.umfpack_factorize_csc(data, indices, indptr)
facts["kept_umfpack_factorize_info"] = int(info)
handle, info = sparse_numba.factorize(data, indices, indptr, 2,
                                      sparse_numba.SUPERLU, sparse_numba.NATURAL)
facts["superlu_factorize_info"] = int(info)
if info == 0:
    sparse_numba.free(handle)

with open(OUT, "w", encoding="ascii") as f:
    json.dump(facts, f)
