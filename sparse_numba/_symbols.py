"""
The external symbols of sparse_numba: the name table, the declared
signatures, and the read-check-register helper the interface modules use
at import
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: _symbols.py

import ctypes
import hashlib
import importlib.util

import llvmlite.binding as llb
from numba import types
from numba.extending import get_cython_function_address

from sparse_numba.errors import BACKEND_ABSENT, SymbolConflictError

_SUPERLU = "sparse_numba.sparse_superlu.cy_superlu_wrapper"
_UMFPACK = "sparse_numba.sparse_umfpack.cy_umfpack_wrapper"
_EIGEN = "sparse_numba.sparse_eigen.cy_eigen_wrapper"

# The extension modules whose bytes enter cache_key()
_EXTENSIONS = (
    "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
    "sparse_numba.sparse_eigen.cy_eigen_wrapper",
)

_SIG_SOLVE = types.intc(types.voidptr, types.voidptr, types.voidptr,
                        types.intc, types.intc, types.intc,
                        types.voidptr, types.voidptr)
_SIG_FACTORIZE = types.intc(types.voidptr, types.voidptr, types.voidptr,
                            types.intc, types.intc, types.intc,
                            types.voidptr)
_SIG_SOLVE_FACTORED = types.intc(types.int64, types.voidptr, types.voidptr, types.intc)
_SIG_FREE = types.intc(types.int64)
_SIG_FACTORIZE_ORDERED = types.intc(types.voidptr, types.voidptr, types.voidptr,
                                    types.intc, types.intc, types.intc,
                                    types.voidptr)
_SIG_REFACTORIZE = types.intc(types.int64, types.voidptr, types.intc)
_SIG_SOLVE_INTO = types.intc(types.int64, types.voidptr, types.voidptr, types.intc, types.intc)
_SIG_HANDLE_INFO = types.intc(types.int64, types.voidptr, types.voidptr)

# (symbol name, cdef api name, module, numba signature): the names are stable
# and package-namespaced, never qualified by a version, a digest or a build
_ROWS = (
    ("sparse_numba_superlu_solve_sparse_system", "cy_solve_sparse_system", _SUPERLU, _SIG_SOLVE),
    ("sparse_numba_superlu_factorize_csc", "cy_factorize_sparse_system", _SUPERLU, _SIG_FACTORIZE),
    ("sparse_numba_superlu_solve_with_factors", "cy_solve_with_factors", _SUPERLU, _SIG_SOLVE_FACTORED),
    ("sparse_numba_superlu_free_factors", "cy_free_sparse_factors", _SUPERLU, _SIG_FREE),
    ("sparse_numba_umfpack_solve_sparse_system", "cy_solve_sparse_system", _UMFPACK, _SIG_SOLVE),
    ("sparse_numba_umfpack_factorize_csc", "cy_factorize_sparse_system", _UMFPACK, _SIG_FACTORIZE),
    ("sparse_numba_umfpack_solve_with_factors", "cy_solve_with_factors", _UMFPACK, _SIG_SOLVE_FACTORED),
    ("sparse_numba_umfpack_free_factors", "cy_free_sparse_factors", _UMFPACK, _SIG_FREE),
    # the backend-neutral API's entries (sparse_numba.api)
    ("sparse_numba_superlu_factorize_ordered", "cy_factorize_ordered", _SUPERLU, _SIG_FACTORIZE_ORDERED),
    ("sparse_numba_superlu_refactorize", "cy_refactorize", _SUPERLU, _SIG_REFACTORIZE),
    ("sparse_numba_superlu_solve_into", "cy_solve_into", _SUPERLU, _SIG_SOLVE_INTO),
    ("sparse_numba_superlu_handle_info", "cy_handle_info", _SUPERLU, _SIG_HANDLE_INFO),
    ("sparse_numba_umfpack_factorize_ordered", "cy_factorize_ordered", _UMFPACK, _SIG_FACTORIZE_ORDERED),
    ("sparse_numba_umfpack_refactorize", "cy_refactorize", _UMFPACK, _SIG_REFACTORIZE),
    ("sparse_numba_umfpack_solve_into", "cy_solve_into", _UMFPACK, _SIG_SOLVE_INTO),
    ("sparse_numba_umfpack_handle_info", "cy_handle_info", _UMFPACK, _SIG_HANDLE_INFO),
    # the Eigen backends, one extension for both
    # kinds; the handle names its kind, so only factorize is per kind
    ("sparse_numba_eigen_lu_factorize_ordered", "cy_lu_factorize_ordered", _EIGEN, _SIG_FACTORIZE_ORDERED),
    ("sparse_numba_eigen_ldlt_factorize_ordered", "cy_ldlt_factorize_ordered", _EIGEN, _SIG_FACTORIZE_ORDERED),
    ("sparse_numba_eigen_refactorize", "cy_refactorize", _EIGEN, _SIG_REFACTORIZE),
    ("sparse_numba_eigen_solve_into", "cy_solve_into", _EIGEN, _SIG_SOLVE_INTO),
    ("sparse_numba_eigen_handle_info", "cy_handle_info", _EIGEN, _SIG_HANDLE_INFO),
    ("sparse_numba_eigen_free", "cy_free_factors", _EIGEN, _SIG_FREE),
)

TABLE = tuple((name, cdef_name, module, str(sig)) for name, cdef_name, module, sig in _ROWS)

_SIGNATURES = {name: sig for name, _, _, sig in _ROWS}


def declared(name):
    """The declared ExternalFunction signature of a symbol of TABLE."""
    return _SIGNATURES[name]


def register(name, address):
    """Register address under name: read before write, never a silent overwrite."""
    existing = llb.address_of_symbol(name)
    if existing is not None and existing != address:
        raise SymbolConflictError(
            f"the LLVM symbol {name!r} is already registered at {existing:#x}, "
            f"not at this process's address {address:#x}")
    if existing is None:
        llb.add_symbol(name, address)


def external(name, address):
    """Register a symbol of TABLE and return its ExternalFunction."""
    register(name, address)
    return types.ExternalFunction(name, declared(name))


_CTYPES = {types.intc: ctypes.c_int, types.int64: ctypes.c_int64,
           types.voidptr: ctypes.c_void_p}

# The C-callable stubs registered for an absent backend's rows, kept alive here
_STUBS = {}


def _absent_stub(name):
    """The address of a C-callable stub with name's declared signature that
    returns BACKEND_ABSENT: an absent backend's first call is refused by name
    (the import stays silent), and a cached caller never meets an unregistered
    symbol."""
    if name not in _STUBS:
        sig = declared(name)
        functype = ctypes.CFUNCTYPE(_CTYPES[sig.return_type], *[_CTYPES[a] for a in sig.args])
        _STUBS[name] = functype(lambda *args: BACKEND_ABSENT)
    return ctypes.cast(_STUBS[name], ctypes.c_void_p).value


def register_backend(module):
    """Register every TABLE row of a backend's extension module, each address
    resolved by module name; when the extension does not import, its
    rows are registered at stubs returning BACKEND_ABSENT. True if present.

    UMFPACK's extension is interface-only and always imports (no link-time
    SuiteSparse): its rows are the shim's own functions, which return
    BACKEND_ABSENT while the user's library is not found, so UMFPACK meets
    the stubs only in a SPARSE_NUMBA_SKIP_UMFPACK build (no extension)."""
    rows = [(name, cdef_name) for name, cdef_name, mod, _ in _ROWS if mod == module]
    try:
        addresses = [(name, get_cython_function_address(module, cdef_name))
                     for name, cdef_name in rows]
    except ImportError:
        for name, _ in rows:
            register(name, _absent_stub(name))
        return False
    for name, address in addresses:
        register(name, address)
    return True


def _sha256_of_module_file(module):
    try:
        spec = importlib.util.find_spec(module)
    except ImportError:
        spec = None
    if spec is None or spec.origin is None:
        return "absent"
    with open(spec.origin, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _superlu_build():
    """The SuperLU version and the shim's compiler flags, compiled into the
    extension; "absent" without the extension."""
    try:
        module = importlib.import_module(_SUPERLU)
    except ImportError:
        return "absent"
    return module.build_info()


def _eigen_build():
    """The Eigen headers' version and the shim's build facts, compiled into
    the extension; "absent" without the extension."""
    try:
        module = importlib.import_module(_EIGEN)
    except ImportError:
        return "absent"
    return module.build_info()


def umfpack_library():
    """(code, version, source) of the user's UMFPACK, loaded by the shim at
    this first need (interface-only): code OK or BACKEND_ABSENT;
    version "x.y.z" (umfpack_version), "unknown" or "absent"; source the
    path or the name it loaded from."""
    try:
        module = importlib.import_module(_UMFPACK)
    except ImportError:
        return BACKEND_ABSENT, "absent", ""
    return module.library_info()


def cache_key():
    """A string a caller may fold into its own cache key: the version, the
    symbol table, the sha256 of each backend's extension module, the
    SuperLU version with the shim's compiler flags, the version of the
    user's UMFPACK library ("absent" when it is not found), and the Eigen
    version with its shim's compiler flags."""
    from sparse_numba import __version__  # the package's one version string
    table = hashlib.sha256(repr(TABLE).encode("ascii")).hexdigest()[:16]
    parts = [f"sparse_numba={__version__}", f"symbols={table}"]
    for module in _EXTENSIONS:
        parts.append(f"{module.rsplit('.', 1)[1]}={_sha256_of_module_file(module)}")
    parts.append(f"superlu={_superlu_build()}")
    parts.append(f"umfpack={umfpack_library()[1]}")
    parts.append(f"eigen={_eigen_build()}")
    return ";".join(parts)
