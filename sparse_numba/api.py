"""
The backend-neutral interface of sparse_numba

    factorize(data, indices, indptr, n, backend, ordering) -> (handle, info)
    solve_into(handle, b, out) -> info
    solve_many_into(handle, B, OUT) -> info
    refactorize(handle, data) -> info
    free(handle) -> info
    solve(data, indices, indptr, n, b, backend, ordering) -> (x, info)
    handle_info(handle) -> (n, nnz_L, nnz_U, backend_id, ordering_id,
                            last_status, pivot_tolerance, info)

all njit(nogil=True, cache=True); the python-level Factors, explain,
SparseNumbaError, require, backends and is_eigen_available. The input is
CSC (float64 data, int32 indices and indptr) of an n x n matrix; b and out
float64, 1-D, len n, C-contiguous; B and OUT (n, k) C-contiguous. A failed
check RETURNS a code of sparse_numba.errors (nothing is printed and nothing
is called).

THE CODES CONTRACT (the info of every call): 0 success; a negative code a
refusal or a backend error, by name in sparse_numba.errors (explain(info)
says it in words); a positive k SINGULAR, the zero pivot's 1-based index.
`info != 0` is failure, as at 0.1.11.

The handle is an int64 index into the backend's C-side registry: a freed
handle returns HANDLE_FREED, another backend's HANDLE_WRONG_BACKEND. The
caller who factorized owns the handle and frees it. A handle is
process-local and never crosses processes (a pickled handle is an error).

Every symbol of sparse_numba.symbols.TABLE is registered at this module's
import; an absent backend's rows at stubs, so its first call returns
BACKEND_ABSENT. The orderings: SuperLU NATURAL, COLAMD,
MMD_AT_PLUS_A (options.ColPerm; AMD ORDERING_UNSUPPORTED); UMFPACK
AMD (STRATEGY_SYMMETRIC), COLAMD (STRATEGY_UNSYMMETRIC), NATURAL
(UMFPACK_ORDERING_NONE; refused by a library without umfpack_version);
MMD_AT_PLUS_A ORDERING_UNSUPPORTED; EIGEN_LU NATURAL, COLAMD, AMD
(Eigen::SparseLU's NaturalOrdering / COLAMDOrdering / AMDOrdering;
MMD_AT_PLUS_A ORDERING_UNSUPPORTED); EIGEN_LDLT AMD only
(Eigen::SimplicialLDLT, a symmetric pattern; else BAD_SHAPE). No default:
the ordering is an argument of factorize. UMFPACK is interface-only: the
user's SuiteSparse is loaded at the first call, from
SPARSE_NUMBA_UMFPACK_LIBRARY (a path) when set, else by the platform's
library names; not found, BACKEND_ABSENT.
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: api.py

import numpy as np
from numba import njit, types
from numba.extending import overload

from sparse_numba import _symbols
from sparse_numba._symbols import cache_key
from sparse_numba.sparse_eigen import eigen_numba_interface as _eigen
from sparse_numba.errors import (
    OK, BAD_DTYPE, BAD_SHAPE, BAD_INDPTR, BAD_INDEX, NOT_SQUARE,
    HANDLE_INVALID, HANDLE_FREED, HANDLE_WRONG_BACKEND,
    BACKEND_ABSENT, ORDERING_UNSUPPORTED, ALLOC_FAILED, BACKEND_ERROR,
    SparseNumbaError, SymbolConflictError, explain,
)

# The backends (mirroring sparse_numba_codes.h)
SUPERLU = 1
UMFPACK = 2
EIGEN_LU = 3
EIGEN_LDLT = 4

# The orderings
NATURAL = 0
COLAMD = 1
AMD = 2
MMD_AT_PLUS_A = 3

_SLOT_BITS = 24     # the backend id is the slot word's high byte
_INFO_LEN = 6

_MODULES = {
    SUPERLU: "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    UMFPACK: "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
}

_HINTS = {
    SUPERLU: "the SuperLU extension did not import; reinstall sparse_numba",
    UMFPACK: "install SuiteSparse (UMFPACK) and set SPARSE_NUMBA_UMFPACK_LIBRARY to "
             "the path of its umfpack library, or put the library where the "
             "platform's loader finds it",
    EIGEN_LU: "the Eigen extension did not import; reinstall sparse_numba",
    EIGEN_LDLT: "the Eigen extension did not import; reinstall sparse_numba",
}

# Register every row; an absent backend's rows at BACKEND_ABSENT stubs.
# The Eigen rows are registered by eigen_numba_interface (one extension for
# both kinds)
_PRESENT = {backend: _symbols.register_backend(module) for backend, module in _MODULES.items()}
_PRESENT[EIGEN_LU] = _PRESENT[EIGEN_LDLT] = _eigen.is_present()


def _external(name):
    return types.ExternalFunction(name, _symbols.declared(name))


_slu_factorize = _external("sparse_numba_superlu_factorize_ordered")
_slu_refactorize = _external("sparse_numba_superlu_refactorize")
_slu_solve_into = _external("sparse_numba_superlu_solve_into")
_slu_handle_info = _external("sparse_numba_superlu_handle_info")
_slu_free = _external("sparse_numba_superlu_free_factors")
_umf_factorize = _external("sparse_numba_umfpack_factorize_ordered")
_umf_refactorize = _external("sparse_numba_umfpack_refactorize")
_umf_solve_into = _external("sparse_numba_umfpack_solve_into")
_umf_handle_info = _external("sparse_numba_umfpack_handle_info")
_umf_free = _external("sparse_numba_umfpack_free_factors")
_eig_lu_factorize = _eigen.c_lu_factorize
_eig_ldlt_factorize = _eigen.c_ldlt_factorize
_eig_refactorize = _eigen.c_refactorize
_eig_solve_into = _eigen.c_solve_into
_eig_handle_info = _eigen.c_handle_info
_eig_free = _eigen.c_free


# ================================================================
# The validated boundary: the dtype and rank checks resolve at
# compile time from the array types, the shape and content checks at run time
# ================================================================

def _vector_code(a):
    """OK, or the code refusing a as a float64 1-D C-contiguous array."""


def _matrix_shape(a):
    """(code, rows, columns) of a as a float64 2-D C-contiguous array."""


def _csc_code(data, indices, indptr, n):
    """OK, or the code refusing (data, indices, indptr) as an n x n CSC."""


def _is_array(a, dtype):
    return isinstance(a, types.Array) and a.dtype == dtype


@overload(_vector_code)
def _ov_vector_code(a):
    if not _is_array(a, types.float64):
        return lambda a: BAD_DTYPE
    if a.ndim != 1:
        return lambda a: BAD_SHAPE
    return lambda a: OK if a.flags.c_contiguous else BAD_SHAPE


@overload(_matrix_shape)
def _ov_matrix_shape(a):
    if not _is_array(a, types.float64):
        return lambda a: (BAD_DTYPE, 0, 0)
    if a.ndim != 2:
        return lambda a: (BAD_SHAPE, 0, 0)

    def impl(a):
        if not a.flags.c_contiguous:
            return (BAD_SHAPE, 0, 0)
        return (OK, a.shape[0], a.shape[1])
    return impl


@overload(_csc_code)
def _ov_csc_code(data, indices, indptr, n):
    if not (_is_array(data, types.float64) and _is_array(indices, types.int32)
            and _is_array(indptr, types.int32)):
        return lambda data, indices, indptr, n: BAD_DTYPE
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 \
            or not isinstance(n, types.Integer):
        return lambda data, indices, indptr, n: BAD_SHAPE

    def impl(data, indices, indptr, n):
        return _csc_content(data, indices, indptr, n)
    return impl


@njit(nogil=True, cache=True)
def _csc_content(data, indices, indptr, n):
    """The run-time checks: one O(nnz) pass, at factorize only."""
    if not (data.flags.c_contiguous and indices.flags.c_contiguous
            and indptr.flags.c_contiguous):
        return BAD_SHAPE
    if n <= 0:
        return BAD_SHAPE
    if len(indptr) != n + 1:
        return NOT_SQUARE
    nnz = len(data)
    if len(indices) != nnz:
        return BAD_SHAPE
    if indptr[0] != 0 or indptr[n] != nnz:
        return BAD_INDPTR
    for j in range(n):
        if indptr[j + 1] < indptr[j]:
            return BAD_INDPTR
    for p in range(nnz):
        i = indices[p]
        if i < 0 or i >= n:
            return BAD_INDEX
    return OK


@njit(nogil=True, cache=True)
def _backend_of(handle):
    """The backend id in a handle's slot word; 0 for none."""
    if handle <= 0:
        return 0
    return (handle >> _SLOT_BITS) & 0xFF


# ================================================================
# The per-backend calls: the 0.1.11 shims call these with their own backend,
# so a foreign handle reaches the C registry and returns HANDLE_WRONG_BACKEND
# ================================================================

@njit(nogil=True, cache=True)
def _factorize_on(backend, data, indices, indptr, n, ordering):
    code = _csc_code(data, indices, indptr, n)
    if code != OK:
        return np.int64(0), np.int64(code)
    handle = np.zeros(1, dtype=np.int64)
    if backend == SUPERLU:
        info = _slu_factorize(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                              n, len(data), ordering, handle.ctypes.data)
    elif backend == UMFPACK:
        info = _umf_factorize(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                              n, len(data), ordering, handle.ctypes.data)
    elif backend == EIGEN_LU:
        info = _eig_lu_factorize(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                                 n, len(data), ordering, handle.ctypes.data)
    elif backend == EIGEN_LDLT:
        info = _eig_ldlt_factorize(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                                   n, len(data), ordering, handle.ctypes.data)
    else:
        return np.int64(0), np.int64(BACKEND_ABSENT)
    return handle[0], np.int64(info)


@njit(nogil=True, cache=True)
def _solve_into_on(backend, handle, b, out):
    code = _vector_code(b)
    if code != OK:
        return np.int64(code)
    code = _vector_code(out)
    if code != OK:
        return np.int64(code)
    n = len(b)
    if len(out) != n:
        return np.int64(BAD_SHAPE)
    if backend == SUPERLU:
        return np.int64(_slu_solve_into(handle, b.ctypes.data, out.ctypes.data, n, 1))
    if backend == UMFPACK:
        return np.int64(_umf_solve_into(handle, b.ctypes.data, out.ctypes.data, n, 1))
    if backend == EIGEN_LU or backend == EIGEN_LDLT:
        return np.int64(_eig_solve_into(handle, b.ctypes.data, out.ctypes.data, n, 1))
    return np.int64(HANDLE_INVALID)


@njit(nogil=True, cache=True)
def _free_on(backend, handle):
    if backend == SUPERLU:
        return np.int64(_slu_free(handle))
    if backend == UMFPACK:
        return np.int64(_umf_free(handle))
    if backend == EIGEN_LU or backend == EIGEN_LDLT:
        return np.int64(_eig_free(handle))
    return np.int64(HANDLE_INVALID)


# ================================================================
# The interface
# ================================================================

@njit(nogil=True, cache=True)
def factorize(data, indices, indptr, n, backend, ordering):
    """Factorize the n x n CSC matrix (data float64, indices / indptr int32)
    with a backend at an ordering; returns (handle, info). The handle is
    nonzero when info is OK and is the caller's to free."""
    return _factorize_on(backend, data, indices, indptr, n, ordering)


@njit(nogil=True, cache=True)
def solve_into(handle, b, out):
    """Solve A x = b into the caller-held out (float64, len n); b is read
    only. Returns info."""
    return _solve_into_on(_backend_of(handle), handle, b, out)


@njit(nogil=True, cache=True)
def solve_many_into(handle, B, OUT):
    """Solve A X = B into OUT, B and OUT (n, k) C-contiguous, in one call.
    Returns info."""
    code, n, k = _matrix_shape(B)
    if code != OK:
        return np.int64(code)
    code, n_out, k_out = _matrix_shape(OUT)
    if code != OK:
        return np.int64(code)
    if n_out != n or k_out != k or k < 1:
        return np.int64(BAD_SHAPE)
    backend = _backend_of(handle)
    if backend == SUPERLU:
        return np.int64(_slu_solve_into(handle, B.ctypes.data, OUT.ctypes.data, n, k))
    if backend == UMFPACK:
        return np.int64(_umf_solve_into(handle, B.ctypes.data, OUT.ctypes.data, n, k))
    if backend == EIGEN_LU or backend == EIGEN_LDLT:
        return np.int64(_eig_solve_into(handle, B.ctypes.data, OUT.ctypes.data, n, k))
    return np.int64(HANDLE_INVALID)


@njit(nogil=True, cache=True)
def refactorize(handle, data):
    """Factorize the handle's pattern again on new values (float64, nnz of
    them, in the pattern's order). Returns info."""
    code = _vector_code(data)
    if code != OK:
        return np.int64(code)
    backend = _backend_of(handle)
    if backend == SUPERLU:
        return np.int64(_slu_refactorize(handle, data.ctypes.data, len(data)))
    if backend == UMFPACK:
        return np.int64(_umf_refactorize(handle, data.ctypes.data, len(data)))
    if backend == EIGEN_LU or backend == EIGEN_LDLT:
        return np.int64(_eig_refactorize(handle, data.ctypes.data, len(data)))
    return np.int64(HANDLE_INVALID)


@njit(nogil=True, cache=True)
def free(handle):
    """Free the handle's factors. A second free returns HANDLE_FREED."""
    return _free_on(_backend_of(handle), handle)


@njit(nogil=True, cache=True)
def solve(data, indices, indptr, n, b, backend, ordering):
    """The one-shot convenience: factorize + solve_into + free, allocating x.
    Returns (x, info)."""
    x = np.zeros(len(b))
    handle, info = _factorize_on(backend, data, indices, indptr, n, ordering)
    if info != OK:
        return x, info
    info = _solve_into_on(backend, handle, b, x)
    info_free = _free_on(backend, handle)
    if info != OK:
        return x, info
    return x, info_free


@njit(nogil=True, cache=True)
def handle_info(handle):
    """(n, nnz_L, nnz_U, backend_id, ordering_id, last_status,
    pivot_tolerance, info); the fields are zero when info is not OK.
    last_status is the backend's own status of the last factorization."""
    iout = np.zeros(_INFO_LEN, dtype=np.int64)
    dout = np.zeros(1, dtype=np.float64)
    backend = _backend_of(handle)
    if backend == SUPERLU:
        info = np.int64(_slu_handle_info(handle, iout.ctypes.data, dout.ctypes.data))
    elif backend == UMFPACK:
        info = np.int64(_umf_handle_info(handle, iout.ctypes.data, dout.ctypes.data))
    elif backend == EIGEN_LU or backend == EIGEN_LDLT:
        info = np.int64(_eig_handle_info(handle, iout.ctypes.data, dout.ctypes.data))
    else:
        info = np.int64(HANDLE_INVALID)
    return (iout[0], iout[1], iout[2], iout[3], iout[4], iout[5], dout[0], info)


# ================================================================
# python-level
# ================================================================

def _available(backend):
    if not _PRESENT.get(backend, False):
        return False
    if backend == UMFPACK:
        return _symbols.umfpack_library()[0] == OK
    return True


def backends():
    """The ids of the backends available in this process (UMFPACK's: its
    library found and loaded; the Eigen kinds': their extension imported)."""
    return tuple(backend for backend in (SUPERLU, UMFPACK, EIGEN_LU, EIGEN_LDLT)
                 if _available(backend))


def is_eigen_available():
    """True when the Eigen extension imported (EIGEN_LU and EIGEN_LDLT)."""
    return _available(EIGEN_LU)


def require(backend):
    """Raise SparseNumbaError(BACKEND_ABSENT) with the install hint when the
    backend is not available."""
    if backend not in backends():
        raise SparseNumbaError(BACKEND_ABSENT,
                               _HINTS.get(backend, f"no backend with id {backend}"))


class Factors:
    """factorize / free for Python callers, as a context manager:

        with Factors(data, indices, indptr, n, SUPERLU, NATURAL) as f:
            x = f.solve(b)

    A failed call raises SparseNumbaError. The handle is process-local and
    cannot be pickled."""

    def __init__(self, data, indices, indptr, n, backend, ordering):
        handle, info = factorize(data, indices, indptr, n, backend, ordering)
        if info != OK:
            raise SparseNumbaError(info)
        self.handle = handle
        self.n = n
        self.backend = backend
        self.ordering = ordering

    def solve(self, b):
        x = np.empty(self.n, dtype=np.float64)
        info = solve_into(self.handle, b, x)
        if info != OK:
            raise SparseNumbaError(info)
        return x

    def refactorize(self, data):
        info = refactorize(self.handle, data)
        if info != OK:
            raise SparseNumbaError(info)

    def free(self):
        if self.handle:
            handle, self.handle = self.handle, 0
            info = free(handle)
            if info != OK:
                raise SparseNumbaError(info)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()
        return False

    def __reduce__(self):
        raise TypeError("a sparse_numba handle is process-local and cannot be pickled")
