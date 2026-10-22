"""
Python Interface with SuperLU sparse linear solver

The 0.1.11 names, kept with their positional signatures and return shapes
as thin njit shims over sparse_numba.api at ColPerm NATURAL (the 0.1.11
factorization, bit for bit). They return the status codes of
sparse_numba.errors and never print.

THE CODES CONTRACT (the info of every call): 0 success; a negative code a
refusal or a backend error, by name in sparse_numba.errors (explain(info)
says it in words); a positive k SINGULAR, the zero pivot's 1-based index.
`info != 0` is failure, as at 0.1.11.
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: superlu_numba_interface.py

import numpy as np
from numba import njit, types
from numba.extending import get_cython_function_address
import ctypes

from sparse_numba import _symbols
from sparse_numba.api import (
    SUPERLU, NATURAL, BAD_INDEX, solve as _solve, _factorize_on, _solve_into_on, _free_on,
)
from sparse_numba.conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc

_MODULE = "sparse_numba.sparse_superlu.cy_superlu_wrapper"

# Every row of this backend registered at import under its stable name,
# each address resolved by module name; an absent extension's rows at
# stubs returning BACKEND_ABSENT. The c_* names are the ExternalFunction
# objects; the functype_* ctypes function types are kept as the declared
# signatures (data for callers' tests, never called)
_PRESENT = _symbols.register_backend(_MODULE)


def _address(cdef_name):
    return get_cython_function_address(_MODULE, cdef_name) if _PRESENT else None


def _function(name):
    return types.ExternalFunction(name, _symbols.declared(name))


addr = _address("cy_solve_sparse_system")
functype = ctypes.CFUNCTYPE(
    ctypes.c_int,  # Return type: status code
    ctypes.c_void_p,  # values array
    ctypes.c_void_p,  # rowind array
    ctypes.c_void_p,  # colptr array
    ctypes.c_int,  # Number of rows
    ctypes.c_int,  # Number of columns
    ctypes.c_int,  # Number of non-zeros
    ctypes.c_void_p,  # RHS array
    ctypes.c_void_p  # Solution array (output)
)
c_solve_sparse_system = _function("sparse_numba_superlu_solve_sparse_system")

__all__ = [
    'superlu_solve_csc', 'superlu_solve_coo', 'superlu_solve_csr',
    'superlu_factorize_csc', 'superlu_factorize_coo', 'superlu_factorize_csr',
    'superlu_solve_factored', 'superlu_free_factors',
]


@njit(nogil=True, cache=True)
def superlu_solve_csc(csc_data, csc_indices, csc_indptr, b):
    """
    Solve a sparse linear system Ax = b using SuperLU.
    Matrix A is in CSC format.

    Parameters:
    -----------
    csc_data : ndarray
        Nonzero values in CSC format
    csc_indices : ndarray
        Row indices in CSC format
    csc_indptr : ndarray
        Column pointers in CSC format
    b : ndarray
        Right-hand side vector

    Returns:
    --------
    x : ndarray
        Solution vector
    info : int
        Status code (0 for success; sparse_numba.explain(info) names it)
    """
    data = np.ascontiguousarray(csc_data)
    indices = np.ascontiguousarray(csc_indices)
    indptr = np.ascontiguousarray(csc_indptr)
    rhs = np.ascontiguousarray(b)
    return _solve(data, indices, indptr, len(indptr) - 1, rhs, SUPERLU, NATURAL)


@njit(nogil=True, cache=True)
def superlu_solve_coo(row_indices, col_indices, data, shape, b):
    """
    Solve a sparse linear system Ax = b using SuperLU.
    Matrix A is in COO format and will be converted to CSC.

    Parameters:
    -----------
    row_indices : ndarray
        Row indices for COO format
    col_indices : ndarray
        Column indices for COO format
    data : ndarray
        Nonzero values in COO format
    shape : tuple
        Shape of the matrix as (n_rows, n_cols)
    b : ndarray
        Right-hand side vector

    Returns:
    --------
    x : ndarray
        Solution vector
    info : int
        Status code (0 for success; BAD_INDEX for an index outside the shape)
    """
    n_rows, n_cols = shape

    # Check for invalid indices
    if np.any(row_indices < 0) or np.any(row_indices >= n_rows) \
            or np.any(col_indices < 0) or np.any(col_indices >= n_cols):
        return np.zeros(len(b)), np.int64(BAD_INDEX)

    # Convert COO to CSC
    csc_data, csc_indices, csc_indptr = convert_coo_to_csc(
        np.ascontiguousarray(row_indices), np.ascontiguousarray(col_indices),
        np.ascontiguousarray(data), n_rows, n_cols
    )

    # Solve using the CSC format
    return superlu_solve_csc(csc_data, csc_indices, csc_indptr, np.ascontiguousarray(b))


@njit(nogil=True, cache=True)
def superlu_solve_csr(csr_data, csr_indices, csr_indptr, b):
    """
    Numba-compatible version that first sorts the CSR indices,
    then passes to the existing superlu_solve_csr function.

    Parameters are the same as superlu_solve_csc.
    """

    # Ensure correct data types
    csr_data_f64 = csr_data.astype(np.float64)
    csr_indices_i32 = csr_indices.astype(np.int32)
    csr_indptr_i32 = csr_indptr.astype(np.int32)
    b_f64 = b.astype(np.float64)

    # Convert CSR to CSC directly
    csc_data, csc_indices, csc_indptr = convert_csr_to_csc(
        csr_data_f64, csr_indices_i32, csr_indptr_i32
    )

    return superlu_solve_csc(csc_data, csc_indices, csc_indptr, b_f64)


# ================================================================
# Pre-factorization API: factorize once, solve many times
# ================================================================

# Load the factorize function
addr_factorize = _address("cy_factorize_sparse_system")
functype_factorize = ctypes.CFUNCTYPE(
    ctypes.c_int,       # return: status
    ctypes.c_void_p,    # values
    ctypes.c_void_p,    # rowind
    ctypes.c_void_p,    # colptr
    ctypes.c_int,       # nrows
    ctypes.c_int,       # ncols
    ctypes.c_int,       # nnz
    ctypes.c_void_p,    # handle_out (pointer to int64)
)
c_factorize_sparse_system = _function("sparse_numba_superlu_factorize_csc")

# Load the solve-with-factors function
addr_solve_factored = _address("cy_solve_with_factors")
functype_solve_factored = ctypes.CFUNCTYPE(
    ctypes.c_int,       # return: status
    ctypes.c_int64,     # handle (int64)
    ctypes.c_void_p,    # rhs
    ctypes.c_void_p,    # solution
    ctypes.c_int,       # nrhs
)
c_solve_with_factors = _function("sparse_numba_superlu_solve_with_factors")

# Load the free-factors function
addr_free = _address("cy_free_sparse_factors")
functype_free = ctypes.CFUNCTYPE(
    ctypes.c_int,       # return: status
    ctypes.c_int64,     # handle
)
c_free_sparse_factors = _function("sparse_numba_superlu_free_factors")

# the entries of the backend-neutral API (sparse_numba.api)
c_factorize_ordered = _function("sparse_numba_superlu_factorize_ordered")
c_refactorize = _function("sparse_numba_superlu_refactorize")
c_solve_into = _function("sparse_numba_superlu_solve_into")
c_handle_info = _function("sparse_numba_superlu_handle_info")


@njit(nogil=True, cache=True)
def superlu_factorize_csc(csc_data, csc_indices, csc_indptr):
    """
    Pre-factorize a sparse matrix in CSC format using SuperLU, at ColPerm
    NATURAL.

    Deprecated through 0.2.x in favour of sparse_numba.factorize(data,
    indices, indptr, n, SUPERLU, ordering); removed at 0.3.

    Parameters:
    -----------
    csc_data : ndarray (float64)
        Nonzero values in CSC format
    csc_indices : ndarray (int32)
        Row indices in CSC format
    csc_indptr : ndarray (int32)
        Column pointers in CSC format

    Returns:
    --------
    handle : int64
        Opaque handle to the stored LU factors (process-local).
        Must be freed with superlu_free_factors(handle).
    info : int
        Status code (0 for success)
    """
    data = np.ascontiguousarray(csc_data)
    indices = np.ascontiguousarray(csc_indices)
    indptr = np.ascontiguousarray(csc_indptr)
    return _factorize_on(SUPERLU, data, indices, indptr, len(indptr) - 1, NATURAL)


@njit(nogil=True, cache=True)
def superlu_solve_factored(handle, b):
    """
    Solve A*x = b using pre-computed LU factors from superlu_factorize_*().

    Deprecated through 0.2.x (it allocates x) in favour of
    sparse_numba.solve_into(handle, b, out); removed at 0.3.

    Parameters:
    -----------
    handle : int64
        LU factors handle from superlu_factorize_*()
    b : ndarray (float64)
        Right-hand side vector

    Returns:
    --------
    x : ndarray (float64)
        Solution vector
    info : int
        Status code (0 for success; HANDLE_FREED after a free,
        HANDLE_WRONG_BACKEND for another backend's handle)
    """
    rhs = np.ascontiguousarray(b)
    result = np.zeros(len(rhs), dtype=np.float64)
    info = _solve_into_on(SUPERLU, handle, rhs, result)
    return result, info


@njit(nogil=True, cache=True)
def superlu_free_factors(handle):
    """
    Free memory associated with LU factors.

    Must be called when the handle is no longer needed to avoid memory leaks.
    A second free of the same handle returns HANDLE_FREED.

    Deprecated through 0.2.x in favour of sparse_numba.free(handle); removed
    at 0.3.

    Parameters:
    -----------
    handle : int64
        LU factors handle from superlu_factorize_*()

    Returns:
    --------
    info : int
        Status code (0 for success)
    """
    return _free_on(SUPERLU, handle)


@njit(nogil=True, cache=True)
def superlu_factorize_coo(row_indices, col_indices, data, shape):
    """
    Pre-factorize a sparse matrix in COO format using SuperLU.
    Converts to CSC internally, then factorizes.

    Deprecated through 0.2.x in favour of sparse_numba.factorize(...,
    ordering); removed at 0.3.

    Parameters:
    -----------
    row_indices : ndarray (int32)
        Row indices for COO format
    col_indices : ndarray (int32)
        Column indices for COO format
    data : ndarray (float64)
        Nonzero values in COO format
    shape : tuple
        Shape of the matrix as (n_rows, n_cols)

    Returns:
    --------
    handle : int64
        Opaque handle to the stored LU factors
    info : int
        Status code (0 for success)
    """
    n_rows, n_cols = shape

    data_f64 = np.ascontiguousarray(data)
    row_indices_i32 = np.ascontiguousarray(row_indices)
    col_indices_i32 = np.ascontiguousarray(col_indices)

    # Convert COO to CSC
    csc_data, csc_indices, csc_indptr = convert_coo_to_csc(
        row_indices_i32, col_indices_i32, data_f64, n_rows, n_cols
    )

    return superlu_factorize_csc(csc_data, csc_indices, csc_indptr)


@njit(nogil=True, cache=True)
def superlu_factorize_csr(csr_data, csr_indices, csr_indptr):
    """
    Pre-factorize a sparse matrix in CSR format using SuperLU.
    Converts to CSC internally, then factorizes.

    Deprecated through 0.2.x in favour of sparse_numba.factorize(...,
    ordering); removed at 0.3.

    Parameters:
    -----------
    csr_data : ndarray (float64)
        Nonzero values in CSR format
    csr_indices : ndarray (int32)
        Column indices in CSR format
    csr_indptr : ndarray (int32)
        Row pointers in CSR format

    Returns:
    --------
    handle : int64
        Opaque handle to the stored LU factors
    info : int
        Status code (0 for success)
    """
    csr_data_f64 = csr_data.astype(np.float64)
    csr_indices_i32 = csr_indices.astype(np.int32)
    csr_indptr_i32 = csr_indptr.astype(np.int32)

    # Convert CSR to CSC
    csc_data, csc_indices, csc_indptr = convert_csr_to_csc(
        csr_data_f64, csr_indices_i32, csr_indptr_i32
    )

    return superlu_factorize_csc(csc_data, csc_indices, csc_indptr)
