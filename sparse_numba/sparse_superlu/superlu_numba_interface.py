"""
Python Interface with SuperLU sparse linear solver
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

# Load the SuperLU wrapper function
addr = get_cython_function_address("sparse_numba.sparse_superlu.cy_superlu_wrapper",
                                   "cy_solve_sparse_system")
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
c_solve_sparse_system = functype(addr)

__all__ = [
    'superlu_solve_csc', 'superlu_solve_coo', 'superlu_solve_csr',
    'superlu_factorize_csc', 'superlu_factorize_coo', 'superlu_factorize_csr',
    'superlu_solve_factored', 'superlu_free_factors',
]

@njit(nogil=True)
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
        Status code (0 for success)
    """
    # Create contiguous copies with correct data types
    data = np.ascontiguousarray(csc_data)
    indices = np.ascontiguousarray(csc_indices)
    indptr = np.ascontiguousarray(csc_indptr)
    rhs = np.ascontiguousarray(b)

    n_rows = len(rhs)
    n_cols = len(indptr) - 1
    nnz = len(data)

    # print(f"Debug: Matrix size: {n_rows}x{n_cols}, NNZ: {nnz}")
    # print(f"Debug: First few values: {data[:min(5, len(data))]}")
    # print(f"Debug: First few indices: {indices[:min(5, len(indices))]}")
    # print(f"Debug: First few indptr: {indptr[:min(5, len(indptr))]}")

    # Validate CSC format
    if indptr[0] != 0:
        print(f"Error: First element of indptr must be 0, got {indptr[0]}")
        return np.zeros_like(rhs), -1

    if indptr[n_cols] != nnz:
        print(f"Error: Last element of indptr must be {nnz}, got {indptr[n_cols]}")
        return np.zeros_like(rhs), -2

    # CRITICAL: Ensure solution array is properly allocated
    # - Must be contiguous
    # - Must be initialized to zeros
    # - Must be the right size and type
    solution = np.zeros(n_rows, dtype=np.float64)

    # # Double-check solution array
    # print(f"Debug: Solution array shape: {solution.shape}, dtype: {solution.dtype}")
    # print(f"Debug: Solution array is contiguous: {solution.flags.c_contiguous}")

    # Explicitly create a new array for the result
    # This can help with some memory alignment issues
    result = np.zeros(n_rows, dtype=np.float64)

    # Call the C function
    info = c_solve_sparse_system(
        data.ctypes.data,
        indices.ctypes.data,
        indptr.ctypes.data,
        n_rows,
        n_cols,
        nnz,
        rhs.ctypes.data,
        result.ctypes.data  # Use our new array here
    )

    return result, info


# Import conversion functions
# from . import convert_coo_to_csc, convert_csr_to_csc
from sparse_numba.conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc

@njit(nogil=True)
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
        Status code (0 for success)
    """
    n_rows, n_cols = shape

    # Check for invalid indices
    if np.any(row_indices < 0) or np.any(row_indices >= n_rows):
        print(f"Error: Invalid row indices. Min: {row_indices.min()}, Max: {row_indices.max()}, Rows: {n_rows}")
        return np.zeros_like(b), -1

    if np.any(col_indices < 0) or np.any(col_indices >= n_cols):
        print(f"Error: Invalid column indices. Min: {col_indices.min()}, Max: {col_indices.max()}, Cols: {n_cols}")
        return np.zeros_like(b), -2

    # Ensure correct data types
    data_f64 = np.ascontiguousarray(data)
    row_indices_i32 = np.ascontiguousarray(row_indices)
    col_indices_i32 = np.ascontiguousarray(col_indices)
    b_f64 = np.ascontiguousarray(b)

    # # Debug messages
    # print(f"COO Debug: Matrix size: {n_rows}x{n_cols}, NNZ: {len(data_f64)}")
    # print(f"COO Debug: Row indices range: [{row_indices_i32.min()}, {row_indices_i32.max()}]")
    # print(f"COO Debug: Col indices range: [{col_indices_i32.min()}, {col_indices_i32.max()}]")

    # Convert COO to CSC
    csc_data, csc_indices, csc_indptr = convert_coo_to_csc(
        row_indices_i32, col_indices_i32, data_f64, n_rows, n_cols
    )

    # Verify CSC format
    if csc_indptr[0] != 0:
        print(f"Error: First element of CSC indptr must be 0, got {csc_indptr[0]}")
        return np.zeros_like(b_f64), -3

    if csc_indptr[-1] != len(csc_data):
        print(f"Error: Last element of CSC indptr must equal NNZ ({len(csc_data)}), got {csc_indptr[-1]}")
        return np.zeros_like(b_f64), -4

    # # Additional debugging for the CSC format
    # print(f"CSC Debug: Converted to CSC format, NNZ: {len(csc_data)}")
    # print(f"CSC Debug: First few indptr values: {csc_indptr[:min(5, len(csc_indptr))]}")
    # print(f"CSC Debug: Last indptr value: {csc_indptr[-1]}")

    # Solve using the CSC format
    return superlu_solve_csc(csc_data, csc_indices, csc_indptr, b_f64)


@njit
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
addr_factorize = get_cython_function_address(
    "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "cy_factorize_sparse_system")
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
c_factorize_sparse_system = functype_factorize(addr_factorize)

# Load the solve-with-factors function
addr_solve_factored = get_cython_function_address(
    "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "cy_solve_with_factors")
functype_solve_factored = ctypes.CFUNCTYPE(
    ctypes.c_int,       # return: status
    ctypes.c_int64,     # handle (int64)
    ctypes.c_void_p,    # rhs
    ctypes.c_void_p,    # solution
    ctypes.c_int,       # nrhs
)
c_solve_with_factors = functype_solve_factored(addr_solve_factored)

# Load the free-factors function
addr_free = get_cython_function_address(
    "sparse_numba.sparse_superlu.cy_superlu_wrapper",
    "cy_free_sparse_factors")
functype_free = ctypes.CFUNCTYPE(
    ctypes.c_int,       # return: status
    ctypes.c_int64,     # handle
)
c_free_sparse_factors = functype_free(addr_free)


@njit(nogil=True)
def superlu_factorize_csc(csc_data, csc_indices, csc_indptr):
    """
    Pre-factorize a sparse matrix in CSC format using SuperLU.

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
        Opaque handle to the stored LU factors.
        Must be freed with superlu_free_factors(handle).
    info : int
        Status code (0 for success)
    """
    data = np.ascontiguousarray(csc_data)
    indices = np.ascontiguousarray(csc_indices)
    indptr = np.ascontiguousarray(csc_indptr)

    n_cols = len(indptr) - 1
    n_rows = n_cols  # Square matrix assumption for linear solvers
    nnz = len(data)

    # Validate CSC format
    if indptr[0] != 0:
        print("Error: First element of indptr must be 0")
        return np.int64(0), -1
    if indptr[n_cols] != nnz:
        print("Error: Last element of indptr must equal nnz")
        return np.int64(0), -2

    # Allocate output handle as a 1-element int64 array (for passing pointer to C)
    handle_arr = np.zeros(1, dtype=np.int64)

    info = c_factorize_sparse_system(
        data.ctypes.data,
        indices.ctypes.data,
        indptr.ctypes.data,
        n_rows,
        n_cols,
        nnz,
        handle_arr.ctypes.data,
    )

    return handle_arr[0], info


@njit(nogil=True)
def superlu_solve_factored(handle, b):
    """
    Solve A*x = b using pre-computed LU factors from superlu_factorize_*().

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
        Status code (0 for success)
    """
    rhs = np.ascontiguousarray(b)
    n = len(rhs)
    result = np.zeros(n, dtype=np.float64)

    info = c_solve_with_factors(
        handle,
        rhs.ctypes.data,
        result.ctypes.data,
        1,  # nrhs = 1
    )

    return result, info


@njit(nogil=True)
def superlu_free_factors(handle):
    """
    Free memory associated with LU factors.

    Must be called when the handle is no longer needed to avoid memory leaks.

    Parameters:
    -----------
    handle : int64
        LU factors handle from superlu_factorize_*()

    Returns:
    --------
    info : int
        Status code (0 for success)
    """
    info = c_free_sparse_factors(handle)
    return info


@njit(nogil=True)
def superlu_factorize_coo(row_indices, col_indices, data, shape):
    """
    Pre-factorize a sparse matrix in COO format using SuperLU.
    Converts to CSC internally, then factorizes.

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


@njit(nogil=True)
def superlu_factorize_csr(csr_data, csr_indices, csr_indptr):
    """
    Pre-factorize a sparse matrix in CSR format using SuperLU.
    Converts to CSC internally, then factorizes.

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
