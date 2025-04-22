"""
Python Interface with UMFPACK sparse linear solver
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: umfpack_numba_interface.py

import numpy as np
from numba import njit, types
from numba.extending import get_cython_function_address
import ctypes

# Load the UMFPACK wrapper function
addr = get_cython_function_address("sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
                                   "cy_solve_sparse_system")
# addr = get_cython_function_address("cy_umfpack_wrapper",
#                                    "cy_solve_sparse_system")
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

__all__ = ['umfpack_solve_csc', 'umfpack_solve_coo', 'umfpack_solve_csr']


@njit(nogil=True)
def umfpack_solve_csc(csc_data, csc_indices, csc_indptr, b):
    """
    Solve a sparse linear system Ax = b using UMFPACK.
    Matrix A is in CSC format.
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
from sparse_numba.conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc
# from . import convert_coo_to_csc, convert_csr_to_csc

@njit(nogil=True)
def umfpack_solve_coo(row_indices, col_indices, data, shape, b):
    """
    Solve a sparse linear system Ax = b using UMFPACK.
    Matrix A is in COO format and will be converted to CSC.
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
    return umfpack_solve_csc(csc_data, csc_indices, csc_indptr, b_f64)


@njit(nogil=True)
def umfpack_solve_csr(csr_data, csr_indices, csr_indptr, b):
    """
    Solve a sparse linear system Ax = b using UMFPACK.
    Matrix A is in CSR format and will be converted to CSC.
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

    return umfpack_solve_csc(csc_data, csc_indices, csc_indptr, b_f64)
