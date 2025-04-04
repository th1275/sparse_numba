import numpy as np
from numba import njit, types
from numba.extending import get_cython_function_address
import ctypes
from ctypes import c_int, c_double, POINTER, c_void_p

# Load the SuperLU wrapper function
addr = get_cython_function_address("sparse_numba.sparse_superlu.cy_superlu_wrapper",
                                   "cy_solve_sparse_system")
functype = ctypes.CFUNCTYPE(
    c_int,  # Return type: status code
    c_void_p,  # values array
    c_void_p,  # rowind array
    c_void_p,  # colptr array
    c_int,  # Number of rows
    c_int,  # Number of columns
    c_int,  # Number of non-zeros
    c_void_p,  # RHS array
    c_void_p  # Solution array (output)
)
c_solve_sparse_system = functype(addr)


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
    n_rows = len(b)
    n_cols = len(csc_indptr) - 1
    nnz = len(csc_data)

    # Prepare solution array
    x = np.zeros_like(b)

    # Call the C function
    info = c_solve_sparse_system(
        csc_data.ctypes.data,
        csc_indices.ctypes.data,
        csc_indptr.ctypes.data,
        n_rows,
        n_cols,
        nnz,
        b.ctypes.data,
        x.ctypes.data
    )

    return x, info


# Import conversion functions
from sparse_numba.superlu_numba_conversion import convert_coo_to_csc, convert_csr_to_csc


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

    # Convert COO to CSC directly
    # We're not going through the test script's custom conversion function
    csc_data, csc_indices, csc_indptr = convert_coo_to_csc(
        row_indices, col_indices, data, n_rows, n_cols
    )

    # Solve using the CSC format
    return superlu_solve_csc(csc_data, csc_indices, csc_indptr, b)


@njit
def superlu_solve_csr(csr_data, csr_indices, csr_indptr, b):
    """
    Numba-compatible version that first sorts the CSR indices,
    then passes to the existing superlu_solve_csr function.

    Parameters are the same as superlu_solve_csc.
    """

    # Sort indices within each row
    # sorted_data, sorted_indices, n_cols = sort_csr_indices(csr_data, csr_indices, csr_indptr)

    # n_rows = len(b)

    # Convert Sorted CSR to CSC directly
    # We're not going through the test script's custom conversion function
    csc_data, csc_indices, csc_indptr = convert_csr_to_csc(csr_data, csr_indices, csr_indptr)

    return superlu_solve_csc(csc_data, csc_indices, csc_indptr, b)
