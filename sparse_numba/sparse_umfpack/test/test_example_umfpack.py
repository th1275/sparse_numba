"""
Example usage of the UMFPACK-based sparse solver.
This demonstrates how to use the solver with different sparse matrix formats.
"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: test_example_umfpack.py

import numpy as np
import random
import scipy.sparse as sp
from sparse_numba.umfpack_numba_interface import (
    umfpack_solve_csc,
    umfpack_solve_csr,
    umfpack_solve_coo
)
from sparse_numba.conversion.matrix_conversion_numba import check_matrix_properties


def example_with_random_matrix():
    """
    Example with a random sparse matrix.
    """
    # Create a random sparse matrix (relatively well-conditioned)
    n = 100
    density = 0.01
    A_coo = sp.random(n, n, density=density, format='coo', dtype=float)

    # Add diagonal dominance to ensure it's non-singular
    diag_indices = np.arange(n)
    A_coo.data = np.concatenate([A_coo.data, np.ones(n) * 10])
    A_coo.row = np.concatenate([A_coo.row, diag_indices])
    A_coo.col = np.concatenate([A_coo.col, diag_indices])

    # Convert to different formats
    A_csr = A_coo.tocsr()
    A_csc = A_coo.tocsc()

    # Create a right-hand side
    x_true = np.ones(n)
    b = A_csc @ x_true

    print(f"Matrix size: {n}x{n}, NNZ: {len(A_coo.data)}")

    # Solve using CSC format
    x_csc, info_csc = umfpack_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )
    print(f"CSC solve status: {info_csc}")
    print(f"CSC solution error: {np.linalg.norm(x_csc - x_true)}")

    # Solve using CSR format
    x_csr, info_csr = umfpack_solve_csr(
        A_csr.data, A_csr.indices, A_csr.indptr, b
    )
    print(f"CSR solve status: {info_csr}")
    print(f"CSR solution error: {np.linalg.norm(x_csr - x_true)}")

    # Solve using COO format
    x_coo, info_coo = umfpack_solve_coo(
        A_coo.row, A_coo.col, A_coo.data, A_coo.shape, b
    )
    print(f"COO solve status: {info_coo}")
    print(f"COO solution error: {np.linalg.norm(x_coo - x_true)}")


def example_with_ill_conditioned_matrix():
    """
    Example with an ill-conditioned matrix to demonstrate robustness.
    """
    # # Create an ill-conditioned matrix
    # n = 100
    #
    # # Start with a random sparse matrix
    # A_coo = sp.random(n, n, density=0.05, format='coo', dtype=float)
    #
    # # Add small values to the diagonal to make it ill-conditioned
    # diag_indices = np.arange(n)
    # small_diags = np.ones(n) * 1e-8
    # small_diags[0] = 1.0  # One normal value to create ill-conditioning
    #
    # A_coo.data = np.concatenate([A_coo.data, small_diags])
    # A_coo.row = np.concatenate([A_coo.row, diag_indices])
    # A_coo.col = np.concatenate([A_coo.col, diag_indices])
    #
    # # Convert to CSC format (needed for check_matrix_properties)
    # A_csc = A_coo.tocsc()
    # Parameters for the test system
    n = 1000  # system size
    density = 0.1  # density for the random sparse matrix
    random.seed(42)
    # Create a random sparse matrix (n x n) in CSR format
    A_coo = sp.rand(n, n, density=density, format='coo', dtype=np.float64)
    # To avoid singular systems, add a diagonal shift
    A_coo = A_coo + 0.1 * sp.eye(n, format='coo')

    # A_sparse = 20*sp.eye(n, format='csr')

    A_dense = A_coo.toarray()
    cond_A = np.linalg.cond(A_dense)

    # # Convert to CSC format (needed for check_matrix_properties)
    A_csc = A_coo.tocsc()

    # Check matrix properties
    is_singular, condition_est, diag_ratio = check_matrix_properties(
        A_csc.data, A_csc.indices, A_csc.indptr, n
    )

    print(f"\nIll-conditioned matrix example:")
    print(f"Matrix size: {n}x{n}, NNZ: {len(A_coo.data)}")
    print(f"Is matrix singular? {is_singular}")
    print(f"Estimated condition number: {condition_est}")
    print("\nCondition number of A:", cond_A)
    print(f"Diagonal ratio (min/max): {diag_ratio}")

    # Create a right-hand side
    x_true = np.ones(n)
    b = A_csc @ x_true

    # Solve using CSC format
    x_csc, info_csc = umfpack_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )

    # Calculate relative error
    rel_error = np.linalg.norm(x_csc - x_true) / np.linalg.norm(x_true)

    print(f"UMFPACK solve status: {info_csc}")
    print(f"Relative solution error: {rel_error}")

    # Compare with scipy.sparse.linalg.spsolve
    try:
        from scipy.sparse.linalg import spsolve
        x_scipy = spsolve(A_csc, b)
        scipy_rel_error = np.linalg.norm(x_scipy - x_true) / np.linalg.norm(x_true)
        print(f"SciPy spsolve relative error: {scipy_rel_error}")
    except Exception as e:
        print(f"SciPy spsolve failed: {e}")


def example_with_nearly_singular_matrix():
    """
    Example with a nearly singular matrix with missing diagonal elements.
    """
    # Create a matrix with some missing diagonal elements
    n = 50
    A = np.eye(n, dtype=float)

    # Make a few rows/columns linearly dependent
    A[10, :] = A[0, :]  # Row 10 same as row 0
    A[20, :] = A[0, :] * 2  # Row 20 is 2 * row 0
    A[30, :] = A[0, :] * 3  # Row 30 is 3 * row 0

    # Remove some diagonal elements
    A[5, 5] = 0
    A[15, 15] = 0

    # Convert to sparse
    A_csc = sp.csc_matrix(A)

    # Check matrix properties
    is_singular, condition_est, diag_ratio = check_matrix_properties(
        A_csc.data, A_csc.indices, A_csc.indptr, n
    )

    print(f"\nNearly singular matrix example:")
    print(f"Matrix size: {n}x{n}, NNZ: {len(A_csc.data)}")
    print(f"Is matrix singular? {is_singular}")
    print(f"Estimated condition number: {condition_est}")
    print(f"Diagonal ratio (min/max): {diag_ratio}")

    # Create a right-hand side (should be in the range of A for a singular matrix)
    x_true = np.ones(n)
    b = A_csc @ x_true

    # Solve using UMFPACK
    x_umfpack, info_umfpack = umfpack_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )

    # Calculate relative error
    rel_error = np.linalg.norm(x_umfpack - x_true) / np.linalg.norm(x_true)

    print(f"UMFPACK solve status: {info_umfpack}")
    print(f"Relative solution error: {rel_error}")

    # Compare with SciPy
    try:
        from scipy.sparse.linalg import spsolve
        x_scipy = spsolve(A_csc, b)
        scipy_rel_error = np.linalg.norm(x_scipy - x_true) / np.linalg.norm(x_true)
        print(f"SciPy spsolve relative error: {scipy_rel_error}")
    except Exception as e:
        print(f"SciPy spsolve failed: {e}")


def run_test():
    print("Running examples of UMFPACK sparse solver...")
    example_with_random_matrix()
    example_with_ill_conditioned_matrix()
    # example_with_nearly_singular_matrix()
