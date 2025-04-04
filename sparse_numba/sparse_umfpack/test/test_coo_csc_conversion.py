"""
Conversion Test.

"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the MIT License.
#
#  File name: test_coo_csc_conversion.py

import numpy as np
import scipy.sparse as sp


def test_coo_to_csc_conversion():
    """
    Test the conversion from COO to CSC format using a simple example.
    """
    # Create a small test matrix in COO format
    row = np.array([0, 3, 1, 0, 2, 1, 3], dtype=np.int32)
    col = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
    data = np.array([1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0], dtype=np.float64)

    # Matrix shape
    n_rows, n_cols = 4, 4

    print("Original COO matrix:")
    print(f"Row indices: {row}")
    print(f"Col indices: {col}")
    print(f"Data: {data}")

    # Create scipy sparse matrix for reference
    A_coo = sp.coo_matrix((data, (row, col)), shape=(n_rows, n_cols))
    A_csc = A_coo.tocsc()

    print("\nSciPy CSC conversion (reference):")
    print(f"Data: {A_csc.data}")
    print(f"Indices: {A_csc.indices}")
    print(f"Indptr: {A_csc.indptr}")

    # Manual implementation for testing
    # Sort by column first
    idx_sort = np.argsort(col)
    sorted_row = row[idx_sort]
    sorted_col = col[idx_sort]
    sorted_data = data[idx_sort]

    # Count elements per column
    col_counts = np.zeros(n_cols + 1, dtype=np.int32)
    for j in sorted_col:
        col_counts[j + 1] += 1

    # Cumulative sum to get column pointers
    col_ptr = np.zeros(n_cols + 1, dtype=np.int32)
    for j in range(1, n_cols + 1):
        col_ptr[j] = col_ptr[j - 1] + col_counts[j]

    # Initialize CSC arrays
    csc_data = np.zeros_like(data)
    csc_indices = np.zeros_like(row)

    # Fill CSC arrays
    col_offsets = np.zeros(n_cols, dtype=np.int32)
    for i in range(len(sorted_data)):
        j = sorted_col[i]
        offset = col_offsets[j]
        idx = col_ptr[j] + offset
        csc_data[idx] = sorted_data[i]
        csc_indices[idx] = sorted_row[i]
        col_offsets[j] += 1

    # Sort rows within each column
    for j in range(n_cols):
        start = col_ptr[j]
        end = col_ptr[j + 1]
        if end > start:
            idx = np.argsort(csc_indices[start:end])
            csc_indices[start:end] = csc_indices[start:end][idx]
            csc_data[start:end] = csc_data[start:end][idx]

    print("\nManual CSC conversion implementation:")
    print(f"Data: {csc_data}")
    print(f"Indices: {csc_indices}")
    print(f"Indptr: {col_ptr}")

    # Verify our manual implementation matches scipy
    data_match = np.allclose(csc_data, A_csc.data)
    indices_match = np.array_equal(csc_indices, A_csc.indices)
    indptr_match = np.array_equal(col_ptr, A_csc.indptr)

    print("\nValidation:")
    print(f"Data matches: {data_match}")
    print(f"Indices match: {indices_match}")
    print(f"Indptr matches: {indptr_match}")

    # Create a dense representation for visual comparison
    dense_coo = A_coo.toarray()
    print("\nDense representation:")
    print(dense_coo)

    return data_match and indices_match and indptr_match


def run_test():
    test_passed = test_coo_to_csc_conversion()
    print(f"\nTest {'passed' if test_passed else 'failed'}")
