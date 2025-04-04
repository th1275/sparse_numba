"""
Conversion functions module.
    convert_coo_to_csc
    convert_csr_to_csc
"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the MIT License.
#
#  File name: matrix_conversion_numba.py

import numpy as np
from numba import njit, types


@njit
def convert_coo_to_csc(row_indices, col_indices, data, n_rows, n_cols):
    """
    Convert COO format to CSC format required by UMFPACK.
    This version handles duplicate entries by summing their values.

    Parameters:
    -----------
    row_indices : ndarray
        Row indices for COO format
    col_indices : ndarray
        Column indices for COO format
    data : ndarray
        Nonzero values in COO format
    n_rows : int
        Number of rows in the matrix
    n_cols : int
        Number of columns in the matrix

    Returns:
    --------
    data_csc : ndarray
        Nonzero values in CSC format
    row_indices_csc : ndarray
        Row indices in CSC format
    col_ptr : ndarray
        Column pointers in CSC format
    """
    nnz = len(data)

    # Allocate arrays for working with the data
    sorted_row_indices = np.zeros_like(row_indices)
    sorted_col_indices = np.zeros_like(col_indices)
    sorted_data = np.zeros_like(data)

    # First, count entries per column to compute column pointers
    col_counts = np.zeros(n_cols, dtype=np.int32)
    for i in range(nnz):
        col_counts[col_indices[i]] += 1

    # Calculate column pointers via cumulative sum
    col_ptr = np.zeros(n_cols + 1, dtype=np.int32)
    for i in range(n_cols):
        col_ptr[i + 1] = col_ptr[i] + col_counts[i]

    # Reset counts for use in the next step
    col_counts[:] = 0

    # Distribute elements to their correct positions in the sorted arrays
    # This effectively sorts by column
    for i in range(nnz):
        col = col_indices[i]
        dest = col_ptr[col] + col_counts[col]

        sorted_col_indices[dest] = col
        sorted_row_indices[dest] = row_indices[i]
        sorted_data[dest] = data[i]

        col_counts[col] += 1

    # Now for each column, sort by row index
    for j in range(n_cols):
        start = col_ptr[j]
        end = col_ptr[j + 1]

        # Skip empty columns or columns with just one entry
        if end - start <= 1:
            continue

        # Simple insertion sort for row indices within each column
        for i in range(start + 1, end):
            key_row = sorted_row_indices[i]
            key_data = sorted_data[i]

            # Move elements greater than key_row one position ahead
            k = i - 1
            while k >= start and sorted_row_indices[k] > key_row:
                sorted_row_indices[k + 1] = sorted_row_indices[k]
                sorted_data[k + 1] = sorted_data[k]
                k -= 1

            # Place the key at its correct position
            sorted_row_indices[k + 1] = key_row
            sorted_data[k + 1] = key_data

    # Now handle duplicates by summing values with the same (row, col)
    # First, count the number of unique entries
    unique_count = 0
    i = 0

    while i < nnz:
        unique_count += 1
        curr_row = sorted_row_indices[i]
        curr_col = sorted_col_indices[i]

        # Skip over duplicate entries
        while i + 1 < nnz and sorted_col_indices[i + 1] == curr_col and sorted_row_indices[i + 1] == curr_row:
            i += 1

        i += 1

    # Allocate final CSC arrays
    final_data = np.zeros(unique_count, dtype=data.dtype)
    final_indices = np.zeros(unique_count, dtype=row_indices.dtype)
    final_indptr = np.zeros(n_cols + 1, dtype=np.int32)

    # Fill the final arrays, summing duplicates
    unique_idx = 0
    i = 0

    while i < nnz:
        curr_row = sorted_row_indices[i]
        curr_col = sorted_col_indices[i]

        # Sum values for duplicate entries
        value_sum = sorted_data[i]
        i += 1

        while i < nnz and sorted_col_indices[i] == curr_col and sorted_row_indices[i] == curr_row:
            value_sum += sorted_data[i]
            i += 1

        # Record the unique entry
        final_data[unique_idx] = value_sum
        final_indices[unique_idx] = curr_row

        # Update column count
        final_indptr[curr_col + 1] += 1

        unique_idx += 1

    # Compute final column pointers
    for j in range(1, n_cols + 1):
        final_indptr[j] += final_indptr[j - 1]

    return final_data, final_indices, final_indptr


@njit
def convert_csr_to_csc(csr_data, csr_indices, csr_indptr):
    """
    Convert CSR format to CSC format while ensuring the indices are sorted.
    """
    n_rows = len(csr_indptr) - 1
    nnz = len(csr_data)

    # Determine n_cols by finding the maximum column index
    n_cols = 0
    for i in range(nnz):
        col = csr_indices[i]
        if col > n_cols:
            n_cols = col
    n_cols += 1  # Adjust for 0-indexing

    # Initialize CSC arrays
    csc_data = np.zeros_like(csr_data)
    csc_indices = np.zeros_like(csr_indices)
    csc_indptr = np.zeros(n_cols + 1, dtype=np.int32)

    # Count number of entries in each column
    for i in range(nnz):
        col = csr_indices[i]
        csc_indptr[col + 1] += 1

    # Cumulative sum to get column pointers
    for i in range(1, n_cols + 1):
        csc_indptr[i] += csc_indptr[i - 1]

    # Create a copy of indptr to keep track of current position in each column
    col_counts = np.zeros(n_cols, dtype=np.int32)

    # Fill in CSC data and row indices (first pass)
    for i in range(n_rows):
        for j in range(csr_indptr[i], csr_indptr[i + 1]):
            col = csr_indices[j]
            val = csr_data[j]

            pos = csc_indptr[col] + col_counts[col]
            csc_data[pos] = val
            csc_indices[pos] = i
            col_counts[col] += 1

    # Now sort row indices within each column
    for j in range(n_cols):
        start = csc_indptr[j]
        end = csc_indptr[j + 1]

        # Skip empty columns or columns with just one entry
        if end - start <= 1:
            continue

        # Simple insertion sort for row indices within this column
        for i in range(start + 1, end):
            key_row = csc_indices[i]
            key_data = csc_data[i]

            k = i - 1
            while k >= start and csc_indices[k] > key_row:
                csc_indices[k + 1] = csc_indices[k]
                csc_data[k + 1] = csc_data[k]
                k -= 1

            csc_indices[k + 1] = key_row
            csc_data[k + 1] = key_data

    # Handle duplicate entries by summing their values
    if nnz > 0:  # Only process if there are non-zero elements
        # Create new arrays to hold the result
        new_nnz = nnz  # Start with the same size, will adjust later
        new_data = np.zeros_like(csc_data)
        new_indices = np.zeros_like(csc_indices)
        new_indptr = np.zeros_like(csc_indptr)

        # Initialize the first column pointer
        new_indptr[0] = 0
        new_pos = 0

        # Process each column
        for j in range(n_cols):
            start = csc_indptr[j]
            end = csc_indptr[j + 1]

            if start == end:  # Empty column
                new_indptr[j + 1] = new_pos
                continue

            # Process the first entry in this column
            prev_row = csc_indices[start]
            sum_val = csc_data[start]

            # Process remaining entries in this column
            for i in range(start + 1, end):
                curr_row = csc_indices[i]

                if curr_row == prev_row:  # Duplicate entry
                    # Add to the sum
                    sum_val += csc_data[i]
                else:
                    # Write the previous entry
                    new_indices[new_pos] = prev_row
                    new_data[new_pos] = sum_val
                    new_pos += 1

                    # Start a new entry
                    prev_row = curr_row
                    sum_val = csc_data[i]

            # Write the last entry for this column
            new_indices[new_pos] = prev_row
            new_data[new_pos] = sum_val
            new_pos += 1

            # Update column pointer
            new_indptr[j + 1] = new_pos

        # Trim arrays to the actual size
        return new_data[:new_pos], new_indices[:new_pos], new_indptr

    return csc_data, csc_indices, csc_indptr


@njit
def validate_sparse_matrix(data, indices, indptr=None, shape=None):
    """
    Validate sparse matrix inputs and return dimensions.

    Parameters:
    -----------
    data : ndarray
        Nonzero values
    indices : ndarray
        Column indices (for CSR) or row indices (for CSC)
    indptr : ndarray, optional
        Row pointers (for CSR) or column pointers (for CSC)
    shape : tuple, optional
        Shape of the matrix as (n_rows, n_cols)

    Returns:
    --------
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
    """
    if indptr is not None:
        # CSR or CSC format
        if shape is None:
            n_rows = len(indptr) - 1
            # Estimate n_cols from maximum index
            n_cols = 0
            for i in range(len(indices)):
                if indices[i] > n_cols:
                    n_cols = indices[i]
            n_cols += 1
        else:
            n_rows, n_cols = shape
    else:
        # COO format requires shape
        if shape is None:
            raise ValueError("Shape must be provided for COO format")
        n_rows, n_cols = shape

    return n_rows, n_cols


@njit
def check_matrix_properties(csc_data, csc_indices, csc_indptr, n_rows):
    """
    Check matrix properties for potential issues with ill-conditioning.

    Parameters:
    -----------
    csc_data : ndarray
        Nonzero values in CSC format
    csc_indices : ndarray
        Row indices in CSC format
    csc_indptr : ndarray
        Column pointers in CSC format
    n_rows : int
        Number of rows in the matrix

    Returns:
    --------
    is_singular : bool
        True if matrix appears singular
    condition_est : float
        Rough estimate of condition number (infinity norm)
    diag_ratio : float
        Ratio of min/max diagonal element (0.0 if no diagonal)
    """
    n_cols = len(csc_indptr) - 1
    nnz = len(csc_data)
    min_dim = min(n_rows, n_cols)

    # Find diagonal elements
    diag_values = np.zeros(min_dim)
    has_diag = np.zeros(min_dim, dtype=np.int32)

    for j in range(min_dim):
        col_start = csc_indptr[j]
        col_end = csc_indptr[j + 1]

        for k in range(col_start, col_end):
            i = csc_indices[k]
            if i == j:  # Diagonal element
                diag_values[j] = csc_data[k]
                has_diag[j] = 1
                break

    # Count missing diagonals
    missing_diags = min_dim - np.sum(has_diag)

    # Find min/max diagonal values (for those that exist)
    min_diag = np.inf
    max_diag = 0.0
    for j in range(min_dim):
        if has_diag[j]:
            abs_diag = abs(diag_values[j])
            if abs_diag < min_diag:
                min_diag = abs_diag
            if abs_diag > max_diag:
                max_diag = abs_diag

    # Calculate row and column norms (infinity norm)
    row_norms = np.zeros(n_rows)
    col_norms = np.zeros(n_cols)

    for j in range(n_cols):
        col_start = csc_indptr[j]
        col_end = csc_indptr[j + 1]

        for k in range(col_start, col_end):
            i = csc_indices[k]
            abs_val = abs(csc_data[k])
            row_norms[i] += abs_val
            col_norms[j] += abs_val

    # Find max norms
    max_row_norm = 0.0
    max_col_norm = 0.0
    for i in range(n_rows):
        if row_norms[i] > max_row_norm:
            max_row_norm = row_norms[i]
    for j in range(n_cols):
        if col_norms[j] > max_col_norm:
            max_col_norm = col_norms[j]

    # Estimate condition number using norms
    condition_est = max_row_norm * max_col_norm
    if condition_est == 0.0:
        condition_est = np.inf

    # Calculate diagonal ratio
    diag_ratio = 0.0
    if min_diag < np.inf and max_diag > 0.0:
        diag_ratio = min_diag / max_diag

    # Determine if likely singular
    is_singular = (missing_diags > 0) or (diag_ratio < 1e-10) or (condition_est > 1e15)

    return is_singular, condition_est, diag_ratio


def validate_csc_matrix(data, indices, indptr, nrows, ncols):
    """Validate CSC matrix format for UMFPACK."""
    nnz = len(data)

    # Check indptr properties
    if indptr[0] != 0:
        print(f"Error: First element of indptr must be 0, got {indptr[0]}")
        return False

    if indptr[ncols] != nnz:
        print(f"Error: Last element of indptr must equal nnz, got {indptr[ncols]} vs {nnz}")
        return False

    # Check monotonicity
    for i in range(1, len(indptr)):
        if indptr[i] < indptr[i - 1]:
            print(f"Error: indptr not monotonically increasing at position {i}")
            return False

    # Check row indices
    if np.min(indices) < 0 or np.max(indices) >= nrows:
        print(f"Error: Row indices out of bounds: min={np.min(indices)}, max={np.max(indices)}, nrows={nrows}")
        return False

    # Check sorted row indices within columns
    for j in range(ncols):
        start, end = indptr[j], indptr[j + 1]
        col_indices = indices[start:end]

        if len(col_indices) > 1:
            for i in range(1, len(col_indices)):
                if col_indices[i] <= col_indices[i - 1]:
                    print(f"Error: Row indices not strictly increasing in column {j}")
                    print(f"  Indices: {col_indices}")
                    return False

    # Check for small pivot elements
    has_diagonal = np.zeros(min(nrows, ncols), dtype=bool)
    for j in range(min(nrows, ncols)):
        start, end = indptr[j], indptr[j + 1]
        for p in range(start, end):
            if indices[p] == j:  # Diagonal element
                has_diagonal[j] = True
                if abs(data[p]) < 1e-10:
                    print(f"Warning: Small diagonal element at position ({j},{j}): {data[p]}")

    missing_diags = np.where(~has_diagonal)[0]
    if len(missing_diags) > 0:
        print(f"Warning: Missing diagonal elements at positions: {missing_diags[:10]}...")

    return True
