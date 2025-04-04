import numpy as np
from numba import njit
import time
from superlu_numba_interface import superlu_solve_csr, superlu_solve_coo, superlu_solve_csr_revised
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Import the revised CSR solver
# You'll need to save the fixed-superlu-interface-2.py file first
# from fixed_superlu_interface_2 import superlu_solve_csr_revised


def main():
    """
    Testing Numba-compatible SuperLU with revised CSR solver
    """
    print("Testing Numba-compatible SuperLU solver with revised CSR solver")

    # Create a simple test problem
    n = 100
    print(f"Creating a sparse matrix of size {n}x{n}")

    # Create a banded matrix in COO format (same as before)
    row_indices = []
    col_indices = []
    data = []

    # Main diagonal and two off-diagonals
    for i in range(n):
        # Main diagonal
        row_indices.append(i)
        col_indices.append(i)
        data.append(2.0)

        # Lower diagonal
        if i > 0:
            row_indices.append(i)
            col_indices.append(i - 1)
            data.append(-1.0)

        # Upper diagonal
        if i < n - 1:
            row_indices.append(i)
            col_indices.append(i + 1)
            data.append(-1.0)

    # Convert to numpy arrays
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data = np.array(data, dtype=np.float64)

    # Create the SciPy sparse matrix (for reference)
    A_scipy = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    A_scipy_csr = A_scipy.tocsr()

    # Create a right-hand side vector
    b = np.ones(n, dtype=np.float64)

    # === SciPy reference solution ===
    print("\nSolving with SciPy (reference)...")
    start_time = time.time()
    x_scipy = spsolve(A_scipy_csr, b)  # Using CSR to avoid warning
    scipy_time = time.time() - start_time
    print(f"SciPy solution time: {scipy_time:.6f} seconds")

    # === Custom SuperLU with COO format ===
    print("\nSolving with COO format...")
    start_time = time.time()
    x_coo, info_coo = superlu_solve_coo(row_indices, col_indices, data, (n, n), b)
    coo_time = time.time() - start_time
    print(f"COO format solution time: {coo_time:.6f} seconds")
    print(f"Solution status: {'Success' if info_coo == 0 else 'Failed'}")

    # === Convert to CSR format ===
    print("\nConverting to CSR format...")
    csr_data, csr_indices, csr_indptr = coo_to_csr(data, row_indices, col_indices, n)

    # === Original CSR solver ===
    print("\nSolving with original CSR solver...")
    start_time = time.time()
    x_csr_original, info_csr_original = superlu_solve_csr(csr_data, csr_indices, csr_indptr, b)
    csr_original_time = time.time() - start_time
    print(f"Original CSR format solution time: {csr_original_time:.6f} seconds")
    print(f"Solution status: {'Success' if info_csr_original == 0 else 'Failed'}")

    # === Revised CSR solver ===
    print("\nSolving with revised CSR solver...")
    start_time = time.time()
    x_csr_revised, info_csr_revised = superlu_solve_csr_revised(csr_data, csr_indices, csr_indptr, b)
    csr_revised_time = time.time() - start_time
    print(f"Revised CSR format solution time: {csr_revised_time:.6f} seconds")
    print(f"Solution status: {'Success' if info_csr_revised == 0 else 'Failed'}")

    # === Compare solutions ===
    print("\nComparing solutions:")
    coo_scipy_diff = np.max(np.abs(x_coo - x_scipy))
    csr_orig_scipy_diff = np.max(np.abs(x_csr_original - x_scipy))
    csr_revised_scipy_diff = np.max(np.abs(x_csr_revised - x_scipy))
    coo_csr_orig_diff = np.max(np.abs(x_coo - x_csr_original))
    coo_csr_revised_diff = np.max(np.abs(x_coo - x_csr_revised))

    print(f"Max difference between COO and SciPy: {coo_scipy_diff:.6e}")
    print(f"Max difference between original CSR and SciPy: {csr_orig_scipy_diff:.6e}")
    print(f"Max difference between revised CSR and SciPy: {csr_revised_scipy_diff:.6e}")
    print(f"Max difference between COO and original CSR: {coo_csr_orig_diff:.6e}")
    print(f"Max difference between COO and revised CSR: {coo_csr_revised_diff:.6e}")

    if coo_csr_revised_diff < 1e-10:
        print("COO and revised CSR solutions match!")
    else:
        print("WARNING: COO and revised CSR solutions still differ!")

    # Check solution accuracy (for this problem, the solution should be all 1's)
    expected = np.ones_like(x_scipy)
    scipy_error = np.max(np.abs(x_scipy - expected))
    coo_error = np.max(np.abs(x_coo - expected))
    csr_orig_error = np.max(np.abs(x_csr_original - expected))
    csr_revised_error = np.max(np.abs(x_csr_revised - expected))

    print(f"\nMaximum solution errors:")
    print(f"SciPy: {scipy_error:.6e}")
    print(f"COO: {coo_error:.6e}")
    print(f"Original CSR: {csr_orig_error:.6e}")
    print(f"Revised CSR: {csr_revised_error:.6e}")

    print("\nNumba-compatible SuperLU testing completed!")


@njit
def coo_to_csr(data, row, col, n_rows):
    """
    Convert COO format to CSR format.

    Parameters:
    -----------
    data : ndarray
        Values in COO format
    row : ndarray
        Row indices in COO format
    col : ndarray
        Column indices in COO format
    n_rows : int
        Number of rows

    Returns:
    --------
    csr_data : ndarray
        Values in CSR format
    csr_indices : ndarray
        Column indices in CSR format
    csr_indptr : ndarray
        Row pointers in CSR format
    """
    nnz = len(data)

    # Initialize CSR arrays
    csr_data = np.zeros_like(data)
    csr_indices = np.zeros_like(col)
    csr_indptr = np.zeros(n_rows + 1, dtype=np.int32)

    # Count number of elements in each row
    for i in range(nnz):
        csr_indptr[row[i] + 1] += 1

    # Cumulative sum to get row pointers
    for i in range(1, n_rows + 1):
        csr_indptr[i] += csr_indptr[i - 1]

    # Fill in CSR data and indices
    row_counts = np.zeros(n_rows, dtype=np.int32)
    for i in range(nnz):
        r = row[i]
        pos = csr_indptr[r] + row_counts[r]
        csr_data[pos] = data[i]
        csr_indices[pos] = col[i]
        row_counts[r] += 1

    return csr_data, csr_indices, csr_indptr


if __name__ == "__main__":
    main()