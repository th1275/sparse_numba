"""
The 0.1.11 names, kept in 0.2.0 with the same signatures and return shapes:
code written for 0.1.11 runs unchanged. The one-shot solves (CSC, COO,
CSR), the factorize-once / solve-many trio, the matrix conversions, and the
availability checks -- for SuperLU, and for UMFPACK when the user's library
is found. Each call's x is compared with the backend-neutral API's.

    python examples/kept_0111_names.py
"""

import numpy as np

import sparse_numba as sn
from sparse_numba import (convert_coo_to_csc, convert_coo_to_csr, is_slu_available,
                          is_umf_available, sparse_matvec_csr, superlu_factorize_csc,
                          superlu_free_factors, superlu_solve_coo, superlu_solve_csc,
                          superlu_solve_csr, superlu_solve_factored, umfpack_factorize_csc,
                          umfpack_free_factors, umfpack_solve_csc, umfpack_solve_factored)


def coo_matrix():
    """A 5 x 5 unsymmetric matrix in COO (row, col, value)."""
    rows = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    cols = np.array([0, 3, 0, 1, 1, 2, 4, 2, 3, 0, 4], dtype=np.int32)
    vals = np.array([4.0, 1.0, -1.0, 5.0, 2.0, 6.0, -1.0, 1.0, 3.0, 2.0, 7.0])
    return rows, cols, vals, 5


def main():
    print("is_slu_available() = %s, is_umf_available() = %s"
          % (is_slu_available(), is_umf_available()))
    rows, cols, vals, n = coo_matrix()
    b = np.arange(1.0, n + 1.0)

    csc_data, csc_indices, csc_indptr = convert_coo_to_csc(rows, cols, vals, n, n)
    csr_data, csr_indices, csr_indptr = convert_coo_to_csr(rows, cols, vals, n, n)
    print("convert_coo_to_csc: indptr %s" % csc_indptr)
    print("convert_coo_to_csr: indptr %s" % csr_indptr)

    x_new, info = sn.solve(csc_data, csc_indices.astype(np.int32),
                           csc_indptr.astype(np.int32), n, b, sn.SUPERLU, sn.NATURAL)
    print("the 0.2.0 API, solve(..., SUPERLU, NATURAL): x = %s, info %d" % (x_new, info))

    x, info = superlu_solve_csc(csc_data, csc_indices, csc_indptr, b)
    print("superlu_solve_csc:  x = %s, info %d, max |x - x_new| %.1e"
          % (x, info, np.max(np.abs(x - x_new))))
    x, info = superlu_solve_coo(rows, cols, vals, (n, n), b)
    print("superlu_solve_coo:  info %d, max |x - x_new| %.1e" % (info, np.max(np.abs(x - x_new))))
    x, info = superlu_solve_csr(csr_data, csr_indices, csr_indptr, b)
    print("superlu_solve_csr:  info %d, max |x - x_new| %.1e" % (info, np.max(np.abs(x - x_new))))

    handle, info = superlu_factorize_csc(csc_data, csc_indices, csc_indptr)
    print("superlu_factorize_csc: info %d" % info)
    for k in range(3):
        bk = b * (k + 1)
        x, info = superlu_solve_factored(handle, bk)
        print("  superlu_solve_factored(b x %d): info %d, residual %.1e"
              % (k + 1, info, np.max(np.abs(sparse_matvec_csr(csr_data, csr_indices,
                                                               csr_indptr, x) - bk))))
    print("superlu_free_factors: %d" % superlu_free_factors(handle))

    if not is_umf_available():
        print("UMFPACK: the user's library is not found; the umfpack_* names return "
              "BACKEND_ABSENT (see examples/umfpack_walk.py)")
        return
    x, info = umfpack_solve_csc(csc_data, csc_indices, csc_indptr, b)
    print("umfpack_solve_csc:  info %d, max |x - x_new| %.1e" % (info, np.max(np.abs(x - x_new))))
    handle, info = umfpack_factorize_csc(csc_data, csc_indices, csc_indptr)
    x, info_solve = umfpack_solve_factored(handle, b)
    print("umfpack_factorize_csc / solve_factored: info %d / %d, max |x - x_new| %.1e; "
          "umfpack_free_factors: %d" % (info, info_solve, np.max(np.abs(x - x_new)),
                                        umfpack_free_factors(handle)))


if __name__ == "__main__":
    main()
