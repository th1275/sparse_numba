/* superlu_wrapper.h - Header file for SuperLU wrapper */
#ifndef SUPERLU_WRAPPER_H
#define SUPERLU_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Every function returns a status code of sparse_numba_codes.h: SN_OK (0),
 * a negative code by name, or SINGULAR = k > 0 (dgstrf's info: U(k, k) is
 * exactly zero). A handle is an index into this module's registry,
 * (generation << 32) | (SN_BACKEND_SUPERLU << 24) | slot, never a pointer:
 * a stale handle returns SN_HANDLE_FREED, another backend's handle
 * SN_HANDLE_WRONG_BACKEND, zero, a negative or a forged one
 * SN_HANDLE_INVALID. A handle is process-local.
 */

/**
 * Solve a sparse linear system using SuperLU (combined factorize + solve)
 *
 * @param values    Array of non-zero values in CSC format (size nnz)
 * @param rowind    Array of row indices (size nnz)
 * @param colptr    Array of column pointers (size ncols+1)
 * @param nrows     Number of rows in the matrix
 * @param ncols     Number of columns in the matrix
 * @param nnz       Number of non-zero elements
 * @param rhs       Right-hand side vector (size nrows)
 * @param solution  Output: Solution vector (size nrows)
 * @return          a status code
 */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                              int nrows, int ncols, int nnz,
                              double *rhs, double *solution);

/**
 * Pre-factorize a sparse matrix (LU decomposition only, no solve) at
 * ColPerm NATURAL (the 0.1.11 factorization)
 *
 * @param values     Array of non-zero values in CSC format (size nnz)
 * @param rowind     Array of row indices (size nnz)
 * @param colptr     Array of column pointers (size ncols+1)
 * @param nrows      Number of rows in the matrix
 * @param ncols      Number of columns in the matrix
 * @param nnz        Number of non-zero elements
 * @param handle_out Output: the handle (int64), 0 on failure
 * @return           a status code
 */
int factorize_sparse_system(double *values, int *rowind, int *colptr,
                            int nrows, int ncols, int nnz,
                            int64_t *handle_out);

/**
 * Solve using pre-computed LU factors from factorize_sparse_system
 *
 * @param handle    the handle
 * @param rhs       Right-hand side, nrhs columns of n (column-major)
 * @param solution  Output: the solution, the same layout
 * @param nrhs      Number of right-hand sides (typically 1)
 * @return          a status code
 */
int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs);

/**
 * Free the factors of a handle; a second free returns SN_HANDLE_FREED
 *
 * @param handle    the handle
 * @return          a status code
 */
int free_sparse_factors(int64_t handle);

/**
 * Factorize an n x n CSC matrix at an ordering of sparse_numba_codes.h:
 * options.ColPerm NATURAL, COLAMD or MMD_AT_PLUS_A; SN_ORDERING_AMD (and
 * any other) SN_ORDERING_UNSUPPORTED. The CSC arrays are copied and held
 * for refactorize_sparse_system; the dense right-hand side and the
 * SuperLUStat_t are made here once and freed at free_sparse_factors.
 */
int factorize_sparse_system_ordered(double *values, int *rowind, int *colptr,
                                    int n, int nnz, int ordering,
                                    int64_t *handle_out);

/**
 * Factorize the handle's pattern again on new values (nnz of them, in the
 * pattern's order): dgstrf with options.Fact = SamePattern on the held
 * column permutation and elimination tree
 */
int refactorize_sparse_system(int64_t handle, double *values, int nnz);

/**
 * Solve into a caller-held solution: rhs and solution are (n, nrhs)
 * row-major (C-contiguous); n must equal the handle's dimension
 */
int solve_into_with_factors(int64_t handle, double *rhs, double *solution,
                            int n, int nrhs);

/**
 * The handle's facts: iout[SN_INFO_LEN] (n, nnz of L, nnz of U, backend id,
 * ordering id, the backend's status of the last factorization), dout[
 * SN_DINFO_LEN] (the pivot tolerance)
 */
int sparse_factors_info(int64_t handle, int64_t *iout, double *dout);

/* "<SuperLU version>,cc=...,optimize=...,fast_math=...,simd=...,fma=...,flags=...":
 * the SuperLU headers' version and this shim's compiler flags, for
 * cache_key() */
const char *superlu_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* SUPERLU_WRAPPER_H */
