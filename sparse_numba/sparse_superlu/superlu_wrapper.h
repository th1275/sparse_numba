/* superlu_wrapper.h - Header file for SuperLU wrapper */
#ifndef SUPERLU_WRAPPER_H
#define SUPERLU_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @return          0 on success, non-zero error code on failure
 */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                              int nrows, int ncols, int nnz,
                              double *rhs, double *solution);

/**
 * Pre-factorize a sparse matrix (LU decomposition only, no solve)
 *
 * @param values     Array of non-zero values in CSC format (size nnz)
 * @param rowind     Array of row indices (size nnz)
 * @param colptr     Array of column pointers (size ncols+1)
 * @param nrows      Number of rows in the matrix
 * @param ncols      Number of columns in the matrix
 * @param nnz        Number of non-zero elements
 * @param handle_out Output: opaque handle to LU factors (int64)
 * @return           0 on success, non-zero error code on failure
 */
int factorize_sparse_system(double *values, int *rowind, int *colptr,
                            int nrows, int ncols, int nnz,
                            int64_t *handle_out);

/**
 * Solve using pre-computed LU factors from factorize_sparse_system
 *
 * @param handle    Opaque handle from factorize_sparse_system
 * @param rhs       Right-hand side vector (size nrows)
 * @param solution  Output: Solution vector (size nrows)
 * @param nrhs      Number of right-hand sides (typically 1)
 * @return          0 on success, non-zero error code on failure
 */
int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs);

/**
 * Free memory associated with LU factors
 *
 * @param handle    Opaque handle from factorize_sparse_system
 * @return          0 on success, non-zero error code on failure
 */
int free_sparse_factors(int64_t handle);

#ifdef __cplusplus
}
#endif

#endif /* SUPERLU_WRAPPER_H */