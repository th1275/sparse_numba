/* superlu_wrapper.h - Header file for SuperLU wrapper */
#ifndef SUPERLU_WRAPPER_H
#define SUPERLU_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve a sparse linear system using SuperLU
 *
 * This function solves the linear system Ax = b where A is a sparse matrix 
 * in CSC (Compressed Sparse Column) format.
 *
 * @param values    Array of non-zero values in the matrix (size nnz)
 * @param rowind    Array of row indices (size nnz)
 * @param colptr    Array of column pointers (size ncols+1)
 * @param nrows     Number of rows in the matrix
 * @param ncols     Number of columns in the matrix
 * @param nnz       Number of non-zero elements in the matrix
 * @param rhs       Right-hand side vector (size nrows)
 * @param solution  Output: Solution vector (size nrows)
 *
 * @return          0 on success, non-zero error code on failure
 */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                              int nrows, int ncols, int nnz,
                              double *rhs, double *solution);

#ifdef __cplusplus
}
#endif

#endif /* SUPERLU_WRAPPER_H */