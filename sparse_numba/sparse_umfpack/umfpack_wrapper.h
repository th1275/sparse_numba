/* umfpack_wrapper.h - A minimal wrapper for UMFPACK to be used with Numba */
#ifndef UMFPACK_WRAPPER_H
#define UMFPACK_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Main solver function - taking CSC format matrix and solving Ax=b */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution);

/* Pre-factorize a sparse matrix (symbolic + numeric), returns opaque handle */
int factorize_sparse_system(double *values, int *rowind, int *colptr,
                            int nrows, int ncols, int nnz,
                            int64_t *handle_out);

/* Solve using pre-computed factors */
int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs);

/* Free memory associated with factors */
int free_sparse_factors(int64_t handle);

#ifdef __cplusplus
}
#endif

#endif /* UMFPACK_WRAPPER_H */
