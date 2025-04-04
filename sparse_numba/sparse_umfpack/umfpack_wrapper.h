/* umfpack_wrapper.h - A minimal wrapper for UMFPACK to be used with Numba */
#ifndef UMFPACK_WRAPPER_H
#define UMFPACK_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Main solver function - taking CSC format matrix and solving Ax=b */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution);

#ifdef __cplusplus
}
#endif

#endif /* UMFPACK_WRAPPER_H */
