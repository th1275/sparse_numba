#ifndef SUPERLU_WRAPPER_H
#define SUPERLU_WRAPPER_H

int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution);

#endif