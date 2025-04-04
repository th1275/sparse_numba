# cython: language_level=3

cdef extern from "umfpack_wrapper.h":
    int solve_sparse_system(double *values, int *rowind, int *colptr,
                          int nrows, int ncols, int nnz,
                          double *rhs, double *solution)


cdef api int cy_solve_sparse_system(double *values, int *rowind, int *colptr,
                                  int nrows, int ncols, int nnz,
                                  double *rhs, double *solution):
    return solve_sparse_system(values, rowind, colptr, nrows, ncols, nnz, rhs, solution)
