# cython: language_level=3

from libc.stdint cimport int64_t

cdef extern from "umfpack_wrapper.h":
    int solve_sparse_system(double *values, int *rowind, int *colptr,
                          int nrows, int ncols, int nnz,
                          double *rhs, double *solution)
    int factorize_sparse_system(double *values, int *rowind, int *colptr,
                                int nrows, int ncols, int nnz,
                                int64_t *handle_out)
    int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs)
    int free_sparse_factors(int64_t handle)


cdef api int cy_solve_sparse_system(double *values, int *rowind, int *colptr,
                                  int nrows, int ncols, int nnz,
                                  double *rhs, double *solution):
    return solve_sparse_system(values, rowind, colptr, nrows, ncols, nnz, rhs, solution)


cdef api int cy_factorize_sparse_system(double *values, int *rowind, int *colptr,
                                        int nrows, int ncols, int nnz,
                                        int64_t *handle_out):
    return factorize_sparse_system(values, rowind, colptr, nrows, ncols, nnz, handle_out)


cdef api int cy_solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs):
    return solve_with_factors(handle, rhs, solution, nrhs)


cdef api int cy_free_sparse_factors(int64_t handle):
    return free_sparse_factors(handle)
