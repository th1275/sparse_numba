# cython: language_level=3

from libc.stdint cimport int64_t

cdef extern from "superlu_wrapper.h":
    int solve_sparse_system(double *values, int *rowind, int *colptr,
                              int nrows, int ncols, int nnz,
                              double *rhs, double *solution)
    int factorize_sparse_system(double *values, int *rowind, int *colptr,
                                int nrows, int ncols, int nnz,
                                int64_t *handle_out)
    int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs)
    int free_sparse_factors(int64_t handle)
    int factorize_sparse_system_ordered(double *values, int *rowind, int *colptr,
                                        int n, int nnz, int ordering,
                                        int64_t *handle_out)
    int refactorize_sparse_system(int64_t handle, double *values, int nnz)
    int solve_into_with_factors(int64_t handle, double *rhs, double *solution,
                                int n, int nrhs)
    int sparse_factors_info(int64_t handle, int64_t *iout, double *dout)
    const char *superlu_build_info()


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


cdef api int cy_factorize_ordered(double *values, int *rowind, int *colptr,
                                  int n, int nnz, int ordering, int64_t *handle_out):
    return factorize_sparse_system_ordered(values, rowind, colptr, n, nnz, ordering,
                                           handle_out)


cdef api int cy_refactorize(int64_t handle, double *values, int nnz):
    return refactorize_sparse_system(handle, values, nnz)


cdef api int cy_solve_into(int64_t handle, double *rhs, double *solution, int n, int nrhs):
    return solve_into_with_factors(handle, rhs, solution, n, nrhs)


cdef api int cy_handle_info(int64_t handle, int64_t *iout, double *dout):
    return sparse_factors_info(handle, iout, dout)


def build_info():
    """The SuperLU version and this shim's compiler flags (for cache_key())."""
    return superlu_build_info().decode("ascii")
