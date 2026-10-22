# cython: language_level=3
# distutils: language = c++

from libc.stdint cimport int64_t

cdef extern from "eigen_wrapper.h":
    int eigen_lu_factorize_ordered(double *values, int *rowind, int *colptr,
                                   int n, int nnz, int ordering, int64_t *handle_out)
    int eigen_ldlt_factorize_ordered(double *values, int *rowind, int *colptr,
                                     int n, int nnz, int ordering, int64_t *handle_out)
    int eigen_refactorize(int64_t handle, double *values, int nnz)
    int eigen_solve_into(int64_t handle, double *rhs, double *solution, int n, int nrhs)
    int eigen_factors_info(int64_t handle, int64_t *iout, double *dout)
    int eigen_free_factors(int64_t handle)
    const char *eigen_build_info()


cdef api int cy_lu_factorize_ordered(double *values, int *rowind, int *colptr,
                                     int n, int nnz, int ordering, int64_t *handle_out):
    return eigen_lu_factorize_ordered(values, rowind, colptr, n, nnz, ordering, handle_out)


cdef api int cy_ldlt_factorize_ordered(double *values, int *rowind, int *colptr,
                                       int n, int nnz, int ordering, int64_t *handle_out):
    return eigen_ldlt_factorize_ordered(values, rowind, colptr, n, nnz, ordering, handle_out)


cdef api int cy_refactorize(int64_t handle, double *values, int nnz):
    return eigen_refactorize(handle, values, nnz)


cdef api int cy_solve_into(int64_t handle, double *rhs, double *solution, int n, int nrhs):
    return eigen_solve_into(handle, rhs, solution, n, nrhs)


cdef api int cy_handle_info(int64_t handle, int64_t *iout, double *dout):
    return eigen_factors_info(handle, iout, dout)


cdef api int cy_free_factors(int64_t handle):
    return eigen_free_factors(handle)


def build_info():
    """The Eigen version and this shim's build facts (for cache_key())."""
    return eigen_build_info().decode("ascii")
