/* eigen_wrapper.h - Header file for the Eigen wrapper */
#ifndef EIGEN_WRAPPER_H
#define EIGEN_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The Eigen 3.4.0 backends of sparse_numba: EIGEN_LU =
 * Eigen::SparseLU at NaturalOrdering / COLAMDOrdering / AMDOrdering, and
 * EIGEN_LDLT = Eigen::SimplicialLDLT (Lower, AMDOrdering) for a symmetric
 * pattern. Every function returns a status code of sparse_numba_codes.h:
 * SN_OK (0), a negative code by name, or SINGULAR = 1 (Eigen's factorize
 * reported a numerical issue: a zero pivot). Nothing prints and no C++
 * exception leaves a function. A handle is an index into this module's
 * registry, (generation << 32) | (backend << 24) | slot with backend
 * SN_BACKEND_EIGEN_LU or SN_BACKEND_EIGEN_LDLT, never a pointer: a stale
 * handle returns SN_HANDLE_FREED, a SuperLU or UMFPACK handle
 * SN_HANDLE_WRONG_BACKEND, zero, a negative or a forged one
 * SN_HANDLE_INVALID. A handle is process-local.
 */

/**
 * Factorize an n x n CSC matrix with Eigen::SparseLU at an ordering of
 * sparse_numba_codes.h: NATURAL, COLAMD or AMD; MMD_AT_PLUS_A (and any
 * other) SN_ORDERING_UNSUPPORTED. The entries are held in Eigen's sorted
 * column storage (a repeated row index in a column summed) with the place
 * of each caller entry, for eigen_refactorize.
 */
int eigen_lu_factorize_ordered(double *values, int *rowind, int *colptr,
                               int n, int nnz, int ordering,
                               int64_t *handle_out);

/**
 * Factorize an n x n CSC matrix with Eigen::SimplicialLDLT (the lower
 * triangle read, AMDOrdering): ordering SN_ORDERING_AMD only (any other
 * SN_ORDERING_UNSUPPORTED); a pattern that is not symmetric SN_BAD_SHAPE.
 */
int eigen_ldlt_factorize_ordered(double *values, int *rowind, int *colptr,
                                 int n, int nnz, int ordering,
                                 int64_t *handle_out);

/**
 * Factorize the handle's pattern again on new values (nnz of them, in the
 * pattern's order): factorize() on the pattern analyzePattern() read once
 */
int eigen_refactorize(int64_t handle, double *values, int nnz);

/**
 * Solve into a caller-held solution: rhs and solution are (n, nrhs)
 * row-major (C-contiguous); n must equal the handle's dimension
 */
int eigen_solve_into(int64_t handle, double *rhs, double *solution, int n, int nrhs);

/**
 * The handle's facts: iout[SN_INFO_LEN] (n, nnz of L, nnz of U, backend id,
 * ordering id, Eigen's ComputationInfo of the last factorization), dout[
 * SN_DINFO_LEN] (the pivot tolerance)
 */
int eigen_factors_info(int64_t handle, int64_t *iout, double *dout);

/**
 * Free the factors of a handle; a second free returns SN_HANDLE_FREED
 */
int eigen_free_factors(int64_t handle);

/* "<Eigen version>,cc=...,optimize=...,fast_math=...,simd=...,fma=...,flags=...":
 * the Eigen headers' version and this shim's build facts, for cache_key() */
const char *eigen_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* EIGEN_WRAPPER_H */
