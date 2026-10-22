/* umfpack_wrapper.h - A minimal wrapper for UMFPACK to be used with Numba */
#ifndef UMFPACK_WRAPPER_H
#define UMFPACK_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * UMFPACK is interface-only: the user's library is loaded at the first call
 * that needs it (SPARSE_NUMBA_UMFPACK_LIBRARY, else the platform's names);
 * not found, the calls return SN_BACKEND_ABSENT.
 *
 * Every function returns a status code of sparse_numba_codes.h: SN_OK (0),
 * a negative code by name, or SINGULAR = k > 0 (the 1-based position of
 * U's first zero diagonal entry). A handle is an index into this module's
 * registry, (generation << 32) | (SN_BACKEND_UMFPACK << 24) | slot, never a
 * pointer; it is process-local.
 */

/* Main solver function - taking CSC format matrix and solving Ax=b */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution);

/* Pre-factorize a sparse matrix (symbolic + numeric) at AMD, returns the handle */
int factorize_sparse_system(double *values, int *rowind, int *colptr,
                            int nrows, int ncols, int nnz,
                            int64_t *handle_out);

/* Solve using pre-computed factors; rhs / solution nrhs columns of n (column-major) */
int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs);

/* Free the factors of a handle; a second free returns SN_HANDLE_FREED */
int free_sparse_factors(int64_t handle);

/* Factorize an n x n CSC matrix at an ordering of sparse_numba_codes.h:
 * SN_ORDERING_AMD (STRATEGY_SYMMETRIC), SN_ORDERING_COLAMD
 * (STRATEGY_UNSYMMETRIC), SN_ORDERING_NATURAL (UMFPACK_ORDERING_NONE, where
 * the library has umfpack_version); SN_ORDERING_MMD_AT_PLUS_A (and any
 * other) SN_ORDERING_UNSUPPORTED. The Control is held per handle. */
int factorize_sparse_system_ordered(double *values, int *rowind, int *colptr,
                                    int n, int nnz, int ordering,
                                    int64_t *handle_out);

/* umfpack_di_numeric again with the held Symbolic on new values (nnz of
 * them, in the pattern's order) */
int refactorize_sparse_system(int64_t handle, double *values, int nnz);

/* Solve into a caller-held solution: rhs and solution (n, nrhs) row-major
 * (C-contiguous); n must equal the handle's dimension */
int solve_into_with_factors(int64_t handle, double *rhs, double *solution,
                            int n, int nrhs);

/* The handle's facts: iout[SN_INFO_LEN] (n, nnz of L, nnz of U, backend id,
 * ordering id, UMFPACK's status of the last factorization), dout[
 * SN_DINFO_LEN] (the pivot tolerance) */
int sparse_factors_info(int64_t handle, int64_t *iout, double *dout);

/* Load the library if not yet tried; SN_OK or SN_BACKEND_ABSENT. version:
 * "x.y.z" (umfpack_version), "unknown" or "absent"; source: the path or
 * the name it loaded from */
int umfpack_library_info(char *version, int version_len, char *source, int source_len);

#ifdef __cplusplus
}
#endif

#endif /* UMFPACK_WRAPPER_H */
