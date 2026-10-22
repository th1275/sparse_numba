/* sparse_numba_codes.h - the status codes, backend ids and orderings of the
 * sparse_numba shims, mirrored by
 * sparse_numba/errors.py (the codes) and sparse_numba/api.py (the enums).
 * The shims return these and never print. */
#ifndef SPARSE_NUMBA_CODES_H
#define SPARSE_NUMBA_CODES_H

/* status codes; a positive k is SINGULAR: the zero pivot's 1-based index */
#define SN_OK 0
#define SN_BAD_DTYPE (-1)
#define SN_BAD_SHAPE (-2)
#define SN_BAD_INDPTR (-3)
#define SN_BAD_INDEX (-4)
#define SN_NOT_SQUARE (-5)
#define SN_HANDLE_INVALID (-10)
#define SN_HANDLE_FREED (-11)
#define SN_HANDLE_WRONG_BACKEND (-12)
#define SN_BACKEND_ABSENT (-20)
#define SN_ORDERING_UNSUPPORTED (-21)
#define SN_ALLOC_FAILED (-30)
#define SN_BACKEND_ERROR (-31)

/* backend ids: the high byte of a handle's slot word */
#define SN_BACKEND_SUPERLU 1
#define SN_BACKEND_UMFPACK 2
#define SN_BACKEND_EIGEN_LU 3
#define SN_BACKEND_EIGEN_LDLT 4

/* orderings */
#define SN_ORDERING_NATURAL 0
#define SN_ORDERING_COLAMD 1
#define SN_ORDERING_AMD 2
#define SN_ORDERING_MMD_AT_PLUS_A 3

/* handle_info: the int64 fields (iout) and the double fields (dout) */
#define SN_INFO_N 0
#define SN_INFO_NNZ_L 1
#define SN_INFO_NNZ_U 2
#define SN_INFO_BACKEND 3
#define SN_INFO_ORDERING 4
#define SN_INFO_LAST_STATUS 5
#define SN_INFO_LEN 6
#define SN_DINFO_PIVOT_TOLERANCE 0
#define SN_DINFO_LEN 1

/* the handle: (generation << 32) | (backend << SN_SLOT_BITS) | slot */
#define SN_SLOT_BITS 24

#endif /* SPARSE_NUMBA_CODES_H */
