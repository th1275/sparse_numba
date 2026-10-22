/* superlu_wrapper.c - A minimal wrapper for SuperLU to be used with Numba */

// File name: superlu_wrapper.c
//
// Every function returns a status code of sparse_numba_codes.h and never
// prints (the debug messages compile in under SPARSE_NUMBA_DEBUG only).
// A handle is an index into this module's registry (sparse_numba_registry.h).
#include "superlu_wrapper.h"
#include "../sparse_numba_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  /* For DBL_MAX */
#include <slu_ddefs.h>  /* SuperLU header */

#ifdef SPARSE_NUMBA_DEBUG
#include <stdio.h>
#define DEBUG_PRINT(fmt, ...) printf("[SuperLU Debug] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#define SN_THIS_BACKEND SN_BACKEND_SUPERLU
#define SN_REGISTRY_FIRST SN_THIS_BACKEND
#define SN_REGISTRY_LAST SN_THIS_BACKEND
#include "../sparse_numba_registry.h"  /* the handle registry */


/* dgstrf / dgssv's info as a status code */
static int slu_code(int info, int n) {
    if (info == 0) return SN_OK;
    if (info < 0) {
        DEBUG_PRINT("SuperLU: Argument %d had an illegal value", -info);
        return SN_BACKEND_ERROR;
    }
    if (info <= n) {
        DEBUG_PRINT("SuperLU: U(%d,%d) is exactly zero", info, info);
        return info;  /* SINGULAR = k */
    }
    DEBUG_PRINT("SuperLU: Memory allocation failed: %d", info - n);
    return SN_ALLOC_FAILED;
}


int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution) {

    /* Input validation */
    if (!values || !rowind || !colptr || !rhs || !solution) {
        DEBUG_PRINT("Error: NULL pointer passed to solve_sparse_system");
        return SN_BAD_SHAPE;
    }

    if (nrows <= 0 || ncols <= 0 || nnz <= 0) {
        DEBUG_PRINT("Error: Invalid dimensions: rows=%d, cols=%d, nnz=%d", nrows, ncols, nnz);
        return SN_BAD_SHAPE;
    }

    if (colptr[0] != 0 || colptr[ncols] != nnz) {
        DEBUG_PRINT("Error: Invalid CSC format - colptr[0]=%d, colptr[%d]=%d, nnz=%d",
                    colptr[0], ncols, colptr[ncols], nnz);
        return SN_BAD_INDPTR;
    }

    /* Initialize all pointers to NULL for safe cleanup */
    SuperMatrix *A = NULL;
    SuperMatrix *B = NULL;
    SuperMatrix *L = NULL;
    SuperMatrix *U = NULL;
    int *perm_r = NULL;
    int *perm_c = NULL;
    superlu_options_t *options = NULL;
    SuperLUStat_t *stat = NULL;

    double *values_copy = NULL;
    int *rowind_copy = NULL;
    int *colptr_copy = NULL;
    double *rhs_copy = NULL;

    int status = SN_ALLOC_FAILED;
    int info = 0;

    /* Allocate everything */
    A = (SuperMatrix*)calloc(1, sizeof(SuperMatrix));  /* Store NULL */
    B = (SuperMatrix*)calloc(1, sizeof(SuperMatrix));
    L = (SuperMatrix*)calloc(1, sizeof(SuperMatrix));
    U = (SuperMatrix*)calloc(1, sizeof(SuperMatrix));
    options = (superlu_options_t*)malloc(sizeof(superlu_options_t));
    stat = (SuperLUStat_t*)malloc(sizeof(SuperLUStat_t));

    if (!A || !B || !L || !U || !options || !stat) {
        DEBUG_PRINT("Failed to allocate SuperLU structures");
        if (stat) free(stat);
        stat = NULL;
        goto cleanup;
    }

    perm_r = (int*)malloc(nrows * sizeof(int));
    perm_c = (int*)malloc(ncols * sizeof(int));

    if (!perm_r || !perm_c) {
        DEBUG_PRINT("Failed to allocate permutation arrays");
        free(stat);
        stat = NULL;
        goto cleanup;
    }

    /* Copy the arrays */
    values_copy = (double*)malloc(nnz * sizeof(double));
    rowind_copy = (int*)malloc(nnz * sizeof(int));
    colptr_copy = (int*)malloc((ncols+1) * sizeof(int));
    rhs_copy = (double*)malloc(nrows * sizeof(double));

    if (!values_copy || !rowind_copy || !colptr_copy || !rhs_copy) {
        DEBUG_PRINT("Failed to allocate data arrays");
        free(stat);
        stat = NULL;
        goto cleanup;
    }

    /* Copy the data */
    memcpy(values_copy, values, nnz * sizeof(double));
    memcpy(rowind_copy, rowind, nnz * sizeof(int));
    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
    memcpy(rhs_copy, rhs, nrows * sizeof(double));

    /* Initialize solution to zeros */
    for (int i = 0; i < nrows; i++) {
        solution[i] = 0.0;
    }

    /* Create A matrix */
    NCformat *Astore = (NCformat*)malloc(sizeof(NCformat));
    if (!Astore) {
        DEBUG_PRINT("Failed to allocate A.Store");
        free(stat);
        stat = NULL;
        goto cleanup;
    }

    A->Stype = SLU_NC;
    A->Dtype = SLU_D;
    A->Mtype = SLU_GE;
    A->nrow = nrows;
    A->ncol = ncols;
    A->Store = Astore;

    Astore->nnz = nnz;
    Astore->nzval = values_copy;
    Astore->rowind = rowind_copy;
    Astore->colptr = colptr_copy;

    /* Create B matrix */
    DNformat *Bstore = (DNformat*)malloc(sizeof(DNformat));
    if (!Bstore) {
        DEBUG_PRINT("Failed to allocate B.Store");
        free(stat);
        stat = NULL;
        goto cleanup;
    }

    B->Stype = SLU_DN;
    B->Dtype = SLU_D;
    B->Mtype = SLU_GE;
    B->nrow = nrows;
    B->ncol = 1;
    B->Store = Bstore;

    Bstore->lda = nrows;
    Bstore->nzval = rhs_copy;

    /* Set options */
    set_default_options(options);
    options->ColPerm = NATURAL;
    options->PrintStat = NO;

    /* Initialize stat */
    StatInit(stat);

    /* Initialize permutation arrays */
    for (int i = 0; i < nrows; i++) perm_r[i] = i;
    for (int i = 0; i < ncols; i++) perm_c[i] = i;

    /* Solve the system */
    dgssv(options, A, perm_c, perm_r, L, U, B, stat, &info);

    /* Check for successful completion */
    if (info != 0) {
        status = slu_code(info, nrows);
    } else {
        /* Copy solution from B to solution array */
        double *sol_ptr = (double*)Bstore->nzval;
        for (int i = 0; i < nrows; i++) {
            solution[i] = sol_ptr[i];
        }
        status = SN_OK;
    }

cleanup:
    /* Carefully free everything in reverse order of allocation */

    /* Free stat first (initialized only once dgssv's inputs were built) */
    if (stat) {
        StatFree(stat);
        free(stat);
    }

    /* Free L and U if they were allocated by dgssv */
    if (L && L->Store) {
        Destroy_SuperNode_Matrix(L);
    }
    if (L) free(L);

    if (U && U->Store) {
        Destroy_CompCol_Matrix(U);
    }
    if (U) free(U);

    /* Free B matrix - careful not to free rhs_copy until later */
    if (B && B->Store) {
        DNformat *Bs = (DNformat*)B->Store;
        Bs->nzval = NULL; /* Don't let SuperLU free our rhs_copy */
        free(B->Store);
    }
    if (B) free(B);

    /* Free A matrix - careful not to free our arrays until later */
    if (A && A->Store) {
        NCformat *As = (NCformat*)A->Store;
        As->nzval = NULL;   /* Don't let SuperLU free our arrays */
        As->rowind = NULL;
        As->colptr = NULL;
        free(A->Store);
    }
    if (A) free(A);

    if (options) free(options);
    if (perm_r) free(perm_r);
    if (perm_c) free(perm_c);

    /* NOW free our data arrays */
    if (values_copy) free(values_copy);
    if (rowind_copy) free(rowind_copy);
    if (colptr_copy) free(colptr_copy);
    if (rhs_copy) free(rhs_copy);

    return status;
}


/* ================================================================
 * Pre-factorization API: factorize once, solve many times
 * ================================================================ */

/* The factors of one handle. The CSC copies, A, the column-permuted AC
 * over them, perm_c and the elimination tree are held for refactorize;
 * L and U are dgstrf's. The dense right-hand side B over a held buffer of
 * n * bcols doubles (n at factorize, grown on the first wider solve) and
 * the SuperLUStat_t are made once at factorize and freed at free
 * (no allocation per solve); busy guards them, so a second solve running on
 * the same handle at the same time takes the allocating path instead. */
typedef struct {
    SuperMatrix A;
    SuperMatrix AC;
    SuperMatrix L;
    SuperMatrix U;
    int *perm_r;
    int *perm_c;
    int *etree;
    double *values;
    int *rowind;
    int *colptr;
    superlu_options_t options;
    SuperMatrix B;
    DNformat Bstore;
    double *bbuf;
    int bcols;        /* the columns bbuf holds */
    SuperLUStat_t stat;
    int have_stat;
    char busy;
    int n;
    int nnz;
    int ordering;
    int have_ac;
    int have_lu;
    int last_status;  /* dgstrf's info of the last factorization */
    int state;        /* SN_OK while L and U are usable for a solve */
} superlu_factors_t;


static void slu_destroy(superlu_factors_t *f) {
    if (!f) return;
    if (f->have_lu) {
        Destroy_SuperNode_Matrix(&f->L);
        Destroy_CompCol_Matrix(&f->U);
    }
    if (f->have_ac) Destroy_CompCol_Permuted(&f->AC);
    if (f->have_stat) StatFree(&f->stat);
    if (f->bbuf) free(f->bbuf);
    if (f->A.Store) free(f->A.Store);  /* the arrays are ours, freed below */
    if (f->perm_r) free(f->perm_r);
    if (f->perm_c) free(f->perm_c);
    if (f->etree) free(f->etree);
    if (f->values) free(f->values);
    if (f->rowind) free(f->rowind);
    if (f->colptr) free(f->colptr);
    free(f);
}


/* dgstrf on the held AC and the held stat: the factorization of factorize
 * and refactorize */
static int slu_numeric(superlu_factors_t *f) {
    GlobalLU_t Glu;
    int info = 0;
    int panel_size = sp_ienv(1);
    int relax = sp_ienv(2);

    memset(&Glu, 0, sizeof(GlobalLU_t));
    f->L.Store = NULL;
    f->U.Store = NULL;
    dgstrf(&f->options, &f->AC, relax, panel_size, f->etree,
           NULL, 0,  /* work=NULL, lwork=0 => SuperLU allocates internally */
           f->perm_c, f->perm_r, &f->L, &f->U, &Glu, &f->stat, &info);

    /* a zero pivot completes the factorization; an allocation failure does not */
    f->have_lu = (info == 0 || (info > 0 && info <= f->n)) && f->L.Store && f->U.Store;
    f->last_status = info;
    f->state = slu_code(info, f->n);
    return f->state;
}


int factorize_sparse_system_ordered(double *values, int *rowind, int *colptr,
                                    int n, int nnz, int ordering,
                                    int64_t *handle_out) {
    superlu_factors_t *f;
    colperm_t colperm;
    int code;

    if (!handle_out) return SN_BAD_SHAPE;
    *handle_out = 0;
    if (!values || !rowind || !colptr) {
        DEBUG_PRINT("Error: NULL pointer passed to factorize_sparse_system_ordered");
        return SN_BAD_SHAPE;
    }
    if (n <= 0 || nnz <= 0) {
        DEBUG_PRINT("Error: Invalid dimensions: n=%d, nnz=%d", n, nnz);
        return SN_BAD_SHAPE;
    }
    if (colptr[0] != 0 || colptr[n] != nnz) {
        DEBUG_PRINT("Error: Invalid CSC format - colptr[0]=%d, colptr[%d]=%d, nnz=%d",
                    colptr[0], n, colptr[n], nnz);
        return SN_BAD_INDPTR;
    }

    /* options.ColPerm from the ordering (NATURAL the 0.1.11
     * factorization); AMD is refused (SuperLU has no AMD column
     * ordering of its own) */
    switch (ordering) {
    case SN_ORDERING_NATURAL: colperm = NATURAL; break;
    case SN_ORDERING_COLAMD: colperm = COLAMD; break;
    case SN_ORDERING_MMD_AT_PLUS_A: colperm = MMD_AT_PLUS_A; break;
    default: return SN_ORDERING_UNSUPPORTED;
    }

    f = (superlu_factors_t*)calloc(1, sizeof(superlu_factors_t));
    if (!f) return SN_ALLOC_FAILED;
    f->n = n;
    f->nnz = nnz;
    f->ordering = ordering;

    f->perm_r = (int*)malloc(n * sizeof(int));
    f->perm_c = (int*)malloc(n * sizeof(int));
    f->etree = (int*)malloc(n * sizeof(int));
    f->values = (double*)malloc(nnz * sizeof(double));
    f->rowind = (int*)malloc(nnz * sizeof(int));
    f->colptr = (int*)malloc((n + 1) * sizeof(int));
    f->A.Store = malloc(sizeof(NCformat));
    f->bbuf = (double*)malloc(n * sizeof(double));
    f->bcols = 1;
    if (!f->perm_r || !f->perm_c || !f->etree || !f->values || !f->rowind ||
            !f->colptr || !f->A.Store || !f->bbuf) {
        DEBUG_PRINT("Failed to allocate the factors");
        slu_destroy(f);
        return SN_ALLOC_FAILED;
    }

    memcpy(f->values, values, nnz * sizeof(double));
    memcpy(f->rowind, rowind, nnz * sizeof(int));
    memcpy(f->colptr, colptr, (n + 1) * sizeof(int));

    /* Build SuperMatrix A in CSC (NC) format over the held copies */
    {
        NCformat *Astore = (NCformat*)f->A.Store;
        f->A.Stype = SLU_NC;
        f->A.Dtype = SLU_D;
        f->A.Mtype = SLU_GE;
        f->A.nrow = n;
        f->A.ncol = n;
        Astore->nnz = nnz;
        Astore->nzval = f->values;
        Astore->rowind = f->rowind;
        Astore->colptr = f->colptr;
    }

    /* The held dense B over bbuf (lda n) */
    f->B.Stype = SLU_DN;
    f->B.Dtype = SLU_D;
    f->B.Mtype = SLU_GE;
    f->B.nrow = n;
    f->B.ncol = 1;
    f->B.Store = &f->Bstore;
    f->Bstore.lda = n;
    f->Bstore.nzval = f->bbuf;

    /* Set options */
    set_default_options(&f->options);
    f->options.ColPerm = colperm;
    f->options.PrintStat = NO;

    /* The stat, once (freed at free) */
    StatInit(&f->stat);
    f->have_stat = 1;

    /* Initialize permutation arrays */
    for (int i = 0; i < n; i++) f->perm_r[i] = i;
    for (int i = 0; i < n; i++) f->perm_c[i] = i;

    /* Step 1: Column permutation */
    get_perm_c(f->options.ColPerm, &f->A, f->perm_c);

    /* Step 2: Pre-order the matrix (AC's values point at the held copy) */
    sp_preorder(&f->options, &f->A, f->perm_c, f->etree, &f->AC);
    f->have_ac = 1;

    /* Step 3: LU factorization */
    code = slu_numeric(f);
    if (code != SN_OK) {
        slu_destroy(f);
        return code;
    }

    code = sn_register(f, SN_THIS_BACKEND, handle_out);
    if (code != SN_OK) {
        slu_destroy(f);
        return code;
    }
    return SN_OK;
}


int factorize_sparse_system(double *values, int *rowind, int *colptr,
                            int nrows, int ncols, int nnz,
                            int64_t *handle_out) {
    if (!handle_out) return SN_BAD_SHAPE;
    *handle_out = 0;
    if (nrows != ncols) return SN_NOT_SQUARE;
    return factorize_sparse_system_ordered(values, rowind, colptr, ncols, nnz,
                                           SN_ORDERING_NATURAL, handle_out);
}


int refactorize_sparse_system(int64_t handle, double *values, int nnz) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    superlu_factors_t *f = (superlu_factors_t*)p;
    if (code != SN_OK) return code;
    if (!values || nnz != f->nnz) return SN_BAD_SHAPE;

    memcpy(f->values, values, nnz * sizeof(double));
    if (f->have_lu) {
        Destroy_SuperNode_Matrix(&f->L);
        Destroy_CompCol_Matrix(&f->U);
        f->have_lu = 0;
    }
    /* the held perm_c and etree; dgstrf again on the new values */
    f->options.Fact = SamePattern;
    return slu_numeric(f);
}


/* rhs -> work (column-major, lda n) and work -> solution; rhs / solution
 * column-major unless row_major (then (n, nrhs) C-contiguous) */
static void slu_copy_in(double *work, const double *rhs, int n, int nrhs, int row_major) {
    size_t count = (size_t)n * (size_t)nrhs;
    if (row_major && nrhs > 1) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nrhs; j++)
                work[(size_t)j * n + i] = rhs[(size_t)i * nrhs + j];
    } else {
        memcpy(work, rhs, count * sizeof(double));
    }
}

static void slu_copy_out(double *solution, const double *work, int n, int nrhs, int row_major) {
    size_t count = (size_t)n * (size_t)nrhs;
    if (row_major && nrhs > 1) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nrhs; j++)
                solution[(size_t)i * nrhs + j] = work[(size_t)j * n + i];
    } else {
        memcpy(solution, work, count * sizeof(double));
    }
}


/* dgstrs on the held factors: memcpy into the held B, dgstrs with the held
 * stat, memcpy out. When another solve holds the handle's B, this
 * one runs over a B and a stat of its own (the 0.1.11 path). */
static int slu_solve(superlu_factors_t *f, const double *rhs, double *solution,
                     int nrhs, int row_major) {
    int info = 0;
    int n = f->n;

    if (f->state != SN_OK) return f->state;

    if (!__atomic_test_and_set(&f->busy, __ATOMIC_ACQUIRE)) {
        if (nrhs > f->bcols) {
            /* grown once to the widest solve seen; never shrunk */
            double *wider = (double*)realloc(f->bbuf, (size_t)n * (size_t)nrhs * sizeof(double));
            if (!wider) {
                __atomic_clear(&f->busy, __ATOMIC_RELEASE);
                return SN_ALLOC_FAILED;
            }
            f->bbuf = wider;
            f->bcols = nrhs;
            f->Bstore.nzval = wider;
        }
        f->B.ncol = nrhs;
        slu_copy_in(f->bbuf, rhs, n, nrhs, row_major);
        dgstrs(NOTRANS, &f->L, &f->U, f->perm_c, f->perm_r, &f->B, &f->stat, &info);
        if (info == 0) slu_copy_out(solution, f->bbuf, n, nrhs, row_major);
        __atomic_clear(&f->busy, __ATOMIC_RELEASE);
    } else {
        SuperMatrix B;
        DNformat Bstore;
        SuperLUStat_t stat;
        double *work = (double*)malloc((size_t)n * (size_t)nrhs * sizeof(double));
        if (!work) {
            DEBUG_PRINT("Failed to allocate rhs_copy");
            return SN_ALLOC_FAILED;
        }
        slu_copy_in(work, rhs, n, nrhs, row_major);
        B.Stype = SLU_DN;
        B.Dtype = SLU_D;
        B.Mtype = SLU_GE;
        B.nrow = n;
        B.ncol = nrhs;
        B.Store = &Bstore;
        Bstore.lda = n;
        Bstore.nzval = work;
        StatInit(&stat);
        dgstrs(NOTRANS, &f->L, &f->U, f->perm_c, f->perm_r, &B, &stat, &info);
        StatFree(&stat);
        if (info == 0) slu_copy_out(solution, work, n, nrhs, row_major);
        free(work);
    }

    if (info != 0) {
        DEBUG_PRINT("SuperLU dgstrs failed with info = %d", info);
        return SN_BACKEND_ERROR;
    }
    return SN_OK;
}


int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    superlu_factors_t *f = (superlu_factors_t*)p;
    if (code != SN_OK) return code;
    if (!rhs || !solution || nrhs <= 0) {
        DEBUG_PRINT("Error: NULL pointer passed to solve_with_factors");
        return SN_BAD_SHAPE;
    }
    return slu_solve(f, rhs, solution, nrhs, 0);
}


int solve_into_with_factors(int64_t handle, double *rhs, double *solution,
                            int n, int nrhs) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    superlu_factors_t *f = (superlu_factors_t*)p;
    if (code != SN_OK) return code;
    if (!rhs || !solution || nrhs <= 0 || n != f->n) return SN_BAD_SHAPE;
    return slu_solve(f, rhs, solution, nrhs, 1);
}


int sparse_factors_info(int64_t handle, int64_t *iout, double *dout) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    superlu_factors_t *f = (superlu_factors_t*)p;
    if (code != SN_OK) return code;
    if (!iout || !dout) return SN_BAD_SHAPE;
    iout[SN_INFO_N] = f->n;
    iout[SN_INFO_NNZ_L] = f->have_lu ? ((SCformat*)f->L.Store)->nnz : 0;
    iout[SN_INFO_NNZ_U] = f->have_lu ? ((NCformat*)f->U.Store)->nnz : 0;
    iout[SN_INFO_BACKEND] = SN_THIS_BACKEND;
    iout[SN_INFO_ORDERING] = f->ordering;
    iout[SN_INFO_LAST_STATUS] = f->last_status;
    dout[SN_DINFO_PIVOT_TOLERANCE] = f->options.DiagPivotThresh;
    return SN_OK;
}


int free_sparse_factors(int64_t handle) {
    void *p = NULL;
    int code = sn_release(handle, &p);
    if (code != SN_OK) return code;
    slu_destroy((superlu_factors_t*)p);
    return SN_OK;
}


/* The SuperLU version (the headers' SUPERLU_*_VERSION, slu_util.h: the
 * library is called through those headers) and this shim's build facts
 * (sparse_numba_registry.h: the compiler, its code-generation flags, and
 * setup.py's compile flags as SN_BUILD_FLAGS), for cache_key() */
const char *superlu_build_info(void) {
    return SN_STR(SUPERLU_MAJOR_VERSION) "." SN_STR(SUPERLU_MINOR_VERSION) "."
           SN_STR(SUPERLU_PATCH_VERSION) "," SN_BUILD_FACTS;
}
