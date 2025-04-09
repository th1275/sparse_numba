/* superlu_wrapper.c - A minimal wrapper for SuperLU to be used with Numba */

// File name: superlu_wrapper.c
//
#include "superlu_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  /* For DBL_MAX */
#include <slu_ddefs.h>  /* SuperLU header */

#define DEBUG_PRINT(fmt, ...) printf("[SuperLU Debug] " fmt "\n", ##__VA_ARGS__)

int solve_sparse_system(double *values, int *rowind, int *colptr,
                        int nrows, int ncols, int nnz,
                        double *rhs, double *solution) {
//    DEBUG_PRINT("Starting solve_sparse_system with matrix %dx%d, NNZ: %d", nrows, ncols, nnz);

    /* Input validation */
    if (!values || !rowind || !colptr || !rhs || !solution) {
        DEBUG_PRINT("Error: NULL pointer passed to solve_sparse_system");
        return -1;
    }

    if (nrows <= 0 || ncols <= 0 || nnz <= 0) {
        DEBUG_PRINT("Error: Invalid dimensions: rows=%d, cols=%d, nnz=%d", nrows, ncols, nnz);
        return -2;
    }

    if (colptr[0] != 0 || colptr[ncols] != nnz) {
        DEBUG_PRINT("Error: Invalid CSC format - colptr[0]=%d, colptr[%d]=%d, nnz=%d",
                    colptr[0], ncols, colptr[ncols], nnz);
        return -3;
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

    int status = -99;
    int info = 0;

    /* Allocate everything */
    A = (SuperMatrix*)malloc(sizeof(SuperMatrix));
    B = (SuperMatrix*)malloc(sizeof(SuperMatrix));
    L = (SuperMatrix*)malloc(sizeof(SuperMatrix));
    U = (SuperMatrix*)malloc(sizeof(SuperMatrix));
    options = (superlu_options_t*)malloc(sizeof(superlu_options_t));
    stat = (SuperLUStat_t*)malloc(sizeof(SuperLUStat_t));

    if (!A || !B || !L || !U || !options || !stat) {
        DEBUG_PRINT("Failed to allocate SuperLU structures");
        status = -10;
        goto cleanup;
    }

    /* Initialize SuperMatrix Store pointers to NULL */
    A->Store = NULL;
    B->Store = NULL;
    L->Store = NULL;  /* This fixes the warning */
    U->Store = NULL;  /* Also initialize U->Store */

    perm_r = (int*)malloc(nrows * sizeof(int));
    perm_c = (int*)malloc(ncols * sizeof(int));

    if (!perm_r || !perm_c) {
        DEBUG_PRINT("Failed to allocate permutation arrays");
        status = -11;
        goto cleanup;
    }

    /* Copy the arrays */
    values_copy = (double*)malloc(nnz * sizeof(double));
    rowind_copy = (int*)malloc(nnz * sizeof(int));
    colptr_copy = (int*)malloc((ncols+1) * sizeof(int));
    rhs_copy = (double*)malloc(nrows * sizeof(double));

    if (!values_copy || !rowind_copy || !colptr_copy || !rhs_copy) {
        DEBUG_PRINT("Failed to allocate data arrays");
        status = -12;
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

//    DEBUG_PRINT("Memory allocated and data copied");

    /* Initialize SuperLU structures manually */

    /* Create A matrix */
    NCformat *Astore = (NCformat*)malloc(sizeof(NCformat));
    if (!Astore) {
        DEBUG_PRINT("Failed to allocate A.Store");
        status = -20;
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

//    DEBUG_PRINT("A matrix initialized manually");

    /* Create B matrix */
    DNformat *Bstore = (DNformat*)malloc(sizeof(DNformat));
    if (!Bstore) {
        DEBUG_PRINT("Failed to allocate B.Store");
        status = -21;
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

//    DEBUG_PRINT("B matrix initialized manually");

    /* Set options */
    set_default_options(options);
    options->ColPerm = NATURAL;
    options->PrintStat = NO;

    /* Initialize stat */
    StatInit(stat);

    /* Initialize permutation arrays */
    for (int i = 0; i < nrows; i++) perm_r[i] = i;
    for (int i = 0; i < ncols; i++) perm_c[i] = i;

//    DEBUG_PRINT("Calling dgssv");

    /* Solve the system */
    dgssv(options, A, perm_c, perm_r, L, U, B, stat, &info);

//    DEBUG_PRINT("dgssv returned with info = %d", info);

    /* Check for successful completion */
    if (info != 0) {
        if (info < 0) {
            DEBUG_PRINT("SuperLU: Argument %d had an illegal value", -info);
        } else if (info <= nrows) {
            DEBUG_PRINT("SuperLU: U(%d,%d) is exactly zero", info, info);
        } else {
            DEBUG_PRINT("SuperLU: Memory allocation failed: %d", info - nrows);
        }
        status = info;
    } else {
        /* Copy solution from B to solution array */
        double *sol_ptr = (double*)Bstore->nzval;
//        DEBUG_PRINT("Copying solution from B to output array");
        for (int i = 0; i < nrows; i++) {
            solution[i] = sol_ptr[i];
        }
//        DEBUG_PRINT("Solution copied successfully");
        status = 0;
    }

cleanup:
//    DEBUG_PRINT("Cleaning up resources");

    /* Carefully free everything in reverse order of allocation */

    /* Free stat first */
    if (stat) {
        StatFree(stat);
        free(stat);
//        DEBUG_PRINT("Freed stat");
    }

    /* CRITICAL: Free L and U if they were allocated by dgssv */
    if (L && L->Store) {
//        DEBUG_PRINT("Freeing L matrix");
        Destroy_SuperNode_Matrix(L);
    }
    if (L) free(L);

    if (U && U->Store) {
//        DEBUG_PRINT("Freeing U matrix");
        Destroy_CompCol_Matrix(U);
    }
    if (U) free(U);

    /* Free B matrix - careful not to free rhs_copy until later */
    if (B && B->Store) {
        DNformat *Bstore = (DNformat*)B->Store;
        Bstore->nzval = NULL; /* Don't let SuperLU free our rhs_copy */
        free(B->Store);
//        DEBUG_PRINT("Freed B.Store");
    }
    if (B) {
        free(B);
//        DEBUG_PRINT("Freed B");
    }

    /* Free A matrix - careful not to free our arrays until later */
    if (A && A->Store) {
        NCformat *Astore = (NCformat*)A->Store;
        Astore->nzval = NULL;   /* Don't let SuperLU free our arrays */
        Astore->rowind = NULL;
        Astore->colptr = NULL;
        free(A->Store);
//        DEBUG_PRINT("Freed A.Store");
    }
    if (A) {
        free(A);
//        DEBUG_PRINT("Freed A");
    }

    /* Free options */
    if (options) {
        free(options);
//        DEBUG_PRINT("Freed options");
    }

    /* Free permutation arrays */
    if (perm_r) {
        free(perm_r);
//        DEBUG_PRINT("Freed perm_r");
    }
    if (perm_c) {
        free(perm_c);
//        DEBUG_PRINT("Freed perm_c");
    }

    /* NOW free our data arrays */
    if (values_copy) {
        free(values_copy);
//        DEBUG_PRINT("Freed values_copy");
    }
    if (rowind_copy) {
        free(rowind_copy);
//        DEBUG_PRINT("Freed rowind_copy");
    }
    if (colptr_copy) {
        free(colptr_copy);
//        DEBUG_PRINT("Freed colptr_copy");
    }
    if (rhs_copy) {
        free(rhs_copy);
//        DEBUG_PRINT("Freed rhs_copy");
    }

//    DEBUG_PRINT("Cleanup complete, returning status %d", status);
    return status;
}