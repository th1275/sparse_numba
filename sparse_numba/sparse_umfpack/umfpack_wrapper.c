/* umfpack_wrapper.c - A minimal wrapper for UMFPACK to be used with Numba */

// File name: umfpack_wrapper.c
//
#include "umfpack_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  /* For DBL_MAX */
#include <umfpack.h>

int solve_sparse_system(double *values, int *rowind, int *colptr,
                      int nrows, int ncols, int nnz,
                      double *rhs, double *solution) {
    /* Initialize variables */
    void *Symbolic = NULL;
    void *Numeric = NULL;
    double *rhs_copy = NULL;
    double *values_copy = NULL;
    int *rowind_copy = NULL;
    int *colptr_copy = NULL;
    double Control[UMFPACK_CONTROL];
    double Info[UMFPACK_INFO];
    int status = -1;  // Default to error

    /* Get default control parameters */
    umfpack_di_defaults(Control);

    /* Set more robust parameters for ill-conditioned matrices */
    Control[UMFPACK_PIVOT_TOLERANCE] = 1.0;  // Max pivot tolerance
    Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;  // More stable strategy

    /* Print size information */
//    printf("Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
//           nrows, ncols, nnz);

    /* Allocate memory with error checking */
    values_copy = (double *) malloc(nnz * sizeof(double));
    if (!values_copy) {
        printf("Failed to allocate memory for values_copy\n");
        goto cleanup;
    }

    rowind_copy = (int *) malloc(nnz * sizeof(int));
    if (!rowind_copy) {
        printf("Failed to allocate memory for rowind_copy\n");
        goto cleanup;
    }

    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
    if (!colptr_copy) {
        printf("Failed to allocate memory for colptr_copy\n");
        goto cleanup;
    }

    rhs_copy = (double *) malloc(nrows * sizeof(double));
    if (!rhs_copy) {
        printf("Failed to allocate memory for rhs_copy\n");
        goto cleanup;
    }

    /* Copy data */
    memcpy(values_copy, values, nnz * sizeof(double));
    memcpy(rowind_copy, rowind, nnz * sizeof(int));
    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
    memcpy(rhs_copy, rhs, nrows * sizeof(double));

    /* Validate matrix format */
    if (colptr_copy[0] != 0) {
        printf("Error: First column pointer must be 0, got %d\n", colptr_copy[0]);
        goto cleanup;
    }

    if (colptr_copy[ncols] != nnz) {
        printf("Error: Last column pointer must equal nnz, got %d vs %d\n",
               colptr_copy[ncols], nnz);
        goto cleanup;
    }

    /* Symbolic analysis phase */
//    printf("Performing symbolic analysis...\n");
    status = umfpack_di_symbolic(nrows, ncols, colptr_copy, rowind_copy, values_copy,
                              &Symbolic, Control, Info);

    if (status != UMFPACK_OK) {
        printf("Symbolic analysis failed with status %d\n", status);
        goto cleanup;
    }

    /* Numeric factorization phase */
//    printf("Performing numeric factorization...\n");
    status = umfpack_di_numeric(colptr_copy, rowind_copy, values_copy,
                              Symbolic, &Numeric, Control, Info);

    if (status != UMFPACK_OK) {
        printf("Numeric factorization failed with status %d\n", status);
        goto cleanup;
    }

    /* Solve the system */
//    printf("Solving the system...\n");
    status = umfpack_di_solve(UMFPACK_A, colptr_copy, rowind_copy, values_copy,
                           solution, rhs_copy, Numeric, Control, Info);

    if (status != UMFPACK_OK) {
        printf("Solve failed with status %d\n", status);
        goto cleanup;
    }

cleanup:
    /* Free resources */
    if (Numeric) umfpack_di_free_numeric(&Numeric);
    if (Symbolic) umfpack_di_free_symbolic(&Symbolic);
    if (values_copy) free(values_copy);
    if (rowind_copy) free(rowind_copy);
    if (colptr_copy) free(colptr_copy);
    if (rhs_copy) free(rhs_copy);

    return status;
}
