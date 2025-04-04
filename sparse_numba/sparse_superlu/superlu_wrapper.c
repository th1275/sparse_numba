/* superlu_wrapper.c - A minimal wrapper for SuperLU to be used with Numba */
#include "superlu_wrapper.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slu_ddefs.h"
#include <float.h>
#include <math.h>

///* Size-adaptive SuperLU wrapper - uses different strategies based on matrix size */
//int solve_sparse_system(double *values, int *rowind, int *colptr,
//                      int nrows, int ncols, int nnz,
//                      double *rhs, double *solution) {
//    /* Print size information */
//    printf("SuperLU Wrapper Debug Info:\n");
//    printf("  Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
//           nrows, ncols, nnz);
//
//    /* Initialize variables */
//    SuperMatrix A, L, U, B;
//    int *perm_r = NULL;
//    int *perm_c = NULL;
//    double *rhs_copy = NULL;
//    double *values_copy = NULL;
//    int *rowind_copy = NULL;
//    int *colptr_copy = NULL;
//    int info = 0;
//    superlu_options_t options;
//    SuperLUStat_t stat;
//
//    /* Set default options */
//    set_default_options(&options);
//
//    /* Always use conservative settings regardless of matrix size */
//    printf("Using consistent conservative strategy for all matrices\n");
//    options.ColPerm = MMD_AT_PLUS_A;
//    options.DiagPivotThresh = 0.001;
//    options.SymmetricMode = NO;
//    options.PivotGrowth = NO;
//    options.ConditionNumber = NO;
//    options.IterRefine = NOREFINE;
//
//    /* Allocate permutation vectors */
//    perm_r = (int *) malloc(nrows * sizeof(int));
//    perm_c = (int *) malloc(ncols * sizeof(int));
//    if (!perm_r || !perm_c) {
//        printf("ERROR: Failed to allocate permutation vectors\n");
//        if (perm_r) free(perm_r);
//        if (perm_c) free(perm_c);
//        return -1;
//    }
//
//    /* Create copies of input arrays */
//    values_copy = (double *) malloc(nnz * sizeof(double));
//    rowind_copy = (int *) malloc(nnz * sizeof(int));
//    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
//
//    if (!values_copy || !rowind_copy || !colptr_copy) {
//        printf("ERROR: Failed to allocate matrix copy buffers\n");
//        if (values_copy) free(values_copy);
//        if (rowind_copy) free(rowind_copy);
//        if (colptr_copy) free(colptr_copy);
//        free(perm_r);
//        free(perm_c);
//        return -1;
//    }
//
//    /* Copy data */
//    memcpy(values_copy, values, nnz * sizeof(double));
//    memcpy(rowind_copy, rowind, nnz * sizeof(int));
//    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
//
//    /* Create compressed column matrix for SuperLU */
//    dCreate_CompCol_Matrix(&A, nrows, ncols, nnz, values_copy, rowind_copy, colptr_copy,
//                          SLU_NC, SLU_D, SLU_GE);
//
//    /* Create a copy of right-hand side */
//    rhs_copy = (double *) malloc(nrows * sizeof(double));
//    if (!rhs_copy) {
//        printf("ERROR: Failed to allocate RHS copy\n");
//        free(perm_r);
//        free(perm_c);
//        Destroy_CompCol_Matrix(&A);
//        return -1;
//    }
//    memcpy(rhs_copy, rhs, nrows * sizeof(double));
//
//    /* Create right-hand side matrix B */
//    dCreate_Dense_Matrix(&B, nrows, 1, rhs_copy, nrows, SLU_DN, SLU_D, SLU_GE);
//
//    /* Initialize statistics */
//    StatInit(&stat);
//
//    /* Try the simple dgssv approach first */
//    printf("Using dgssv for solving the system\n");
//
//    /* This is the simplest way to solve the system in SuperLU */
//    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
//
//    /* Copy solution if successful */
//    if (info == 0) {
//        printf("Solver successful, copying solution\n");
//        DNformat *Bstore = (DNformat *) B.Store;
//        double *Bval = (double *) Bstore->nzval;
//        memcpy(solution, Bval, nrows * sizeof(double));
//    } else {
//        printf("Solver failed with info = %d\n", info);
//    }
//
//    /* Free SuperLU structures */
//    printf("Cleaning up SuperLU structures\n");
//    Destroy_CompCol_Matrix(&A);
//    Destroy_SuperMatrix_Store(&B);
//
//    /* Free L and U if they were allocated */
//    if (L.Store) Destroy_SuperNode_Matrix(&L);
//    if (U.Store) Destroy_CompCol_Matrix(&U);
//
//    StatFree(&stat);
//
//    /* Free allocated memory */
//    free(perm_r);
//    free(perm_c);
//
//    return info;
//}

/* Row and column scaling factors */
typedef struct {
    double *r;    /* Row scaling factors */
    double *c;    /* Column scaling factors */
} scaling_factors_t;


/* Safe memory allocation with error checking */
void* safe_malloc(size_t size, const char* what) {
    void* ptr = malloc(size);
    if (!ptr) {
        printf("ERROR: Failed to allocate memory for %s (%zu bytes)\n", what, size);
    }
    return ptr;
}

/* Compute row and column scaling to equilibrate the matrix - safe version */
scaling_factors_t compute_scaling(double *values, int *rowind, int *colptr,
                                      int nrows, int ncols, int nnz) {
    scaling_factors_t factors = {NULL, NULL};
    int i, j, k;
    double *row_max = NULL;
    double *col_max = NULL;

    /* Validate inputs */
    if (!values || !rowind || !colptr || nrows <= 0 || ncols <= 0 || nnz <= 0) {
        printf("ERROR: Invalid inputs to compute_scaling_safe\n");
        return factors;
    }

    /* Allocate scaling vectors with error checking */
    factors.r = (double*) safe_malloc(nrows * sizeof(double), "row scaling");
    factors.c = (double*) safe_malloc(ncols * sizeof(double), "column scaling");
    row_max = (double*) safe_malloc(nrows * sizeof(double), "row maxima");
    col_max = (double*) safe_malloc(ncols * sizeof(double), "column maxima");

    if (!factors.r || !factors.c || !row_max || !col_max) {
        /* Error already printed by safe_malloc */
        if (factors.r) free(factors.r);
        if (factors.c) free(factors.c);
        if (row_max) free(row_max);
        if (col_max) free(col_max);
        factors.r = factors.c = NULL;
        return factors;
    }

    /* Initialize arrays to safe values */
    for (i = 0; i < nrows; i++) {
        row_max[i] = 0.0;
        /* Initialize scaling to identity in case we need to exit early */
        factors.r[i] = 1.0;
    }

    for (j = 0; j < ncols; j++) {
        col_max[j] = 0.0;
        /* Initialize scaling to identity in case we need to exit early */
        factors.c[j] = 1.0;
    }

    /* Find maximum absolute value in each row and column - with bounds checking */
    for (j = 0; j < ncols; j++) {
        /* Validate column pointers */
        if (colptr[j] < 0 || colptr[j] > nnz) {
            printf("ERROR: Invalid column pointer at column %d: %d\n", j, colptr[j]);
            goto cleanup;
        }

        if (j < ncols-1 && (colptr[j+1] < colptr[j] || colptr[j+1] > nnz)) {
            printf("ERROR: Invalid next column pointer at column %d: %d\n", j, colptr[j+1]);
            goto cleanup;
        }

        int col_end = (j < ncols-1) ? colptr[j+1] : nnz;

        for (k = colptr[j]; k < col_end; k++) {
            /* Validate row index */
            if (k < 0 || k >= nnz) {
                printf("ERROR: k index out of bounds: %d (nnz=%d)\n", k, nnz);
                goto cleanup;
            }

            i = rowind[k];

            /* Validate row index */
            if (i < 0 || i >= nrows) {
                printf("ERROR: Invalid row index at position %d: %d\n", k, i);
                goto cleanup;
            }

            double abs_val = fabs(values[k]);

            /* Update row and column maximums - with safety bounds check */
            if (i < nrows && j < ncols) {
                if (abs_val > row_max[i]) row_max[i] = abs_val;
                if (abs_val > col_max[j]) col_max[j] = abs_val;
            }
        }
    }

    /* Compute scaling factors - avoid division by zero and extreme values */
    for (i = 0; i < nrows; i++) {
        if (row_max[i] > 1e-10) {
            /* Avoid very small values that could lead to overflow */
            factors.r[i] = 1.0 / sqrt(row_max[i]);

            /* Limit scaling factors to reasonable range */
            if (factors.r[i] > 1e10) factors.r[i] = 1e10;
            if (factors.r[i] < 1e-10) factors.r[i] = 1e-10;
        } else {
            /* For zero rows, use identity scaling */
            factors.r[i] = 1.0;
            printf("WARNING: Row %d has no significant entries\n", i);
        }
    }

    for (j = 0; j < ncols; j++) {
        if (col_max[j] > 1e-10) {
            /* Avoid very small values that could lead to overflow */
            factors.c[j] = 1.0 / sqrt(col_max[j]);

            /* Limit scaling factors to reasonable range */
            if (factors.c[j] > 1e10) factors.c[j] = 1e10;
            if (factors.c[j] < 1e-10) factors.c[j] = 1e-10;
        } else {
            /* For zero columns, use identity scaling */
            factors.c[j] = 1.0;
            printf("WARNING: Column %d has no significant entries\n", j);
        }
    }

    /* Print some diagnostics for the first few scaling factors */
    printf("Sample scaling factors (first 5):\n");
    printf("  Row scaling: ");
    for (i = 0; i < 5 && i < nrows; i++) {
        printf("%.2e ", factors.r[i]);
    }
    printf("\n  Col scaling: ");
    for (j = 0; j < 5 && j < ncols; j++) {
        printf("%.2e ", factors.c[j]);
    }
    printf("\n");

    /* Free temporary arrays */
    free(row_max);
    free(col_max);

    return factors;

cleanup:
    /* Clean up on error */
    if (factors.r) free(factors.r);
    if (factors.c) free(factors.c);
    if (row_max) free(row_max);
    if (col_max) free(col_max);

    /* Return empty factors */
    factors.r = factors.c = NULL;
    return factors;
}

/* Apply scaling to matrix and right-hand side - safe version */
int apply_scaling(double *values, int *rowind, int *colptr,
                      double *rhs, int nrows, int ncols, int nnz,
                      scaling_factors_t factors) {
    int j, k, i;

    /* Validate inputs */
    if (!values || !rowind || !colptr || !rhs ||
        !factors.r || !factors.c ||
        nrows <= 0 || ncols <= 0 || nnz <= 0) {
        printf("ERROR: Invalid inputs to apply_scaling_safe\n");
        return -1;
    }

    /* Scale the matrix: A_scaled = diag(r) * A * diag(c) */
    for (j = 0; j < ncols; j++) {
        /* Validate column pointers */
        if (colptr[j] < 0 || colptr[j] > nnz) {
            printf("ERROR: Invalid column pointer at column %d: %d\n", j, colptr[j]);
            return -1;
        }

        if (j < ncols-1 && (colptr[j+1] < colptr[j] || colptr[j+1] > nnz)) {
            printf("ERROR: Invalid next column pointer at column %d: %d\n", j, colptr[j+1]);
            return -1;
        }

        int col_end = (j < ncols-1) ? colptr[j+1] : nnz;

        for (k = colptr[j]; k < col_end; k++) {
            /* Validate indices */
            if (k < 0 || k >= nnz) {
                printf("ERROR: k index out of bounds: %d (nnz=%d)\n", k, nnz);
                return -1;
            }

            i = rowind[k];

            if (i < 0 || i >= nrows) {
                printf("ERROR: Invalid row index at position %d: %d\n", k, i);
                return -1;
            }

            /* Apply scaling with safety checks */
            if (i < nrows && j < ncols && k < nnz) {
                values[k] = values[k] * factors.r[i] * factors.c[j];
            }
        }
    }

    /* Scale the right-hand side: b_scaled = diag(r) * b */
    for (i = 0; i < nrows; i++) {
        rhs[i] = rhs[i] * factors.r[i];
    }

    return 0;
}

///* Un-scale the solution: x = diag(c) * x_scaled - safe version */
//int unscale_solution(double *solution, int nrows, double *c) {
//    int i;
//
//    /* Validate inputs */
//    if (!solution || !c || nrows <= 0) {
//        printf("ERROR: Invalid inputs to unscale_solution_safe\n");
//        return -1;
//    }
//
//    for (i = 0; i < nrows; i++) {
//        solution[i] = solution[i] * c[i];
//    }
//
//    return 0;
//}

/* Un-scale the solution: x = diag(c) * x_scaled - fixed version */
int unscale_solution(double *solution, int nrows, double *c) {
    int i;

    /* Validate inputs */
    if (!solution || !c || nrows <= 0) {
        printf("ERROR: Invalid inputs to unscale_solution\n");
        return -1;
    }

    /* In scipy.spsolve with equilibration, only column scaling factors
     * are used to unscale the solution vector.
     * This is because if Ax=b becomes (Dr*A*Dc)(Dc^-1*x)=(Dr*b)
     * where Dr and Dc are row and column scaling matrices,
     * then the solution to the original system is x = Dc^-1 * x_scaled
     */
    for (i = 0; i < nrows; i++) {
        /* Apply column scaling factor */
        /* For symmetric systems where nrows == ncols, this works directly */
        solution[i] = solution[i] * c[i];
    }

    return 0;
}


///* Size-adaptive SuperLU wrapper - uses different strategies based on matrix size */
//int solve_sparse_system(double *values, int *rowind, int *colptr,
//                      int nrows, int ncols, int nnz,
//                      double *rhs, double *solution) {
//    /* Print size information */
//    printf("SuperLU Wrapper Debug Info:\n");
//    printf("  Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
//           nrows, ncols, nnz);
//
//    /* Initialize variables */
//    SuperMatrix A, L, U, B;
//    int *perm_r = NULL;
//    int *perm_c = NULL;
//    double *rhs_copy = NULL;
//    double *values_copy = NULL;
//    int *rowind_copy = NULL;
//    int *colptr_copy = NULL;
//    int info = 0;
//    superlu_options_t options;
//    SuperLUStat_t stat;
//    scaling_factors_t scaling = {NULL, NULL};
//
//    /* Set all structure pointers to NULL initially */
//    A.Store = NULL;
//    B.Store = NULL;
//    L.Store = NULL;
//    U.Store = NULL;
//
//    /* Set default options */
//    set_default_options(&options);
//
//    /* Configure options based on matrix size */
//    if (nrows <= 100) {
//        printf("Small matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = NATURAL;
//        options.DiagPivotThresh = 1.0;
//    } else if (nrows <= 1000) {
//        printf("Medium matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = COLAMD;
//        options.DiagPivotThresh = 0.1;
//    } else {
//        printf("Large matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = MMD_AT_PLUS_A;
//        options.DiagPivotThresh = 0.001;
//    }
//
//    options.SymmetricMode = NO;
//    options.PivotGrowth = NO;
//    options.ConditionNumber = NO;
//    options.IterRefine = NOREFINE;
//
//    /* Allocate permutation vectors */
//    perm_r = (int *) malloc(nrows * sizeof(int));
//    perm_c = (int *) malloc(ncols * sizeof(int));
//    if (!perm_r || !perm_c) {
//        printf("ERROR: Failed to allocate permutation vectors\n");
//        goto cleanup;
//    }
//
//    /* Create copies of input arrays */
//    values_copy = (double *) malloc(nnz * sizeof(double));
//    rowind_copy = (int *) malloc(nnz * sizeof(int));
//    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
//
//    if (!values_copy || !rowind_copy || !colptr_copy) {
//        printf("ERROR: Failed to allocate matrix copy buffers\n");
//        goto cleanup;
//    }
//
//    /* Copy data */
//    memcpy(values_copy, values, nnz * sizeof(double));
//    memcpy(rowind_copy, rowind, nnz * sizeof(int));
//    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
//
//    /* Compute and apply scipy-like scaling (equilibration) */
//    printf("Applying scipy-like matrix equilibration...\n");
//    scaling = compute_scaling(values_copy, rowind_copy, colptr_copy, nrows, ncols, nnz);
//
//    if (!scaling.r || !scaling.c) {
//        printf("ERROR: Failed to compute scaling factors\n");
//        goto cleanup;
//    }
//
//    apply_scaling(values_copy, rowind_copy, colptr_copy, rhs_copy,
//                 nrows, ncols, nnz, scaling);
//
//    /* Create compressed column matrix for SuperLU */
//    dCreate_CompCol_Matrix(&A, nrows, ncols, nnz, values_copy, rowind_copy, colptr_copy,
//                          SLU_NC, SLU_D, SLU_GE);
//
//    /* Create a copy of right-hand side */
//    rhs_copy = (double *) malloc(nrows * sizeof(double));
//    if (!rhs_copy) {
//        printf("ERROR: Failed to allocate RHS copy\n");
//        goto cleanup;
//    }
//    memcpy(rhs_copy, rhs, nrows * sizeof(double));
//
//    /* Create right-hand side matrix B */
//    dCreate_Dense_Matrix(&B, nrows, 1, rhs_copy, nrows, SLU_DN, SLU_D, SLU_GE);
//
//    /* Initialize statistics */
//    StatInit(&stat);
//
//    /* Try the simple dgssv approach first */
//    printf("Using dgssv for solving the system\n");
//
//    /* This is the simplest way to solve the system in SuperLU */
//    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
//
//    /* Copy solution if successful */
//    if (info == 0) {
//        printf("Solver successful, copying solution\n");
//        DNformat *Bstore = (DNformat *) B.Store;
//        double *Bval = (double *) Bstore->nzval;
//        memcpy(solution, Bval, nrows * sizeof(double));
//        /* Unscale solution to get original system solution */
//        unscale_solution(solution, nrows, scaling.c);
//
//        printf("Solution unscaled to match original system\n");
//    } else {
//        printf("Solver failed with info = %d\n", info);
//        if (info > 0 && info <= nrows) {
//            printf("Zero pivot found at element %d\n", info);
//        }
//    }
//
//cleanup:
//    /* Free SuperLU structures - check if they were actually created first */
//    printf("Cleaning up SuperLU structures\n");
//
//    /* Clean up SuperLU structures if they were initialized */
//    if (A.Store) Destroy_CompCol_Matrix(&A);
//    if (B.Store) Destroy_SuperMatrix_Store(&B);
//    if (L.Store) Destroy_SuperNode_Matrix(&L);
//    if (U.Store) Destroy_CompCol_Matrix(&U);
//
//    /* Free allocated memory - only free if it was allocated */
//    if (perm_r) free(perm_r);
//    if (perm_c) free(perm_c);
//    if (scaling.r) free(scaling.r);
//    if (scaling.c) free(scaling.c);
//
//    /* These are now owned by SuperLU and freed by the Destroy_* functions
//       Note: DO NOT free these directly as they're now managed by SuperLU
//    if (values_copy) free(values_copy);
//    if (rowind_copy) free(rowind_copy);
//    if (colptr_copy) free(colptr_copy);
//    if (rhs_copy) free(rhs_copy);
//    */
//
//    /* Free statistics structure */
//    StatFree(&stat);
//
//    return info;
//}

///* Size-adaptive SuperLU wrapper - uses different strategies based on matrix size */
//int solve_sparse_system(double *values, int *rowind, int *colptr,
//                      int nrows, int ncols, int nnz,
//                      double *rhs, double *solution) {
//    /* Print size information */
//    printf("SuperLU Wrapper Debug Info:\n");
//    printf("  Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
//           nrows, ncols, nnz);
//
//    /* Initialize variables */
//    SuperMatrix A, L, U, B;
//    int *perm_r = NULL;
//    int *perm_c = NULL;
//    double *rhs_copy = NULL;
//    double *values_copy = NULL;
//    int *rowind_copy = NULL;
//    int *colptr_copy = NULL;
//    int info = 0;
//    superlu_options_t options;
//    SuperLUStat_t stat;
//    scaling_factors_t scaling = {NULL, NULL};
//
//    /* Set all structure pointers to NULL initially */
//    A.Store = NULL;
//    B.Store = NULL;
//    L.Store = NULL;
//    U.Store = NULL;
//
//    /* Set default options */
//    set_default_options(&options);
//
//    /* Configure options based on matrix size */
//    if (nrows <= 100) {
//        printf("Small matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = NATURAL;
//        options.DiagPivotThresh = 1.0;
//    } else if (nrows <= 1000) {
//        printf("Medium matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = COLAMD;
//        options.DiagPivotThresh = 0.1;
//    } else {
//        printf("Large matrix strategy: nrows=%d\n", nrows);
//        options.ColPerm = MMD_AT_PLUS_A;
//        options.DiagPivotThresh = 0.001;
//    }
//
//    options.SymmetricMode = NO;
//    options.PivotGrowth = NO;
//    options.ConditionNumber = NO;
//    options.IterRefine = NOREFINE;
//
//    /* Allocate permutation vectors */
//    perm_r = (int *) malloc(nrows * sizeof(int));
//    perm_c = (int *) malloc(ncols * sizeof(int));
//    if (!perm_r || !perm_c) {
//        printf("ERROR: Failed to allocate permutation vectors\n");
//        goto cleanup;
//    }
//
//    /* Create copies of input arrays */
//    values_copy = (double *) malloc(nnz * sizeof(double));
//    rowind_copy = (int *) malloc(nnz * sizeof(int));
//    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
//
//    if (!values_copy || !rowind_copy || !colptr_copy) {
//        printf("ERROR: Failed to allocate matrix copy buffers\n");
//        goto cleanup;
//    }
//
//    /* Copy data */
//    memcpy(values_copy, values, nnz * sizeof(double));
//    memcpy(rowind_copy, rowind, nnz * sizeof(int));
//    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
//
//    /* Create a copy of right-hand side - do this BEFORE scaling */
//    rhs_copy = (double *) malloc(nrows * sizeof(double));
//    if (!rhs_copy) {
//        printf("ERROR: Failed to allocate RHS copy\n");
//        goto cleanup;
//    }
//    memcpy(rhs_copy, rhs, nrows * sizeof(double));
//
//    /* Compute scipy-like scaling (equilibration) */
//    printf("Computing matrix equilibration factors...\n");
//    scaling = compute_scaling(values_copy, rowind_copy, colptr_copy, nrows, ncols, nnz);
//
//    if (!scaling.r || !scaling.c) {
//        printf("ERROR: Failed to compute scaling factors\n");
//        goto cleanup;
//    }
//
//    /* Apply scaling to matrix and RHS */
//    printf("Applying scaling to matrix and RHS...\n");
//    if (apply_scaling(values_copy, rowind_copy, colptr_copy, rhs_copy,
//                     nrows, ncols, nnz, scaling) != 0) {
//        printf("ERROR: Failed to apply scaling\n");
//        goto cleanup;
//    }
//
//    /* Create compressed column matrix for SuperLU */
//    dCreate_CompCol_Matrix(&A, nrows, ncols, nnz, values_copy, rowind_copy, colptr_copy,
//                          SLU_NC, SLU_D, SLU_GE);
//
//    /* Create right-hand side matrix B */
//    dCreate_Dense_Matrix(&B, nrows, 1, rhs_copy, nrows, SLU_DN, SLU_D, SLU_GE);
//
//    /* Initialize statistics */
//    StatInit(&stat);
//
//    /* Solve the system */
//    printf("Using dgssv for solving the system\n");
//    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
//
//    /* Copy solution if successful */
//    if (info == 0) {
//        printf("Solver successful, copying solution\n");
//        DNformat *Bstore = (DNformat *) B.Store;
//        double *Bval = (double *) Bstore->nzval;
//
//        /* Copy solution first */
//        memcpy(solution, Bval, nrows * sizeof(double));
//
//        /* Unscale solution to get original system solution */
//        printf("Unscaling solution using column factors...\n");
//
//        /* IMPORTANT: In scipy, we only use the column scaling factors for unscaling */
//        if (unscale_solution(solution, nrows, scaling.c) != 0) {
//            printf("WARNING: Issue during solution unscaling\n");
//        } else {
//            printf("Solution unscaled successfully\n");
//        }
//    } else {
//        printf("Solver failed with info = %d\n", info);
//        if (info > 0 && info <= nrows) {
//            printf("Zero pivot found at element %d\n", info);
//        }
//    }
//
//cleanup:
//    /* Free SuperLU structures - check if they were actually created first */
//    printf("Cleaning up SuperLU structures\n");
//
//    /* Clean up SuperLU structures if they were initialized */
//    if (A.Store) Destroy_CompCol_Matrix(&A);
//    if (B.Store) Destroy_SuperMatrix_Store(&B);
//    if (L.Store) Destroy_SuperNode_Matrix(&L);
//    if (U.Store) Destroy_CompCol_Matrix(&U);
//
//    /* Free allocated memory - only free if it was allocated */
//    if (perm_r) free(perm_r);
//    if (perm_c) free(perm_c);
//    if (scaling.r) free(scaling.r);
//    if (scaling.c) free(scaling.c);
//
//    /* These are now owned by SuperLU and freed by the Destroy_* functions */
//    StatFree(&stat);
//
//    return info;
//}

///* Size-adaptive SuperLU wrapper with safe scaling implementation */
//int solve_sparse_system(double *values, int *rowind, int *colptr,
//                      int nrows, int ncols, int nnz,
//                      double *rhs, double *solution) {
//    /* Print size information */
//    printf("SuperLU Wrapper Debug Info:\n");
//    printf("  Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
//           nrows, ncols, nnz);
//
//    /* Initialize variables */
//    SuperMatrix A, L, U, B;
//    int *perm_r = NULL;
//    int *perm_c = NULL;
//    double *rhs_copy = NULL;
//    double *values_copy = NULL;
//    int *rowind_copy = NULL;
//    int *colptr_copy = NULL;
//    int info = 0;
//    superlu_options_t options;
//    SuperLUStat_t stat;
//    scaling_factors_t scaling = {NULL, NULL};
//    int i, j, k;
//
//    /* Set all structure pointers to NULL initially */
//    A.Store = NULL;
//    B.Store = NULL;
//    L.Store = NULL;
//    U.Store = NULL;
//
//    /* Set default options */
//    set_default_options(&options);
//
//    /* Configure options based on matrix size and match scipy's defaults */
//    printf("Using scipy-compatible options\n");
//    options.ColPerm = COLAMD;       /* Same as scipy default */
//    options.DiagPivotThresh = 1.0;  /* Same as scipy default */
//    options.SymmetricMode = NO;
//    options.PivotGrowth = NO;
//    options.ConditionNumber = NO;
//    options.IterRefine = NOREFINE;
//
//    /* Allocate permutation vectors */
//    perm_r = (int *) malloc(nrows * sizeof(int));
//    perm_c = (int *) malloc(ncols * sizeof(int));
//    if (!perm_r || !perm_c) {
//        printf("ERROR: Failed to allocate permutation vectors\n");
//        goto cleanup;
//    }
//
//    /* Create copies of input arrays */
//    values_copy = (double *) malloc(nnz * sizeof(double));
//    rowind_copy = (int *) malloc(nnz * sizeof(int));
//    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
//
//    if (!values_copy || !rowind_copy || !colptr_copy) {
//        printf("ERROR: Failed to allocate matrix copy buffers\n");
//        goto cleanup;
//    }
//
//    /* Copy data */
//    memcpy(values_copy, values, nnz * sizeof(double));
//    memcpy(rowind_copy, rowind, nnz * sizeof(int));
//    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
//
//    /* Create a copy of right-hand side - BEFORE scaling */
//    rhs_copy = (double *) malloc(nrows * sizeof(double));
//    if (!rhs_copy) {
//        printf("ERROR: Failed to allocate RHS copy\n");
//        goto cleanup;
//    }
//    memcpy(rhs_copy, rhs, nrows * sizeof(double));
//
//    /* Compute row and column scaling factors directly here for safety */
//    printf("Computing scaling factors...\n");
//
//    /* Allocate scaling vectors with default values (identity scaling) */
//    scaling.r = (double*) malloc(nrows * sizeof(double));
//    scaling.c = (double*) malloc(ncols * sizeof(double));
//
//    if (!scaling.r || !scaling.c) {
//        printf("ERROR: Failed to allocate scaling vectors\n");
//        goto cleanup;
//    }
//
//    /* Initialize to identity scaling */
//    for (i = 0; i < nrows; i++) {
//        scaling.r[i] = 1.0;
//    }
//    for (j = 0; j < ncols; j++) {
//        scaling.c[j] = 1.0;
//    }
//
//    /* Find row and column maximums */
//    double *row_max = (double*) malloc(nrows * sizeof(double));
//    double *col_max = (double*) malloc(ncols * sizeof(double));
//
//    if (!row_max || !col_max) {
//        printf("ERROR: Failed to allocate max value arrays\n");
//        if (row_max) free(row_max);
//        if (col_max) free(col_max);
//        goto cleanup;
//    }
//
//    /* Initialize max arrays */
//    for (i = 0; i < nrows; i++) {
//        row_max[i] = 0.0;
//    }
//    for (j = 0; j < ncols; j++) {
//        col_max[j] = 0.0;
//    }
//
//    /* Iterate through matrix to find maximums */
//    for (j = 0; j < ncols; j++) {
//        int col_start = colptr[j];
//        int col_end = (j < ncols-1) ? colptr[j+1] : nnz;
//
//        for (k = col_start; k < col_end; k++) {
//            if (k < 0 || k >= nnz) {
//                printf("ERROR: k index out of bounds: %d (nnz=%d)\n", k, nnz);
//                free(row_max);
//                free(col_max);
//                goto cleanup;
//            }
//
//            i = rowind[k];
//
//            if (i < 0 || i >= nrows) {
//                printf("ERROR: Invalid row index at position %d: %d\n", k, i);
//                free(row_max);
//                free(col_max);
//                goto cleanup;
//            }
//
//            double abs_val = fabs(values_copy[k]);
//
//            if (abs_val > row_max[i]) row_max[i] = abs_val;
//            if (abs_val > col_max[j]) col_max[j] = abs_val;
//        }
//    }
//
//    /* Count non-empty rows/columns */
//    int non_empty_rows = 0, non_empty_cols = 0;
//    for (i = 0; i < nrows; i++) {
//        if (row_max[i] > 1e-10) non_empty_rows++;
//    }
//    for (j = 0; j < ncols; j++) {
//        if (col_max[j] > 1e-10) non_empty_cols++;
//    }
//
//    printf("Matrix has %d/%d non-empty rows and %d/%d non-empty columns\n",
//           non_empty_rows, nrows, non_empty_cols, ncols);
//
//    /* Check if matrix is too singular to scale */
//    if (non_empty_rows < nrows/2 || non_empty_cols < ncols/2) {
//        printf("WARNING: Matrix appears too sparse/singular - using identity scaling\n");
//        /* Keep identity scaling - already initialized */
//    } else {
//        /* Compute scaling factors with safety bounds */
//        for (i = 0; i < nrows; i++) {
//            if (row_max[i] > 1e-10) {
//                /* Use a lower limit for scaling to avoid extreme values */
//                scaling.r[i] = 1.0 / sqrt(row_max[i]);
//
//                /* Limit scaling factors to reasonable range */
//                if (scaling.r[i] > 1e5) scaling.r[i] = 1e5;
//                if (scaling.r[i] < 1e-5) scaling.r[i] = 1e-5;
//            } else {
//                scaling.r[i] = 1.0;  /* Identity scaling for empty rows */
//                printf("WARNING: Row %d has no significant entries\n", i);
//            }
//        }
//
//        for (j = 0; j < ncols; j++) {
//            if (col_max[j] > 1e-10) {
//                scaling.c[j] = 1.0 / sqrt(col_max[j]);
//
//                /* Limit scaling factors */
//                if (scaling.c[j] > 1e5) scaling.c[j] = 1e5;
//                if (scaling.c[j] < 1e-5) scaling.c[j] = 1e-5;
//            } else {
//                scaling.c[j] = 1.0;  /* Identity scaling for empty columns */
//                printf("WARNING: Column %d has no significant entries\n", j);
//            }
//        }
//    }
//
//    /* Print sample scaling factors */
//    printf("Sample scaling factors (first 5):\n");
//    printf("  Row scaling: ");
//    for (i = 0; i < 5 && i < nrows; i++) {
//        printf("%.2e ", scaling.r[i]);
//    }
//    printf("\n  Col scaling: ");
//    for (j = 0; j < 5 && j < ncols; j++) {
//        printf("%.2e ", scaling.c[j]);
//    }
//    printf("\n");
//
//    /* Free temp arrays */
//    free(row_max);
//    free(col_max);
//
//    /* Apply scaling safely */
//    printf("Applying scaling to matrix and RHS...\n");
//
//    /* Scale the matrix: A_scaled = diag(r) * A * diag(c) */
//    for (j = 0; j < ncols; j++) {
//        int col_start = colptr_copy[j];
//        int col_end = (j < ncols-1) ? colptr_copy[j+1] : nnz;
//
//        for (k = col_start; k < col_end; k++) {
//            i = rowind_copy[k];
//            values_copy[k] = values_copy[k] * scaling.r[i] * scaling.c[j];
//        }
//    }
//
//    /* Scale the right-hand side: b_scaled = diag(r) * b */
//    for (i = 0; i < nrows; i++) {
//        rhs_copy[i] = rhs_copy[i] * scaling.r[i];
//    }
//
//    /* Create compressed column matrix for SuperLU */
//    dCreate_CompCol_Matrix(&A, nrows, ncols, nnz, values_copy, rowind_copy, colptr_copy,
//                          SLU_NC, SLU_D, SLU_GE);
//
//    /* Create right-hand side matrix B */
//    dCreate_Dense_Matrix(&B, nrows, 1, rhs_copy, nrows, SLU_DN, SLU_D, SLU_GE);
//
//    /* Initialize statistics */
//    StatInit(&stat);
//
//    /* Solve the system */
//    printf("Using dgssv to solve the scaled system\n");
//    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
//
//    /* Copy and unscale solution if successful */
//    if (info == 0) {
//        printf("Solver successful, copying solution\n");
//        DNformat *Bstore = (DNformat *) B.Store;
//        double *Bval = (double *) Bstore->nzval;
//
//        /* Copy the solution */
//        memcpy(solution, Bval, nrows * sizeof(double));
//
//        /* Unscale the solution - x = diag(c) * x_scaled */
//        printf("Unscaling solution to match original system...\n");
//
//        /* In scipy, only column scaling factors are used for unscaling */
//        if (ncols == nrows) {  /* Square system with matching dimensions */
//            for (i = 0; i < nrows; i++) {
//                solution[i] = solution[i] * scaling.c[i];
//            }
//        } else {
//            /* For non-square systems, make sure we don't go out of bounds */
//            int min_dim = (nrows < ncols) ? nrows : ncols;
//            for (i = 0; i < min_dim; i++) {
//                solution[i] = solution[i] * scaling.c[i];
//            }
//            /* For any remaining elements, use identity scaling */
//            for (i = min_dim; i < nrows; i++) {
//                /* No change needed for identity scaling */
//            }
//        }
//    } else {
//        printf("Solver failed with info = %d\n", info);
//        if (info > 0 && info <= nrows) {
//            printf("Zero pivot found at element %d\n", info);
//        }
//    }
//
//cleanup:
//    /* Free SuperLU structures - check if they were actually created first */
//    printf("Cleaning up SuperLU structures\n");
//
//    /* Clean up SuperLU structures if they were initialized */
//    if (A.Store) Destroy_CompCol_Matrix(&A);
//    if (B.Store) Destroy_SuperMatrix_Store(&B);
//    if (L.Store) Destroy_SuperNode_Matrix(&L);
//    if (U.Store) Destroy_CompCol_Matrix(&U);
//
//    /* Free allocated memory - only free if it was allocated */
//    if (perm_r) free(perm_r);
//    if (perm_c) free(perm_c);
//    if (scaling.r) free(scaling.r);
//    if (scaling.c) free(scaling.c);
//
//    /* These are now owned by SuperLU and freed by the Destroy_* functions */
//    StatFree(&stat);
//
//    return info;
//}

/* Robust SuperLU wrapper with regularization for singular matrices */
int solve_sparse_system(double *values, int *rowind, int *colptr,
                      int nrows, int ncols, int nnz,
                      double *rhs, double *solution) {
    /* Print size information */
    printf("SuperLU Wrapper Debug Info:\n");
    printf("  Matrix dimensions: nrows=%d, ncols=%d, nnz=%d\n",
           nrows, ncols, nnz);

    /* Initialize variables */
    SuperMatrix A, L, U, B;
    int *perm_r = NULL;
    int *perm_c = NULL;
    double *rhs_copy = NULL;
    double *values_copy = NULL;
    int *rowind_copy = NULL;
    int *colptr_copy = NULL;
    int info = 0;
    superlu_options_t options;
    SuperLUStat_t stat;
    scaling_factors_t scaling = {NULL, NULL};
    int i, j, k;
    double regularization_applied = 0.0;

    /* Set all structure pointers to NULL initially */
    A.Store = NULL;
    B.Store = NULL;
    L.Store = NULL;
    U.Store = NULL;

    /* Set default options */
    set_default_options(&options);

    /* Configure options to match scipy's defaults */
    printf("Using scipy-compatible solver options\n");
    options.ColPerm = COLAMD;       /* Same as scipy default */
    options.DiagPivotThresh = 1.0;  /* Same as scipy default */
    options.SymmetricMode = NO;
    options.PivotGrowth = NO;
    options.ConditionNumber = NO;
    options.IterRefine = NOREFINE;  /* Fixed: Using NOREFINE instead of DOUBLE */

    /* Allocate permutation vectors */
    perm_r = (int *) malloc(nrows * sizeof(int));
    perm_c = (int *) malloc(ncols * sizeof(int));
    if (!perm_r || !perm_c) {
        printf("ERROR: Failed to allocate permutation vectors\n");
        goto cleanup;
    }

    /* Create copies of input arrays */
    values_copy = (double *) malloc(nnz * sizeof(double));
    rowind_copy = (int *) malloc(nnz * sizeof(int));
    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));

    if (!values_copy || !rowind_copy || !colptr_copy) {
        printf("ERROR: Failed to allocate matrix copy buffers\n");
        goto cleanup;
    }

    /* Copy data */
    memcpy(values_copy, values, nnz * sizeof(double));
    memcpy(rowind_copy, rowind, nnz * sizeof(int));
    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));

    /* Create a copy of right-hand side - BEFORE scaling */
    rhs_copy = (double *) malloc(nrows * sizeof(double));
    if (!rhs_copy) {
        printf("ERROR: Failed to allocate RHS copy\n");
        goto cleanup;
    }
    memcpy(rhs_copy, rhs, nrows * sizeof(double));

    /* Apply regularization to diagonal elements for singular/ill-conditioned matrices
     * This helps to handle matrices with empty rows or near-singular properties
     */
    printf("Analyzing matrix structure for potential regularization...\n");

    /* Find diagonal elements and check for empty rows */
    int *diag_indices = (int *) malloc(nrows * sizeof(int));
    int *has_diag = (int *) malloc(nrows * sizeof(int));
    double *row_abs_sum = (double *) malloc(nrows * sizeof(double));

    if (!diag_indices || !has_diag || !row_abs_sum) {
        printf("ERROR: Failed to allocate regularization analysis arrays\n");
        if (diag_indices) free(diag_indices);
        if (has_diag) free(has_diag);
        if (row_abs_sum) free(row_abs_sum);
        goto cleanup;
    }

    /* Initialize arrays */
    for (i = 0; i < nrows; i++) {
        diag_indices[i] = -1;  /* No diagonal found yet */
        has_diag[i] = 0;       /* No diagonal element yet */
        row_abs_sum[i] = 0.0;  /* Initialize row sum */
    }

    /* Traverse matrix to find diagonal elements and compute row sums */
    for (j = 0; j < ncols && j < nrows; j++) {  /* Only check up to min(nrows, ncols) */
        int col_start = colptr_copy[j];
        int col_end = (j < ncols-1) ? colptr_copy[j+1] : nnz;

        for (k = col_start; k < col_end; k++) {
            i = rowind_copy[k];
            double abs_val = fabs(values_copy[k]);
            row_abs_sum[i] += abs_val;

            if (i == j) {  /* Diagonal element found */
                diag_indices[i] = k;
                has_diag[i] = 1;
            }
        }
    }

    /* Count rows with and without diagonal elements */
    int empty_rows = 0;
    int rows_without_diag = 0;
    int small_diag_elems = 0;

    for (i = 0; i < nrows && i < ncols; i++) {
        if (row_abs_sum[i] < 1e-10) {
            empty_rows++;
        }

        if (!has_diag[i]) {
            rows_without_diag++;
        } else if (fabs(values_copy[diag_indices[i]]) < 1e-10) {
            small_diag_elems++;
        }
    }

    printf("Matrix analysis: %d empty rows, %d rows missing diagonal, %d near-zero diagonals\n",
           empty_rows, rows_without_diag, small_diag_elems);

    /* Decide whether to apply regularization */
    if (empty_rows > 0 || rows_without_diag > 0 || small_diag_elems > 0) {
        printf("Applying targeted regularization for stability...\n");

        /* Find the average magnitude in the matrix for scaling regularization */
        double avg_magnitude = 0.0;
        double count = 0;
        for (k = 0; k < nnz; k++) {
            if (fabs(values_copy[k]) > 1e-10) {
                avg_magnitude += fabs(values_copy[k]);
                count += 1;
            }
        }
        avg_magnitude = (count > 0) ? avg_magnitude / count : 1.0;

        /* Compute regularization based on average magnitude */
        double reg_value = avg_magnitude * 1e-8;  /* Typically 1e-8 to 1e-12 of average */
        if (reg_value < 1e-14) reg_value = 1e-14;  /* Minimum floor for stability */
        printf("Using regularization value: %.2e\n", reg_value);

        /* Apply regularization to problematic rows/columns */
        int added_elems = 0;
        for (i = 0; i < nrows && i < ncols; i++) {
            if (row_abs_sum[i] < 1e-10 || !has_diag[i]) {
                /* This is a problematic row - add a diagonal element or strengthen existing one */
                if (has_diag[i]) {
                    /* Strengthen existing diagonal */
                    values_copy[diag_indices[i]] += reg_value;
                    added_elems++;
                } else {
                    /* Would need to add a new element - this requires reallocating arrays
                     * For now, we'll skip this as it's complex to implement
                     * A better approach is to pre-process the matrix before calling this function
                     */
                    printf("WARNING: Unable to add missing diagonal for row %d - matrix will remain singular\n", i);
                }
            } else if (has_diag[i] && fabs(values_copy[diag_indices[i]]) < 1e-10) {
                /* Small existing diagonal - strengthen it */
                values_copy[diag_indices[i]] = (values_copy[diag_indices[i]] >= 0) ?
                                              reg_value : -reg_value;
                added_elems++;
            }
        }

        regularization_applied = reg_value;
        printf("Regularization applied to %d elements\n", added_elems);
    }

    /* Free regularization analysis arrays */
    free(diag_indices);
    free(has_diag);
    free(row_abs_sum);

    /* Compute scaling factors */
    printf("Computing scaling factors...\n");

    /* Allocate scaling vectors with default values (identity scaling) */
    scaling.r = (double*) malloc(nrows * sizeof(double));
    scaling.c = (double*) malloc(ncols * sizeof(double));

    if (!scaling.r || !scaling.c) {
        printf("ERROR: Failed to allocate scaling vectors\n");
        goto cleanup;
    }

    /* Initialize to identity scaling */
    for (i = 0; i < nrows; i++) {
        scaling.r[i] = 1.0;
    }
    for (j = 0; j < ncols; j++) {
        scaling.c[j] = 1.0;
    }

    /* Find row and column maximums */
    double *row_max = (double*) malloc(nrows * sizeof(double));
    double *col_max = (double*) malloc(ncols * sizeof(double));

    if (!row_max || !col_max) {
        printf("ERROR: Failed to allocate max value arrays\n");
        if (row_max) free(row_max);
        if (col_max) free(col_max);
        goto cleanup;
    }

    /* Initialize max arrays */
    for (i = 0; i < nrows; i++) {
        row_max[i] = 0.0;
    }
    for (j = 0; j < ncols; j++) {
        col_max[j] = 0.0;
    }

    /* Iterate through matrix to find maximums */
    for (j = 0; j < ncols; j++) {
        int col_start = colptr_copy[j];
        int col_end = (j < ncols-1) ? colptr_copy[j+1] : nnz;

        for (k = col_start; k < col_end; k++) {
            i = rowind_copy[k];

            double abs_val = fabs(values_copy[k]);

            if (abs_val > row_max[i]) row_max[i] = abs_val;
            if (abs_val > col_max[j]) col_max[j] = abs_val;
        }
    }

    /* Count non-empty rows/columns */
    int non_empty_rows = 0, non_empty_cols = 0;
    for (i = 0; i < nrows; i++) {
        if (row_max[i] > 1e-10) non_empty_rows++;
    }
    for (j = 0; j < ncols; j++) {
        if (col_max[j] > 1e-10) non_empty_cols++;
    }

    printf("Matrix has %d/%d non-empty rows and %d/%d non-empty columns\n",
           non_empty_rows, nrows, non_empty_cols, ncols);

    /* Compute scaling factors with safety bounds */
    for (i = 0; i < nrows; i++) {
        if (row_max[i] > 1e-10) {
            /* Use a lower limit for scaling to avoid extreme values */
            scaling.r[i] = 1.0 / sqrt(row_max[i]);

            /* Limit scaling factors to reasonable range */
            if (scaling.r[i] > 1e4) scaling.r[i] = 1e4;
            if (scaling.r[i] < 1e-4) scaling.r[i] = 1e-4;
        } else {
            scaling.r[i] = 1.0;  /* Identity scaling for empty rows */
            printf("WARNING: Row %d has no significant entries\n", i);
        }
    }

    for (j = 0; j < ncols; j++) {
        if (col_max[j] > 1e-10) {
            scaling.c[j] = 1.0 / sqrt(col_max[j]);

            /* Limit scaling factors */
            if (scaling.c[j] > 1e4) scaling.c[j] = 1e4;
            if (scaling.c[j] < 1e-4) scaling.c[j] = 1e-4;
        } else {
            scaling.c[j] = 1.0;  /* Identity scaling for empty columns */
            printf("WARNING: Column %d has no significant entries\n", j);
        }
    }

    /* Print sample scaling factors */
    printf("Sample scaling factors (first 5):\n");
    printf("  Row scaling: ");
    for (i = 0; i < 5 && i < nrows; i++) {
        printf("%.2e ", scaling.r[i]);
    }
    printf("\n  Col scaling: ");
    for (j = 0; j < 5 && j < ncols; j++) {
        printf("%.2e ", scaling.c[j]);
    }
    printf("\n");

    /* Free temp arrays */
    free(row_max);
    free(col_max);

    /* Apply scaling safely */
    printf("Applying scaling to matrix and RHS...\n");

    /* Scale the matrix: A_scaled = diag(r) * A * diag(c) */
    for (j = 0; j < ncols; j++) {
        int col_start = colptr_copy[j];
        int col_end = (j < ncols-1) ? colptr_copy[j+1] : nnz;

        for (k = col_start; k < col_end; k++) {
            i = rowind_copy[k];
            values_copy[k] = values_copy[k] * scaling.r[i] * scaling.c[j];
        }
    }

    /* Scale the right-hand side: b_scaled = diag(r) * b */
    for (i = 0; i < nrows; i++) {
        rhs_copy[i] = rhs_copy[i] * scaling.r[i];
    }

    /* Create compressed column matrix for SuperLU */
    dCreate_CompCol_Matrix(&A, nrows, ncols, nnz, values_copy, rowind_copy, colptr_copy,
                          SLU_NC, SLU_D, SLU_GE);

    /* Create right-hand side matrix B */
    dCreate_Dense_Matrix(&B, nrows, 1, rhs_copy, nrows, SLU_DN, SLU_D, SLU_GE);

    /* Initialize statistics */
    StatInit(&stat);

    /* Solve the system */
    printf("Using dgssv to solve the regularized and scaled system\n");
    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

    /* If first attempt fails, try with different permutation strategy */
    if (info != 0) {
        printf("First attempt failed with info=%d, trying with MMD_AT_PLUS_A permutation\n", info);

        /* Free previous factors if allocated */
        if (L.Store) { Destroy_SuperNode_Matrix(&L); L.Store = NULL; }
        if (U.Store) { Destroy_CompCol_Matrix(&U); U.Store = NULL; }

        /* Change permutation strategy */
        options.ColPerm = MMD_AT_PLUS_A;
        StatInit(&stat);  /* Reset statistics */

        /* Try again */
        dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
    }

    /* Copy and unscale solution if successful */
    if (info == 0) {
        printf("Solver successful, copying solution\n");
        DNformat *Bstore = (DNformat *) B.Store;
        double *Bval = (double *) Bstore->nzval;

        /* Copy the solution */
        memcpy(solution, Bval, nrows * sizeof(double));

        /* Unscale the solution - x = diag(c) * x_scaled */
        printf("Unscaling solution to match original system...\n");

        /* Only use column scaling factors for unscaling */
        for (i = 0; i < nrows && i < ncols; i++) {
            solution[i] = solution[i] * scaling.c[i];
        }

        if (regularization_applied > 0.0) {
            printf("Note: Solution obtained with regularization (%.2e)\n", regularization_applied);
        }
    } else {
        printf("Solver failed with info = %d\n", info);
        if (info > 0 && info <= nrows) {
            printf("Zero pivot found at element %d\n", info);
            printf("System appears to be singular, no unique solution exists\n");
        }
    }

cleanup:
    /* Free SuperLU structures - check if they were actually created first */
    printf("Cleaning up SuperLU structures\n");

    /* Clean up SuperLU structures if they were initialized */
    if (A.Store) Destroy_CompCol_Matrix(&A);
    if (B.Store) Destroy_SuperMatrix_Store(&B);
    if (L.Store) Destroy_SuperNode_Matrix(&L);
    if (U.Store) Destroy_CompCol_Matrix(&U);

    /* Free allocated memory */
    if (perm_r) free(perm_r);
    if (perm_c) free(perm_c);
    if (scaling.r) free(scaling.r);
    if (scaling.c) free(scaling.c);

    /* These are now owned by SuperLU and freed by the Destroy_* functions */
    StatFree(&stat);

    return info;
}