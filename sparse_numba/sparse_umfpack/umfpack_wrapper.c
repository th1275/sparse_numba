/* umfpack_wrapper.c - A minimal wrapper for UMFPACK to be used with Numba */

// File name: umfpack_wrapper.c
//
// Every function returns a status code of sparse_numba_codes.h and never
// prints (the debug messages compile in under SPARSE_NUMBA_DEBUG only).
// A handle is an index into this module's registry (sparse_numba_registry.h).
//
// UMFPACK is interface-only: this module has no
// link-time dependency on SuiteSparse. umfpack.h gives the constants and
// the Control / Info layout; the user's library is loaded at the first
// call that needs it (LoadLibraryEx / dlopen) and its umfpack_di_* entry
// points are read into a held table (GetProcAddress / dlsym). Not found ->
// SN_BACKEND_ABSENT.
#include "umfpack_wrapper.h"
#include "../sparse_numba_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  /* For DBL_MAX */
#include <umfpack.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif
#else
#include <dlfcn.h>
#endif

#ifdef SPARSE_NUMBA_DEBUG
#include <stdio.h>
#define DEBUG_PRINT(fmt, ...) printf("[UMFPACK Debug] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#define SN_THIS_BACKEND SN_BACKEND_UMFPACK
#define SN_REGISTRY_FIRST SN_THIS_BACKEND
#define SN_REGISTRY_LAST SN_THIS_BACKEND
#include "../sparse_numba_registry.h"  /* the handle registry */


/* ================================================================
 * The runtime load of the user's UMFPACK. How the library is
 * found: SPARSE_NUMBA_UMFPACK_LIBRARY, when set, names its path (and only
 * that path is tried); else the platform names below through the
 * platform loader (on Windows LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: the
 * directories the package and the caller add with os.add_dll_directory,
 * the package's own vendored SuiteSparse directory in the private tree).
 * The first call that needs the library loads it, once per process; the
 * outcome is held (a library found later is not picked up).
 * ================================================================ */

#define SN_UMFPACK_LIBRARY_ENV "SPARSE_NUMBA_UMFPACK_LIBRARY"

static const char *const sn_umf_names[] = {
#if defined(_WIN32)
    "libumfpack.dll", "umfpack.dll",
#elif defined(__APPLE__)
    "libumfpack.6.dylib", "libumfpack.5.dylib", "libumfpack.dylib",
#else
    "libumfpack.so.6", "libumfpack.so.5", "libumfpack.so",
#endif
    NULL
};

/* the umfpack_di_* prototypes of umfpack.h (int32_t indices) */
typedef void (*sn_umf_defaults_t)(double *Control);
typedef int (*sn_umf_symbolic_t)(int32_t n_row, int32_t n_col, const int32_t *Ap,
                                 const int32_t *Ai, const double *Ax, void **Symbolic,
                                 const double *Control, double *Info);
typedef int (*sn_umf_numeric_t)(const int32_t *Ap, const int32_t *Ai, const double *Ax,
                                void *Symbolic, void **Numeric, const double *Control,
                                double *Info);
typedef int (*sn_umf_solve_t)(int sys, const int32_t *Ap, const int32_t *Ai,
                              const double *Ax, double *X, const double *B, void *Numeric,
                              const double *Control, double *Info);
typedef void (*sn_umf_free_t)(void **Object);
typedef int (*sn_umf_get_numeric_t)(int32_t *Lp, int32_t *Lj, double *Lx, int32_t *Up,
                                    int32_t *Ui, double *Ux, int32_t *P, int32_t *Q,
                                    double *Dx, int32_t *do_recip, double *Rs,
                                    void *Numeric);
typedef int (*sn_umf_get_lunz_t)(int32_t *lnz, int32_t *unz, int32_t *n_row,
                                 int32_t *n_col, int32_t *nz_udiag, void *Numeric);
typedef void (*sn_umf_version_t)(int version[3]);

static struct {
    sn_umf_defaults_t defaults;
    sn_umf_symbolic_t symbolic;
    sn_umf_numeric_t numeric;
    sn_umf_solve_t solve;
    sn_umf_free_t free_symbolic;
    sn_umf_free_t free_numeric;
    sn_umf_get_numeric_t get_numeric;
    sn_umf_get_lunz_t get_lunz;
    int version[3];       /* umfpack_version's; zero when the library has none */
    char source[512];     /* the path or the name the library loaded from */
} sn_umf;

static int sn_umf_state = 0;   /* 0 not tried, 1 loaded, -1 not found */
static char sn_umf_lock_flag = 0;

#ifdef _WIN32
typedef HMODULE sn_lib_t;
static sn_lib_t sn_lib_open(const char *name, int is_path) {
    return LoadLibraryExA(name, NULL, is_path
        ? (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
        : LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
static void *sn_lib_sym(sn_lib_t lib, const char *name) {
    return (void *)GetProcAddress(lib, name);
}
static void sn_lib_close(sn_lib_t lib) { FreeLibrary(lib); }
static int sn_env(const char *name, char *out, int len) {
    DWORD got = GetEnvironmentVariableA(name, out, (DWORD)len);
    return got > 0 && got < (DWORD)len;
}
#else
typedef void *sn_lib_t;
static sn_lib_t sn_lib_open(const char *name, int is_path) {
    (void)is_path;
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}
static void *sn_lib_sym(sn_lib_t lib, const char *name) { return dlsym(lib, name); }
static void sn_lib_close(sn_lib_t lib) { dlclose(lib); }
static int sn_env(const char *name, char *out, int len) {
    const char *v = getenv(name);
    if (!v || !v[0] || (int)strlen(v) >= len) return 0;
    strcpy(out, v);
    return 1;
}
#endif

/* the entry points into the table; 0 if one is missing */
static int sn_umf_resolve(sn_lib_t lib) {
    sn_umf_version_t version;
    sn_umf.defaults = (sn_umf_defaults_t)sn_lib_sym(lib, "umfpack_di_defaults");
    sn_umf.symbolic = (sn_umf_symbolic_t)sn_lib_sym(lib, "umfpack_di_symbolic");
    sn_umf.numeric = (sn_umf_numeric_t)sn_lib_sym(lib, "umfpack_di_numeric");
    sn_umf.solve = (sn_umf_solve_t)sn_lib_sym(lib, "umfpack_di_solve");
    sn_umf.free_symbolic = (sn_umf_free_t)sn_lib_sym(lib, "umfpack_di_free_symbolic");
    sn_umf.free_numeric = (sn_umf_free_t)sn_lib_sym(lib, "umfpack_di_free_numeric");
    sn_umf.get_numeric = (sn_umf_get_numeric_t)sn_lib_sym(lib, "umfpack_di_get_numeric");
    sn_umf.get_lunz = (sn_umf_get_lunz_t)sn_lib_sym(lib, "umfpack_di_get_lunz");
    if (!sn_umf.defaults || !sn_umf.symbolic || !sn_umf.numeric || !sn_umf.solve ||
            !sn_umf.free_symbolic || !sn_umf.free_numeric || !sn_umf.get_numeric ||
            !sn_umf.get_lunz) {
        return 0;
    }
    /* UMFPACK >= 6 (SuiteSparse >= 7) has umfpack_version; older ones none */
    version = (sn_umf_version_t)sn_lib_sym(lib, "umfpack_version");
    sn_umf.version[0] = sn_umf.version[1] = sn_umf.version[2] = 0;
    if (version) version(sn_umf.version);
    return 1;
}

/* dst = src, cut to len - 1 characters and terminated */
static void sn_copy(char *dst, int len, const char *src) {
    int i = 0;
    for (; i < len - 1 && src[i]; i++) dst[i] = src[i];
    dst[i] = 0;
}

/* the decimal digits of v >= 0 at dst; returns the next position */
static char *sn_digits(char *dst, int v) {
    char tmp[12];
    int k = 0;
    if (v < 0) v = 0;
    do {
        tmp[k++] = (char)('0' + v % 10);
        v /= 10;
    } while (v > 0);
    while (k > 0) *dst++ = tmp[--k];
    return dst;
}

static int sn_umf_try(const char *name, int is_path) {
    sn_lib_t lib = sn_lib_open(name, is_path);
    if (!lib) return 0;
    if (!sn_umf_resolve(lib)) {
        sn_lib_close(lib);
        memset(&sn_umf, 0, sizeof(sn_umf));
        return 0;
    }
    sn_copy(sn_umf.source, (int)sizeof(sn_umf.source), name);
    return 1;  /* the library stays loaded for the process's life */
}

static void sn_umf_load(void) {
    char path[sizeof(sn_umf.source)];
    int found = 0;
    if (sn_env(SN_UMFPACK_LIBRARY_ENV, path, (int)sizeof(path))) {
        found = sn_umf_try(path, 1);
    } else {
        for (int i = 0; sn_umf_names[i] && !found; i++) {
            found = sn_umf_try(sn_umf_names[i], 0);
        }
    }
    DEBUG_PRINT("UMFPACK library %s", found ? sn_umf.source : "not found");
    __atomic_store_n(&sn_umf_state, found ? 1 : -1, __ATOMIC_RELEASE);
}

/* SN_OK once the library is loaded, else SN_BACKEND_ABSENT */
static int sn_umf_ready(void) {
    int state = __atomic_load_n(&sn_umf_state, __ATOMIC_ACQUIRE);
    if (state == 0) {
        while (__atomic_test_and_set(&sn_umf_lock_flag, __ATOMIC_ACQUIRE)) {
        }
        if (__atomic_load_n(&sn_umf_state, __ATOMIC_ACQUIRE) == 0) sn_umf_load();
        __atomic_clear(&sn_umf_lock_flag, __ATOMIC_RELEASE);
        state = __atomic_load_n(&sn_umf_state, __ATOMIC_ACQUIRE);
    }
    return state == 1 ? SN_OK : SN_BACKEND_ABSENT;
}

/* UMFPACK_ORDERING_NONE needs SuiteSparse >= 5; a library
 * without umfpack_version (UMFPACK < 6) cannot say which it is */
static int sn_umf_has_ordering_none(void) {
    return sn_umf.version[0] >= 6;
}


int umfpack_library_info(char *version, int version_len, char *source, int source_len) {
    int code = sn_umf_ready();
    if (version && version_len > 0) {
        if (code != SN_OK) {
            sn_copy(version, version_len, "absent");
        } else if (sn_umf.version[0] > 0) {
            char text[40];
            char *q = sn_digits(text, sn_umf.version[0]);
            *q++ = '.';
            q = sn_digits(q, sn_umf.version[1]);
            *q++ = '.';
            q = sn_digits(q, sn_umf.version[2]);
            *q = 0;
            sn_copy(version, version_len, text);
        } else {
            sn_copy(version, version_len, "unknown");
        }
    }
    if (source && source_len > 0)
        sn_copy(source, source_len, code == SN_OK ? sn_umf.source : "");
    return code;
}


/* The factorizations' Control: pivot tolerance 1.0 and
 * the ordering's pairing, set explicitly: AMD with STRATEGY_SYMMETRIC (the
 * 0.1.11 factorization: AMD is UMFPACK_DEFAULT_ORDERING), COLAMD =
 * UMFPACK_ORDERING_AMD under STRATEGY_UNSYMMETRIC (UMFPACK runs COLAMD for
 * the unsymmetric strategy), NATURAL = UMFPACK_ORDERING_NONE under
 * STRATEGY_UNSYMMETRIC. Returns SN_OK or SN_ORDERING_UNSUPPORTED. */
static int umf_factorize_control(double *Control, int ordering) {
    sn_umf.defaults(Control);
    Control[UMFPACK_PIVOT_TOLERANCE] = 1.0;  // Max pivot tolerance
    switch (ordering) {
    case SN_ORDERING_AMD:
        Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
        Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
        return SN_OK;
    case SN_ORDERING_COLAMD:
        Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
        Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
        return SN_OK;
    case SN_ORDERING_NATURAL:
        if (!sn_umf_has_ordering_none()) return SN_ORDERING_UNSUPPORTED;
        Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
        Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_NONE;
        return SN_OK;
    default:
        return SN_ORDERING_UNSUPPORTED;  /* MMD_AT_PLUS_A is not UMFPACK's */
    }
}


/* SINGULAR = k: the 1-based position of U's first zero diagonal entry */
static int umf_singular_index(void *Numeric, int n) {
    int k = 1;
    int do_recip = 0;
    double *Dx = (double *) malloc(n * sizeof(double));
    if (!Dx) return SN_ALLOC_FAILED;
    if (sn_umf.get_numeric(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                               Dx, &do_recip, NULL, Numeric) == UMFPACK_OK) {
        for (int i = 0; i < n; i++) {
            if (Dx[i] == 0.0) {
                k = i + 1;
                break;
            }
        }
    }
    free(Dx);
    return k;
}


/* an UMFPACK status as a status code (the singular case handled by the caller) */
static int umf_code(int status) {
    if (status == UMFPACK_OK) return SN_OK;
    if (status == UMFPACK_ERROR_out_of_memory) return SN_ALLOC_FAILED;
    DEBUG_PRINT("UMFPACK failed with status %d", status);
    return SN_BACKEND_ERROR;
}


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
    int status;
    int code = SN_ALLOC_FAILED;

    if (!values || !rowind || !colptr || !rhs || !solution) return SN_BAD_SHAPE;
    if (nrows <= 0 || ncols <= 0 || nnz <= 0) return SN_BAD_SHAPE;
    code = sn_umf_ready();
    if (code != SN_OK) return code;
    code = SN_ALLOC_FAILED;

    /* Get the 0.1.11 control parameters (AMD, STRATEGY_SYMMETRIC) */
    umf_factorize_control(Control, SN_ORDERING_AMD);

    /* Allocate memory with error checking */
    values_copy = (double *) malloc(nnz * sizeof(double));
    rowind_copy = (int *) malloc(nnz * sizeof(int));
    colptr_copy = (int *) malloc((ncols+1) * sizeof(int));
    rhs_copy = (double *) malloc(nrows * sizeof(double));
    if (!values_copy || !rowind_copy || !colptr_copy || !rhs_copy) {
        DEBUG_PRINT("Failed to allocate the copies");
        goto cleanup;
    }

    /* Copy data */
    memcpy(values_copy, values, nnz * sizeof(double));
    memcpy(rowind_copy, rowind, nnz * sizeof(int));
    memcpy(colptr_copy, colptr, (ncols+1) * sizeof(int));
    memcpy(rhs_copy, rhs, nrows * sizeof(double));

    /* Validate matrix format */
    if (colptr_copy[0] != 0 || colptr_copy[ncols] != nnz) {
        DEBUG_PRINT("Error: Invalid CSC format");
        code = SN_BAD_INDPTR;
        goto cleanup;
    }

    /* Symbolic analysis phase */
    status = sn_umf.symbolic(nrows, ncols, colptr_copy, rowind_copy, values_copy,
                              &Symbolic, Control, Info);
    if (status != UMFPACK_OK) {
        code = umf_code(status);
        goto cleanup;
    }

    /* Numeric factorization phase */
    status = sn_umf.numeric(colptr_copy, rowind_copy, values_copy,
                              Symbolic, &Numeric, Control, Info);
    if (status == UMFPACK_WARNING_singular_matrix) {
        code = umf_singular_index(Numeric, ncols);
        goto cleanup;
    }
    if (status != UMFPACK_OK) {
        code = umf_code(status);
        goto cleanup;
    }

    /* Solve the system */
    status = sn_umf.solve(UMFPACK_A, colptr_copy, rowind_copy, values_copy,
                           solution, rhs_copy, Numeric, Control, Info);
    code = umf_code(status);

cleanup:
    /* Free resources */
    if (Numeric) sn_umf.free_numeric(&Numeric);
    if (Symbolic) sn_umf.free_symbolic(&Symbolic);
    if (values_copy) free(values_copy);
    if (rowind_copy) free(rowind_copy);
    if (colptr_copy) free(colptr_copy);
    if (rhs_copy) free(rhs_copy);

    return code;
}


/* ================================================================
 * Pre-factorization API: factorize once, solve many times
 * ================================================================ */

/* The factors of one handle. UMFPACK's umfpack_di_solve requires the
 * original CSC arrays, so we store copies alongside the Symbolic/Numeric
 * handles; the Symbolic is held for refactorize. */
typedef struct {
    void *Symbolic;
    void *Numeric;
    int *colptr;
    int *rowind;
    double *values;
    double Control[UMFPACK_CONTROL];       /* the factorizations' */
    double SolveControl[UMFPACK_CONTROL];  /* UMFPACK's defaults, the solves' (0.1.11) */
    int n;
    int nnz;
    int ordering;
    int last_status;  /* UMFPACK's status of the last factorization */
    int state;        /* SN_OK while Numeric is usable for a solve */
} umfpack_factors_t;


static void umf_destroy(umfpack_factors_t *f) {
    if (!f) return;
    if (f->Numeric) sn_umf.free_numeric(&f->Numeric);
    if (f->Symbolic) sn_umf.free_symbolic(&f->Symbolic);
    if (f->colptr) free(f->colptr);
    if (f->rowind) free(f->rowind);
    if (f->values) free(f->values);
    free(f);
}


/* umfpack_di_numeric on the held arrays and Symbolic */
static int umf_numeric(umfpack_factors_t *f) {
    double Info[UMFPACK_INFO];
    int status;

    if (f->Numeric) sn_umf.free_numeric(&f->Numeric);
    status = sn_umf.numeric(f->colptr, f->rowind, f->values,
                                f->Symbolic, &f->Numeric, f->Control, Info);
    f->last_status = status;
    if (status == UMFPACK_WARNING_singular_matrix) {
        f->state = umf_singular_index(f->Numeric, f->n);
    } else {
        f->state = umf_code(status);
    }
    return f->state;
}


int factorize_sparse_system_ordered(double *values, int *rowind, int *colptr,
                                    int n, int nnz, int ordering,
                                    int64_t *handle_out) {
    umfpack_factors_t *f;
    double Info[UMFPACK_INFO];
    int status;
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
    if (colptr[0] != 0 || colptr[n] != nnz) return SN_BAD_INDPTR;
    code = sn_umf_ready();
    if (code != SN_OK) return code;

    f = (umfpack_factors_t*)calloc(1, sizeof(umfpack_factors_t));
    if (!f) return SN_ALLOC_FAILED;
    f->n = n;
    f->nnz = nnz;
    f->ordering = ordering;
    /* the Control held per handle: the ordering's */
    code = umf_factorize_control(f->Control, ordering);
    if (code != SN_OK) {
        free(f);
        return code;
    }
    sn_umf.defaults(f->SolveControl);

    /* Copy CSC arrays */
    f->values = (double*)malloc(nnz * sizeof(double));
    f->rowind = (int*)malloc(nnz * sizeof(int));
    f->colptr = (int*)malloc((n + 1) * sizeof(int));
    if (!f->values || !f->rowind || !f->colptr) {
        DEBUG_PRINT("Failed to allocate data copies");
        umf_destroy(f);
        return SN_ALLOC_FAILED;
    }
    memcpy(f->values, values, nnz * sizeof(double));
    memcpy(f->rowind, rowind, nnz * sizeof(int));
    memcpy(f->colptr, colptr, (n + 1) * sizeof(int));

    /* Symbolic analysis */
    status = sn_umf.symbolic(n, n, f->colptr, f->rowind, f->values,
                                 &f->Symbolic, f->Control, Info);
    if (status != UMFPACK_OK) {
        f->Symbolic = NULL;
        code = umf_code(status);
        umf_destroy(f);
        return code;
    }

    /* Numeric factorization */
    code = umf_numeric(f);
    if (code != SN_OK) {
        umf_destroy(f);
        return code;
    }

    code = sn_register(f, SN_THIS_BACKEND, handle_out);
    if (code != SN_OK) {
        umf_destroy(f);
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
                                           SN_ORDERING_AMD, handle_out);
}


int refactorize_sparse_system(int64_t handle, double *values, int nnz) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    umfpack_factors_t *f = (umfpack_factors_t*)p;
    if (code != SN_OK) return code;
    if (!values || nnz != f->nnz) return SN_BAD_SHAPE;
    memcpy(f->values, values, nnz * sizeof(double));
    return umf_numeric(f);
}


/* umfpack_di_solve per column; rhs / solution column-major unless
 * row_major (then (n, nrhs) C-contiguous) */
static int umf_solve(umfpack_factors_t *f, double *rhs, double *solution,
                     int nrhs, int row_major) {
    double Info[UMFPACK_INFO];
    int status;
    int n = f->n;

    if (f->state != SN_OK) return f->state;

    if (!row_major || nrhs == 1) {
        /* Solve: A*x = b using stored factors and CSC arrays */
        for (int k = 0; k < nrhs; k++) {
            status = sn_umf.solve(UMFPACK_A,
                                      f->colptr, f->rowind, f->values,
                                      solution + (size_t)k * n,
                                      rhs + (size_t)k * n,
                                      f->Numeric, f->SolveControl, Info);
            if (status != UMFPACK_OK) return umf_code(status);
        }
        return SN_OK;
    }

    {
        double *col = (double *) malloc(2 * (size_t)n * sizeof(double));
        if (!col) return SN_ALLOC_FAILED;
        for (int k = 0; k < nrhs; k++) {
            for (int i = 0; i < n; i++) col[i] = rhs[(size_t)i * nrhs + k];
            status = sn_umf.solve(UMFPACK_A,
                                      f->colptr, f->rowind, f->values,
                                      col + n, col,
                                      f->Numeric, f->SolveControl, Info);
            if (status != UMFPACK_OK) {
                free(col);
                return umf_code(status);
            }
            for (int i = 0; i < n; i++) solution[(size_t)i * nrhs + k] = col[n + i];
        }
        free(col);
    }
    return SN_OK;
}


int solve_with_factors(int64_t handle, double *rhs, double *solution, int nrhs) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    umfpack_factors_t *f = (umfpack_factors_t*)p;
    if (code != SN_OK) return code;
    if (!rhs || !solution || nrhs <= 0) {
        DEBUG_PRINT("Error: NULL pointer passed to solve_with_factors");
        return SN_BAD_SHAPE;
    }
    return umf_solve(f, rhs, solution, nrhs, 0);
}


int solve_into_with_factors(int64_t handle, double *rhs, double *solution,
                            int n, int nrhs) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    umfpack_factors_t *f = (umfpack_factors_t*)p;
    if (code != SN_OK) return code;
    if (!rhs || !solution || nrhs <= 0 || n != f->n) return SN_BAD_SHAPE;
    return umf_solve(f, rhs, solution, nrhs, 1);
}


int sparse_factors_info(int64_t handle, int64_t *iout, double *dout) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    umfpack_factors_t *f = (umfpack_factors_t*)p;
    int32_t lnz = 0, unz = 0, n_row = 0, n_col = 0, nz_udiag = 0;
    if (code != SN_OK) return code;
    if (!iout || !dout) return SN_BAD_SHAPE;
    if (f->Numeric)
        sn_umf.get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, f->Numeric);
    iout[SN_INFO_N] = f->n;
    iout[SN_INFO_NNZ_L] = lnz;
    iout[SN_INFO_NNZ_U] = unz;
    iout[SN_INFO_BACKEND] = SN_THIS_BACKEND;
    iout[SN_INFO_ORDERING] = f->ordering;
    iout[SN_INFO_LAST_STATUS] = f->last_status;
    dout[SN_DINFO_PIVOT_TOLERANCE] = f->Control[UMFPACK_PIVOT_TOLERANCE];
    return SN_OK;
}


int free_sparse_factors(int64_t handle) {
    void *p = NULL;
    int code = sn_release(handle, &p);
    if (code != SN_OK) return code;
    umf_destroy((umfpack_factors_t*)p);
    return SN_OK;
}
