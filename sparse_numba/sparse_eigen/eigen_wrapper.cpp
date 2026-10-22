/* eigen_wrapper.cpp - A minimal wrapper for Eigen's sparse solvers to be used with Numba */

// File name: eigen_wrapper.cpp
//
// The Eigen 3.4.0 backends: EIGEN_LU = SparseLU at
// NaturalOrdering / COLAMDOrdering / AMDOrdering (three instantiations
// behind one switch), EIGEN_LDLT = SimplicialLDLT (Lower, AMDOrdering) for
// a symmetric pattern. Header-only, compiled in with -DEIGEN_MPL2_ONLY (a
// non-MPL2 Eigen header is a build error) and
// -DEIGEN_DONT_PARALLELIZE (single-threaded by construction).
//
// Every extern "C" function returns a status code of sparse_numba_codes.h,
// never prints, and lets no C++ exception out (std::bad_alloc ->
// SN_ALLOC_FAILED, any other -> SN_BACKEND_ERROR). A handle is an index
// into this module's registry (sparse_numba_registry.h).
#include "eigen_wrapper.h"
#include "../sparse_numba_codes.h"

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#ifndef EIGEN_MPL2_ONLY
#error "the Eigen shim is built with -DEIGEN_MPL2_ONLY (REQUIREMENT sec 7.1 Q8 A)"
#endif
#ifndef EIGEN_DONT_PARALLELIZE
#error "the Eigen shim is built with -DEIGEN_DONT_PARALLELIZE (REQUIREMENT sec 3.3)"
#endif

#define SN_REGISTRY_FIRST SN_BACKEND_EIGEN_LU
#define SN_REGISTRY_LAST SN_BACKEND_EIGEN_LDLT
#include "../sparse_numba_registry.h"  /* the handle registry */

namespace {

typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int> SpMat;


/* One solver kind behind one interface: analyzePattern once, factorize per
 * (re)factorization, solve on (n, k) column-major blocks */
class Solver {
public:
    virtual ~Solver() {}
    virtual void analyze(const SpMat &A) = 0;
    virtual void factorize(const SpMat &A) = 0;
    virtual Eigen::ComputationInfo info() const = 0;
    virtual void solve(const double *b, double *x, int n, int k) const = 0;
    virtual int64_t nnz_l() const = 0;
    virtual int64_t nnz_u() const = 0;
    virtual double pivot_tolerance() const = 0;
    /* false when the factorization succeeded but the input is refused by
     * the kind (LDLT: a D entry <= 0, not SPD) */
    virtual bool admissible() const { return true; }
};


template <class Ordering>
class LuSolver : public Solver {
    Eigen::SparseLU<SpMat, Ordering> s_;
public:
    LuSolver() { s_.setPivotThreshold(1.0); }  /* Eigen's default, set: handle_info */
    void analyze(const SpMat &A) { s_.analyzePattern(A); }
    void factorize(const SpMat &A) { s_.factorize(A); }
    Eigen::ComputationInfo info() const { return s_.info(); }
    void solve(const double *b, double *x, int n, int k) const {
        Eigen::Map<const Eigen::MatrixXd> B(b, n, k);
        Eigen::Map<Eigen::MatrixXd> X(x, n, k);
        X = s_.solve(B);
    }
    int64_t nnz_l() const { return (int64_t)s_.nnzL(); }
    int64_t nnz_u() const { return (int64_t)s_.nnzU(); }
    double pivot_tolerance() const { return 1.0; }
};


class LdltSolver : public Solver {
    Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int> > s_;
public:
    void analyze(const SpMat &A) { s_.analyzePattern(A); }
    void factorize(const SpMat &A) { s_.factorize(A); }
    Eigen::ComputationInfo info() const { return s_.info(); }
    void solve(const double *b, double *x, int n, int k) const {
        Eigen::Map<const Eigen::MatrixXd> B(b, n, k);
        Eigen::Map<Eigen::MatrixXd> X(x, n, k);
        X = s_.solve(B);
    }
    /* the strictly lower L (its unit diagonal and D not stored in it) */
    int64_t nnz_l() const { return (int64_t)s_.matrixL().nestedExpression().nonZeros(); }
    int64_t nnz_u() const { return 0; }  /* U = L^T, not stored */
    double pivot_tolerance() const { return 0.0; }  /* no pivoting */
    /* every D entry > 0: SPD input; an indefinite matrix whose
     * pivots are nonzero factors with Success and is refused here (one
     * pass over n values) */
    bool admissible() const { return (s_.vectorD().array() > 0.0).all(); }
};


/* The factors of one handle: the matrix in Eigen's sorted column storage,
 * the place of each caller entry in it (for refactorize), and the solver */
struct Factors {
    int backend;
    int ordering;
    int n;
    int nnz;              /* the caller's entries */
    SpMat A;
    std::vector<int> place;
    Solver *solver;
    int last_status;      /* Eigen's ComputationInfo of the last factorization */
    int state;            /* SN_OK while the factors are usable for a solve */

    Factors() : backend(0), ordering(0), n(0), nnz(0), solver(NULL),
                last_status(0), state(SN_BACKEND_ERROR) {}
    ~Factors() { delete solver; }
};


/* Eigen's ComputationInfo as a status code: a numerical issue (a zero
 * pivot) is SINGULAR = 1 (Eigen does not report which pivot) */
int eigen_code(Eigen::ComputationInfo info) {
    if (info == Eigen::Success) return SN_OK;
    if (info == Eigen::NumericalIssue) return 1;
    return SN_BACKEND_ERROR;
}


/* The caller's values into the held storage: zeroed, then each entry
 * added at its place (a repeated row index in a column summed) */
void eigen_values(Factors *f, const double *values) {
    double *v = f->A.valuePtr();
    std::fill(v, v + f->A.nonZeros(), 0.0);
    for (int p = 0; p < f->nnz; p++) v[f->place[p]] += values[p];
}


/* The pattern into Eigen's compressed storage, each column's rows sorted */
int eigen_pattern(Factors *f, const int *rowind, const int *colptr) {
    int n = f->n;
    std::vector<int> outer(n + 1, 0);
    std::vector<int> inner;
    std::vector<std::pair<int, int> > col;

    inner.reserve(f->nnz);
    f->place.assign(f->nnz, 0);
    for (int j = 0; j < n; j++) {
        if (colptr[j + 1] < colptr[j]) return SN_BAD_INDPTR;
        col.clear();
        for (int p = colptr[j]; p < colptr[j + 1]; p++) {
            if (rowind[p] < 0 || rowind[p] >= n) return SN_BAD_INDEX;
            col.push_back(std::make_pair(rowind[p], p));
        }
        std::sort(col.begin(), col.end());
        int last = -1;
        for (size_t q = 0; q < col.size(); q++) {
            if (col[q].first != last) {
                inner.push_back(col[q].first);
                last = col[q].first;
            }
            f->place[col[q].second] = (int)inner.size() - 1;
        }
        outer[j + 1] = (int)inner.size();
    }

    f->A.resize(n, n);
    f->A.resizeNonZeros((Eigen::Index)inner.size());
    std::copy(outer.begin(), outer.end(), f->A.outerIndexPtr());
    std::copy(inner.begin(), inner.end(), f->A.innerIndexPtr());
    return SN_OK;
}


/* (pattern != pattern^T) is empty: the transpose's sorted columns equal
 * the matrix's (the stored entries, explicit zeros among them) */
bool eigen_pattern_symmetric(const SpMat &A) {
    int n = (int)A.cols();
    const int *op = A.outerIndexPtr();
    const int *ip = A.innerIndexPtr();
    int nz = op[n];
    std::vector<int> tp(n + 1, 0);
    std::vector<int> ti(nz);

    for (int k = 0; k < nz; k++) tp[ip[k] + 1]++;
    for (int i = 0; i < n; i++) tp[i + 1] += tp[i];
    std::vector<int> next(tp.begin(), tp.end() - 1);
    for (int j = 0; j < n; j++)
        for (int k = op[j]; k < op[j + 1]; k++) ti[next[ip[k]]++] = j;
    return std::equal(tp.begin(), tp.end(), op) && std::equal(ti.begin(), ti.end(), ip);
}


/* factorize() on the held matrix: the factorization of factorize and
 * refactorize; on EIGEN_LDLT a D entry <= 0 is SINGULAR = 1 (= not SPD
 * here) */
int eigen_numeric(Factors *f) {
    f->solver->factorize(f->A);
    Eigen::ComputationInfo info = f->solver->info();
    f->last_status = (int)info;
    f->state = eigen_code(info);
    if (f->state == SN_OK && !f->solver->admissible()) f->state = 1;
    return f->state;
}


int eigen_factorize(int backend, double *values, int *rowind, int *colptr,
                    int n, int nnz, int ordering, int64_t *handle_out) {
    Factors *f = NULL;
    int code;

    if (!handle_out) return SN_BAD_SHAPE;
    *handle_out = 0;
    if (!values || !rowind || !colptr) return SN_BAD_SHAPE;
    if (n <= 0 || nnz <= 0) return SN_BAD_SHAPE;
    if (colptr[0] != 0 || colptr[n] != nnz) return SN_BAD_INDPTR;
    if (backend == SN_BACKEND_EIGEN_LU) {
        if (ordering != SN_ORDERING_NATURAL && ordering != SN_ORDERING_COLAMD &&
                ordering != SN_ORDERING_AMD)
            return SN_ORDERING_UNSUPPORTED;
    } else if (ordering != SN_ORDERING_AMD) {
        return SN_ORDERING_UNSUPPORTED;  /* LDLT: AMD only */
    }

    try {
        f = new Factors();
        f->backend = backend;
        f->ordering = ordering;
        f->n = n;
        f->nnz = nnz;
        code = eigen_pattern(f, rowind, colptr);
        if (code == SN_OK && backend == SN_BACKEND_EIGEN_LDLT &&
                !eigen_pattern_symmetric(f->A))
            code = SN_BAD_SHAPE;
        if (code != SN_OK) {
            delete f;
            return code;
        }
        eigen_values(f, values);

        if (backend == SN_BACKEND_EIGEN_LDLT) {
            f->solver = new LdltSolver();
        } else if (ordering == SN_ORDERING_NATURAL) {
            f->solver = new LuSolver<Eigen::NaturalOrdering<int> >();
        } else if (ordering == SN_ORDERING_COLAMD) {
            f->solver = new LuSolver<Eigen::COLAMDOrdering<int> >();
        } else {
            f->solver = new LuSolver<Eigen::AMDOrdering<int> >();
        }
        f->solver->analyze(f->A);
        code = eigen_numeric(f);
    } catch (const std::bad_alloc &) {
        delete f;
        return SN_ALLOC_FAILED;
    } catch (...) {
        delete f;
        return SN_BACKEND_ERROR;
    }
    if (code != SN_OK) {
        delete f;
        return code;
    }

    code = sn_register(f, backend, handle_out);
    if (code != SN_OK) {
        delete f;
        return code;
    }
    return SN_OK;
}


/* the (n, k) row-major block <-> column-major */
void eigen_transpose(double *dst, const double *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
}

}  /* namespace */


extern "C" {

int eigen_lu_factorize_ordered(double *values, int *rowind, int *colptr,
                               int n, int nnz, int ordering,
                               int64_t *handle_out) {
    return eigen_factorize(SN_BACKEND_EIGEN_LU, values, rowind, colptr, n, nnz,
                           ordering, handle_out);
}


int eigen_ldlt_factorize_ordered(double *values, int *rowind, int *colptr,
                                 int n, int nnz, int ordering,
                                 int64_t *handle_out) {
    return eigen_factorize(SN_BACKEND_EIGEN_LDLT, values, rowind, colptr, n, nnz,
                           ordering, handle_out);
}


int eigen_refactorize(int64_t handle, double *values, int nnz) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    Factors *f = (Factors*)p;
    if (code != SN_OK) return code;
    if (!values || nnz != f->nnz) return SN_BAD_SHAPE;
    try {
        eigen_values(f, values);
        /* the pattern analyzePattern() read at factorize; factorize() again */
        return eigen_numeric(f);
    } catch (const std::bad_alloc &) {
        f->state = SN_ALLOC_FAILED;
        return SN_ALLOC_FAILED;
    } catch (...) {
        f->state = SN_BACKEND_ERROR;
        return SN_BACKEND_ERROR;
    }
}


/* k = 1: the solve reads b and writes the solution in place (a column is
 * the same bytes in either layout); k > 1: through column-major copies.
 * Eigen's solve allocates its own work inside (no promise of
 * zero allocation for Eigen) */
int eigen_solve_into(int64_t handle, double *rhs, double *solution, int n, int nrhs) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    Factors *f = (Factors*)p;
    if (code != SN_OK) return code;
    if (!rhs || !solution || nrhs <= 0 || n != f->n) return SN_BAD_SHAPE;
    if (f->state != SN_OK) return f->state;
    try {
        if (nrhs == 1) {
            f->solver->solve(rhs, solution, n, 1);
        } else {
            std::vector<double> b((size_t)n * (size_t)nrhs);
            std::vector<double> x((size_t)n * (size_t)nrhs);
            eigen_transpose(b.data(), rhs, n, nrhs);
            f->solver->solve(b.data(), x.data(), n, nrhs);
            eigen_transpose(solution, x.data(), nrhs, n);
        }
    } catch (const std::bad_alloc &) {
        return SN_ALLOC_FAILED;
    } catch (...) {
        return SN_BACKEND_ERROR;
    }
    return SN_OK;
}


int eigen_factors_info(int64_t handle, int64_t *iout, double *dout) {
    void *p = NULL;
    int code = sn_lookup(handle, &p);
    Factors *f = (Factors*)p;
    if (code != SN_OK) return code;
    if (!iout || !dout) return SN_BAD_SHAPE;
    iout[SN_INFO_N] = f->n;
    iout[SN_INFO_NNZ_L] = f->state == SN_OK ? f->solver->nnz_l() : 0;
    iout[SN_INFO_NNZ_U] = f->state == SN_OK ? f->solver->nnz_u() : 0;
    iout[SN_INFO_BACKEND] = f->backend;
    iout[SN_INFO_ORDERING] = f->ordering;
    iout[SN_INFO_LAST_STATUS] = f->last_status;
    dout[SN_DINFO_PIVOT_TOLERANCE] = f->solver->pivot_tolerance();
    return SN_OK;
}


int eigen_free_factors(int64_t handle) {
    void *p = NULL;
    int code = sn_release(handle, &p);
    if (code != SN_OK) return code;
    delete (Factors*)p;
    return SN_OK;
}


/* The Eigen version (the headers' EIGEN_*_VERSION, Macros.h) and this
 * shim's build facts (sparse_numba_registry.h), for cache_key() */
const char *eigen_build_info(void) {
    return SN_STR(EIGEN_WORLD_VERSION) "." SN_STR(EIGEN_MAJOR_VERSION) "."
           SN_STR(EIGEN_MINOR_VERSION) "," SN_BUILD_FACTS;
}

}  /* extern "C" */
