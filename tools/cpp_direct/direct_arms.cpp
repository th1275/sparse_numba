// direct_arms.cpp -- THE C++ DIRECT-CALL ARM of tools/compare_record.py (a
// downstream project's direct-call benchmark read as the reference, not
// copied).
//
// SuperLU and Eigen called directly, no sparse_numba in between, with the
// shims' options and template parameters:
//   SUPERLU     set_default_options, ColPerm NATURAL / COLAMD / MMD_AT_PLUS_A,
//               PrintStat NO, StatInit once, get_perm_c, sp_preorder, dgstrf;
//               a held dense B over a held buffer, memcpy in, dgstrs(NOTRANS),
//               memcpy out; refactorize = Fact SamePattern, dgstrf again
//               (sparse_numba/sparse_superlu/superlu_wrapper.c);
//   EIGEN_LU    Eigen::SparseLU<SparseMatrix<double, ColMajor, int>, Ordering>
//               at NaturalOrdering / COLAMDOrdering / AMDOrdering,
//               setPivotThreshold(1.0), analyzePattern + factorize; refactorize
//               = factorize again on the held pattern;
//   EIGEN_LDLT  Eigen::SimplicialLDLT<..., Lower, AMDOrdering<int>>, the same
//               (sparse_numba/sparse_eigen/eigen_wrapper.cpp).
// The input is the CSC the harness passes to sparse_numba, written by it as
// one binary file: int32 n, int32 nnz, int32 indptr[n + 1], int32
// indices[nnz], float64 data[nnz], float64 b[n] (little-endian).
//
// Built as a shared library and called IN THE HARNESS'S OWN PROCESS through
// ctypes, one call per arm, the loops inside it: the heap, the loaded
// SuperLU / OpenBLAS DLLs and the thread are the sparse_numba arms' (a
// fresh process measured the small factorizations 2-3x slower in a trial
// run: a cold heap).
//
//   int direct_arms_run(13, {name, input.bin, backend, ordering,
//       auto|function, x_out.bin, factorize_repeats, substitution_calls,
//       substitution_repeats, refactorize_calls, refactorize_repeats, cap_s,
//       threshold_s}, buf, cap)
//
// THE PROTOCOL (the harness's, the same rule): the first factorize call is
// timed; over threshold_s (or with "function") the arm is FUNCTION-ONLY: that
// factorization, one solve, one refactorize and one solve, no timing protocol.
// Otherwise the first call is the warm-up of repeat 1, and each of the
// factorize_repeats repeats times the second of two calls (the first freed).
// Substitution and refactorize: two untimed calls, one timed probe call, then
// N = min(calls, max(1, floor(cap_s / probe))) calls a repeat, each repeat's
// mean a sample. Writes the x of the substitution and the x after the
// refactorize (2n float64) to x_out.bin and one JSON object into buf.
// Returns 0 written, 2 usage, 3 input / output, 4 a backend / ordering pair
// the shims refuse, 5 a C++ exception (buf then holds the reason).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#include "slu_ddefs.h"

typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int> SpMat;
typedef std::chrono::steady_clock Clock;

static double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Csc {
    int n = 0, nnz = 0;
    std::vector<int> indptr, indices;
    std::vector<double> data, b;
};


/* One arm: factorize (a fresh factorization), solve into a held out,
 * refactorize on new values of the same pattern, release */
class Arm {
public:
    virtual ~Arm() {}
    virtual int factorize(const Csc &m) = 0;
    virtual int solve(const double *b, double *x) = 0;
    virtual int refactorize(const double *values) = 0;
    virtual void release() = 0;
    virtual long long nnz_l() const = 0;
    virtual long long nnz_u() const = 0;
};


class SluArm : public Arm {
    colperm_t colperm_;
    int n_ = 0, nnz_ = 0;
    std::vector<double> values_, bbuf_;
    std::vector<int> rowind_, colptr_, perm_r_, perm_c_, etree_;
    SuperMatrix A_, AC_, L_, U_, B_;
    NCformat Astore_;
    DNformat Bstore_;
    superlu_options_t options_;
    SuperLUStat_t stat_;
    bool have_lu_ = false, live_ = false;

    int numeric() {
        GlobalLU_t glu;
        int_t info = 0;
        int panel_size = sp_ienv(1), relax = sp_ienv(2);
        std::memset(&glu, 0, sizeof glu);
        L_.Store = NULL;
        U_.Store = NULL;
        dgstrf(&options_, &AC_, relax, panel_size, etree_.data(), NULL, 0, perm_c_.data(),
               perm_r_.data(), &L_, &U_, &glu, &stat_, &info);
        have_lu_ = (info == 0 || (info > 0 && info <= n_)) && L_.Store && U_.Store;
        return (int)info;
    }

public:
    explicit SluArm(colperm_t colperm) : colperm_(colperm) {}

    int factorize(const Csc &m) {
        n_ = m.n;
        nnz_ = m.nnz;
        values_.assign(m.data.begin(), m.data.end());
        rowind_.assign(m.indices.begin(), m.indices.end());
        colptr_.assign(m.indptr.begin(), m.indptr.end());
        perm_r_.assign(n_, 0);
        perm_c_.assign(n_, 0);
        etree_.assign(n_, 0);
        bbuf_.assign(n_, 0.0);
        A_.Stype = SLU_NC;
        A_.Dtype = SLU_D;
        A_.Mtype = SLU_GE;
        A_.nrow = n_;
        A_.ncol = n_;
        A_.Store = &Astore_;
        Astore_.nnz = nnz_;
        Astore_.nzval = values_.data();
        Astore_.rowind = rowind_.data();
        Astore_.colptr = colptr_.data();
        B_.Stype = SLU_DN;
        B_.Dtype = SLU_D;
        B_.Mtype = SLU_GE;
        B_.nrow = n_;
        B_.ncol = 1;
        B_.Store = &Bstore_;
        Bstore_.lda = n_;
        Bstore_.nzval = bbuf_.data();
        set_default_options(&options_);
        options_.ColPerm = colperm_;
        options_.PrintStat = NO;
        StatInit(&stat_);
        for (int i = 0; i < n_; i++) perm_r_[i] = perm_c_[i] = i;
        get_perm_c(options_.ColPerm, &A_, perm_c_.data());
        sp_preorder(&options_, &A_, perm_c_.data(), etree_.data(), &AC_);
        live_ = true;
        return numeric();
    }

    int solve(const double *b, double *x) {
        int info = 0;
        std::memcpy(bbuf_.data(), b, sizeof(double) * n_);
        dgstrs(NOTRANS, &L_, &U_, perm_c_.data(), perm_r_.data(), &B_, &stat_, &info);
        std::memcpy(x, bbuf_.data(), sizeof(double) * n_);
        return info;
    }

    int refactorize(const double *values) {
        std::memcpy(values_.data(), values, sizeof(double) * nnz_);
        if (have_lu_) {
            Destroy_SuperNode_Matrix(&L_);
            Destroy_CompCol_Matrix(&U_);
            have_lu_ = false;
        }
        options_.Fact = SamePattern;
        return numeric();
    }

    void release() {
        if (!live_) return;
        if (have_lu_) {
            Destroy_SuperNode_Matrix(&L_);
            Destroy_CompCol_Matrix(&U_);
            have_lu_ = false;
        }
        Destroy_CompCol_Permuted(&AC_);
        StatFree(&stat_);
        live_ = false;
    }

    long long nnz_l() const { return have_lu_ ? ((SCformat *)L_.Store)->nnz : 0; }
    long long nnz_u() const { return have_lu_ ? ((NCformat *)U_.Store)->nnz : 0; }
    ~SluArm() { release(); }
};


/* Eigen's solvers: a fresh solver object per factorization (as the shim's
 * new LuSolver / LdltSolver), the matrix held for refactorize */
template <class Solver>
class EigenArm : public Arm {
    SpMat A_;
    std::unique_ptr<Solver> s_;
    int n_ = 0;

    static int code(Eigen::ComputationInfo info) { return info == Eigen::Success ? 0 : 1 + (int)info; }

public:
    int factorize(const Csc &m) {
        n_ = m.n;
        A_ = Eigen::Map<const SpMat>(m.n, m.n, m.nnz, m.indptr.data(), m.indices.data(),
                                     m.data.data());
        s_.reset(new Solver());
        configure(*s_);
        s_->analyzePattern(A_);
        s_->factorize(A_);
        return code(s_->info());
    }

    int solve(const double *b, double *x) {
        Eigen::Map<const Eigen::MatrixXd> B(b, n_, 1);
        Eigen::Map<Eigen::MatrixXd> X(x, n_, 1);
        X = s_->solve(B);
        return 0;
    }

    int refactorize(const double *values) {
        std::memcpy(A_.valuePtr(), values, sizeof(double) * A_.nonZeros());
        s_->factorize(A_);
        return code(s_->info());
    }

    void release() { s_.reset(); }
    long long nnz_l() const;
    long long nnz_u() const;

    static void configure(Solver &s);
};

template <class Ordering>
struct LuOf { typedef Eigen::SparseLU<SpMat, Ordering> type; };
typedef Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int> > Ldlt;

template <class Solver> void EigenArm<Solver>::configure(Solver &s) { s.setPivotThreshold(1.0); }
template <> void EigenArm<Ldlt>::configure(Ldlt &) {}
template <class Solver> long long EigenArm<Solver>::nnz_l() const { return s_ ? (long long)s_->nnzL() : 0; }
template <class Solver> long long EigenArm<Solver>::nnz_u() const { return s_ ? (long long)s_->nnzU() : 0; }
template <> long long EigenArm<Ldlt>::nnz_l() const {
    return s_ ? (long long)s_->matrixL().nestedExpression().nonZeros() : 0;
}
template <> long long EigenArm<Ldlt>::nnz_u() const { return 0; }


static Arm *make_arm(const std::string &backend, const std::string &ordering, std::string &solver) {
    if (backend == "SUPERLU") {
        solver = "SuperLU dgstrf / dgstrs called directly, options.ColPerm = " + ordering;
        if (ordering == "NATURAL") return new SluArm(NATURAL);
        if (ordering == "COLAMD") return new SluArm(COLAMD);
        if (ordering == "MMD_AT_PLUS_A") return new SluArm(MMD_AT_PLUS_A);
        return NULL;
    }
    if (backend == "EIGEN_LU") {
        solver = "Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, Eigen::" +
                 std::string(ordering == "NATURAL" ? "NaturalOrdering" :
                             ordering == "COLAMD" ? "COLAMDOrdering" : "AMDOrdering") +
                 "<int>> called directly, setPivotThreshold(1.0)";
        if (ordering == "NATURAL") return new EigenArm<LuOf<Eigen::NaturalOrdering<int> >::type>();
        if (ordering == "COLAMD") return new EigenArm<LuOf<Eigen::COLAMDOrdering<int> >::type>();
        if (ordering == "AMD") return new EigenArm<LuOf<Eigen::AMDOrdering<int> >::type>();
        return NULL;
    }
    if (backend == "EIGEN_LDLT" && ordering == "AMD") {
        solver = "Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, "
                 "Eigen::Lower, Eigen::AMDOrdering<int>> called directly";
        return new EigenArm<Ldlt>();
    }
    return NULL;
}


static bool read_input(const char *path, Csc &m) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t head[2];
    bool ok = std::fread(head, sizeof(int32_t), 2, f) == 2 && head[0] > 0 && head[1] > 0;
    if (ok) {
        m.n = head[0];
        m.nnz = head[1];
        m.indptr.resize(m.n + 1);
        m.indices.resize(m.nnz);
        m.data.resize(m.nnz);
        m.b.resize(m.n);
        ok = std::fread(m.indptr.data(), sizeof(int32_t), m.n + 1, f) == (size_t)(m.n + 1) &&
             std::fread(m.indices.data(), sizeof(int32_t), m.nnz, f) == (size_t)m.nnz &&
             std::fread(m.data.data(), sizeof(double), m.nnz, f) == (size_t)m.nnz &&
             std::fread(m.b.data(), sizeof(double), m.n, f) == (size_t)m.n;
    }
    std::fclose(f);
    if (!ok || m.indptr[0] != 0 || m.indptr[m.n] != m.nnz) return false;
    /* the rows of each column strictly ascending: refactorize copies the
     * values in place over Eigen's storage */
    for (int j = 0; j < m.n; j++)
        for (int p = m.indptr[j] + 1; p < m.indptr[j + 1]; p++)
            if (m.indices[p] <= m.indices[p - 1]) return false;
    return true;
}


static std::string samples_json(const std::vector<double> &s) {
    std::string out = "[";
    char buf[64];
    for (size_t i = 0; i < s.size(); i++) {
        std::snprintf(buf, sizeof buf, "%s%.6f", i ? ", " : "", s[i]);
        out += buf;
    }
    return out + "]";
}


/* two untimed calls, one timed probe, then repeats x N calls (N cut to fit
 * cap_s from the probe); each repeat's mean (us) a sample */
template <class Call>
static int timed_loop(Call call, int calls, int repeats, double cap_s, double &probe_us,
                      int &n_actual, std::vector<double> &samples) {
    int status = 0;
    status |= call();
    status |= call();
    Clock::time_point t0 = Clock::now();
    status |= call();
    probe_us = us_since(t0);
    n_actual = calls;
    if (probe_us > 0.0) {
        double fit = std::floor(cap_s * 1e6 / probe_us);
        if (fit < (double)calls) n_actual = std::max(1, (int)fit);
    }
    for (int r = 0; r < repeats; r++) {
        t0 = Clock::now();
        for (int k = 0; k < n_actual; k++) status |= call();
        samples.push_back(us_since(t0) / n_actual);
    }
    return status;
}


/* one arm; the JSON object (or the reason of a failure) into out */
static int run_arm(int argc, const char *const *argv, std::string &out) {
    if (argc != 13) {
        out = "usage: direct_arms_run(13, {name, input.bin, backend, ordering, auto|function, "
              "x_out.bin, factorize_repeats, substitution_calls, substitution_repeats, "
              "refactorize_calls, refactorize_repeats, cap_s, threshold_s})";
        return 2;
    }
    std::string backend = argv[2], ordering = argv[3], mode = argv[4];
    int f_repeats = std::atoi(argv[6]), s_calls = std::atoi(argv[7]), s_repeats = std::atoi(argv[8]);
    int r_calls = std::atoi(argv[9]), r_repeats = std::atoi(argv[10]);
    double cap_s = std::atof(argv[11]), threshold_s = std::atof(argv[12]);
    Csc m;
    if (!read_input(argv[1], m)) {
        out = std::string("could not read ") + argv[1];
        return 3;
    }
    std::string solver;
    std::unique_ptr<Arm> arm(make_arm(backend, ordering, solver));
    if (!arm) {
        out = "no direct arm for " + backend + " x " + ordering;
        return 4;
    }
    std::vector<double> x(m.n, 0.0), after(m.n, 0.0);
    out = "{";
    char buf[1024];

    Clock::time_point t0 = Clock::now();
    int info = arm->factorize(m);
    double first_us = us_since(t0);
    bool function_only = mode == "function" || first_us > threshold_s * 1e6;
    std::snprintf(buf, sizeof buf, "\"solver\": \"%s\", \"first_factorize_us\": %.3f, ",
                  solver.c_str(), first_us);
    out += buf;
    if (info != 0) {
        std::snprintf(buf, sizeof buf, "\"status\": \"error\", \"factorize_info\": %d}", info);
        out += buf;
        return 0;
    }

    if (function_only) {
        /* one factorize (above), one solve, one refactorize and one solve */
        int s_info = arm->solve(m.b.data(), x.data());
        t0 = Clock::now();
        int r_info = arm->refactorize(m.data.data());
        double refac_us = us_since(t0);
        int a_info = arm->solve(m.b.data(), after.data());
        std::snprintf(buf, sizeof buf,
                      "\"status\": \"function_only\", \"function_only_reason\": \"%s\", "
                      "\"solve_info\": %d, \"refactorize_info\": %d, \"refactorize_single_us\": %.3f, "
                      "\"solve_after_refactorize_info\": %d, \"nnz_L\": %lld, \"nnz_U\": %lld}",
                      mode == "function" ? "the sparse_numba arm of this pair is function-only"
                                         : "the first factorize call over the threshold",
                      s_info, r_info, refac_us, a_info, arm->nnz_l(), arm->nnz_u());
        out += buf;
    } else {
        /* the first call was repeat 1's warm-up */
        std::vector<double> f_samples;
        for (int r = 0; r < f_repeats; r++) {
            if (r > 0) {
                arm->release();
                info |= arm->factorize(m);
            }
            arm->release();
            t0 = Clock::now();
            info |= arm->factorize(m);
            f_samples.push_back(us_since(t0));
        }
        double s_probe = 0.0, r_probe = 0.0;
        int s_n = 0, r_n = 0;
        std::vector<double> s_samples, r_samples;
        const double *bp = m.b.data();
        double *xp = x.data();
        int s_status = timed_loop([&] { return arm->solve(bp, xp); }, s_calls, s_repeats, cap_s,
                                  s_probe, s_n, s_samples);
        const double *vp = m.data.data();
        int r_status = timed_loop([&] { return arm->refactorize(vp); }, r_calls, r_repeats, cap_s,
                                  r_probe, r_n, r_samples);
        int a_info = arm->solve(m.b.data(), after.data());
        std::snprintf(buf, sizeof buf,
                      "\"status\": \"measured\", \"factorize_info\": %d, \"substitution_status\": %d, "
                      "\"refactorize_status\": %d, \"solve_after_refactorize_info\": %d, "
                      "\"substitution_probe_us\": %.3f, \"substitution_calls_actual\": %d, "
                      "\"refactorize_probe_us\": %.3f, \"refactorize_calls_actual\": %d, "
                      "\"nnz_L\": %lld, \"nnz_U\": %lld, ",
                      info, s_status, r_status, a_info, s_probe, s_n, r_probe, r_n, arm->nnz_l(),
                      arm->nnz_u());
        out += buf;
        out += "\"factorize_samples_us\": " + samples_json(f_samples) +
               ", \"substitution_samples_us\": " + samples_json(s_samples) +
               ", \"refactorize_samples_us\": " + samples_json(r_samples) + "}";
    }
    arm->release();

    FILE *f = std::fopen(argv[5], "wb");
    if (!f) {
        out = std::string("could not write ") + argv[5];
        return 3;
    }
    std::fwrite(x.data(), sizeof(double), m.n, f);
    std::fwrite(after.data(), sizeof(double), m.n, f);
    std::fclose(f);
    return 0;
}


/* The entry the harness calls through ctypes: run_arm's code, its text
 * (NUL-terminated, cut to cap) into buf */
extern "C" __declspec(dllexport) int direct_arms_run(int argc, const char *const *argv, char *buf,
                                                     int cap) {
    std::string out;
    int code;
    try {
        code = run_arm(argc, argv, out);
    } catch (const std::exception &e) {
        out = std::string("exception: ") + e.what();
        code = 5;
    }
    if (cap > 0) {
        size_t k = std::min(out.size(), (size_t)cap - 1);
        std::memcpy(buf, out.data(), k);
        buf[k] = 0;
    }
    return code;
}
