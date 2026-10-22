"""
compare_record.py -- the comparison harness behind the README's
performance tables: sparse_numba 0.2.0 against 0.1.11 and against direct
C++ calls of the same libraries (the full protocol with the function-only
rule and the per-arm cap, and the C++ direct-call arms).

    python tools/compare_record.py --name NAME [--build-log PATH]
        (--release-site DIR | --reference-record PATH) [--sizes n69,n207,n690,n2070]
        [--out-dir DIR] [--scratch DIR]
        [--estimate-from RECORD.json] [--no-direct] [--smoke]

writes records/compare_<NAME>_<sha>.json and .md at the committed tree
<sha> (git rev-parse --short=7 HEAD of the checkout this script sits in),
never over a file (exit 3 when any exists), and with --build-log a copy of
that log beside them as compare_<NAME>_<sha>.build.log (line ends LF, a
non-ASCII byte as '?'; the original's sha256 in the json). A dirty tree (any
line of `git status --porcelain`) is refused, exit 4, at the start and again
before the write; --smoke skips that check for a trial run and then refuses
an out-dir inside the repository.

THE VEHICLES: records/vehicles/solve_gap_0/<size>/ G.mtx with
b.mtx (the chain matrix) and S.mtx with b_s.mtx (its nodal reduction), and
vehicle.npz for keep (the indices of G's unknowns S keeps); every file's
sha256 read against SHA256SUMS before use (exit 7 on a difference).

THE ARMS, each on the same rhs every call:
- on G, every backend x ordering of sparse_numba.api (SUPERLU, UMFPACK,
  EIGEN_LU, EIGEN_LDLT x NATURAL, COLAMD, AMD, MMD_AT_PLUS_A); a pair whose
  factorize returns ORDERING_UNSUPPORTED, BAD_SHAPE or BACKEND_ABSENT is
  recorded not_applicable with the code's name;
- on S, EIGEN_LDLT x every ordering (AMD the one it offers), and SUPERLU x
  COLAMD and EIGEN_LU x COLAMD as the LU arms on S;
- THE 0.1.11 ARM on G: the installed 0.1.11 read from --release-site (the
  site-packages directory holding it, required without --reference-record; the
  files of sparse_numba-0.1.11.dist-info/RECORD, read only) copied into
  the scratch directory with each file's sha256 checked against RECORD
  (exit 6 on a difference), run in a child of this interpreter with that
  directory first on sys.path (sparse_numba.__file__ asserted inside it):
  superlu_factorize_csc / superlu_solve_factored, ColPerm NATURAL (fixed
  in 0.1.11). With --reference-record (a record of this tool at the same
  tree) the arm is not run: its samples and its x are cited from that
  record (its path and sha256 kept).

THE PROTOCOL ("full"). The njit dispatchers
are compiled first on a 3 x 3 matrix, outside every timing. Per arm:
- FUNCTION-ONLY BY RULE: the FIRST factorize call is timed; over 0.5 s the
  arm is run as a function test only -- that factorization, one solve_into,
  one refactorize and one solve_into, the checks; no timing protocol --
  status function_only, the triggering wall recorded;
- otherwise that first call is repeat 1's warm-up: 7 repeats of two
  factorize calls, the second timed (the first freed);
- substitution from a numba kernel looping solve_into on a held out (0.1.11:
  superlu_solve_factored), refactorize on the same values from a numba
  kernel: each loop compiled by two calls, then ONE TIMED PROBE CALL, then
  7 repeats of N calls, N = min(nominal, max(1, floor(2 s / probe))) -- THE
  PER-ARM CAP, a repeat's calls cut to fit 2 s from the first timed call;
  nominal 10,000 for the substitution and 100 for refactorize; the actual N
  per arm and loop recorded; each repeat's mean a sample.
Per arm the samples, median, min, max. Timer: time.perf_counter around each
Python-level call (the factorization includes one njit dispatch). Threads:
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS
set to 1 before numpy imports, and asserted. THE EXPECTED TOTAL WALL is
printed at the start and recorded: from --estimate-from (a record of this
tool; default the banked reduced -O3 record) by the rule and the cap, an
arm absent there at the cap's bound. The power plan (powercfg
/getactivescheme) is recorded.

THE C++ DIRECT-CALL ARMS (--no-direct leaves them out):
tools/cpp_direct/direct_arms.cpp, built in this run by the g++ on PATH with
the flags read from the Eigen extension's compile line in --build-log (the
compiler, -shared, -c, the source, -o and -DSN_BUILD_FLAGS dropped; without a
log, the flags compiled into the Eigen shim as cache_key() names them), the
vendored libsuperlu / libopenblas and third_party/eigen-3.4.0, and the link
options of the Eigen extension's link line, as a shared library loaded in
this process by ctypes (the package's vendored DLL directories added first;
one call per arm, its loops inside the library, so the heap, the DLLs and
the thread are the sparse_numba arms'). For every pair of the update's arms on
SUPERLU, EIGEN_LU and EIGEN_LDLT that factorized (UMFPACK has none: the
user's library), on the same CSC and rhs (passed as one binary file), the
same protocol, cap and rule -- a pair whose sparse_numba arm is
function-only is function-only here too. Its x checked against x_ref; the
compile line and the binary's sha256 recorded; per pair and size the
quotients sparse_numba / direct of the factorize, substitution and
refactorize medians.

THE CHECKS (recorded, never a verdict): every arm's x against the
0.1.11 arm's x on G (on S: its kept entries) as max |x - x_ref| /
max |x_ref| against 1e-10. The update's SUPERLU NATURAL x on G bit-equal to
the 0.1.11 arm's is asserted: otherwise exit 5 and nothing is written.

THE ENVIRONMENT: the machine, the CPU, the OS, the versions,
the compiler on PATH, cache_key() (the flags compiled into the SuperLU and
Eigen shims), the extensions' and the libraries' sha256, and -- read from
--build-log, the build's own output -- every compile and link line per
extension (the -I / -L arguments dropped), whether NDEBUG was defined, the
last -O.

Exit codes: 0 written; 2 usage; 3 a file exists; 4 dirty tree; 5 the
oracle bit-equality failed; 6 a RECORD sha256 differs; 7 a vehicle sha256
differs; 8 the 0.1.11 child failed or timed out; 9 the C++ direct-call
driver failed to build or to run.
"""

import os

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")
for _v in THREAD_VARIABLES:
    os.environ[_v] = "1"

import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VEHICLES = os.path.join(REPO, "records", "vehicles", "solve_gap_0")
SIZES = ("n69", "n207", "n690", "n2070")
RELEASE_DIST_INFO = "sparse_numba-0.1.11.dist-info"
DIRECT_SOURCE = os.path.join(HERE, "cpp_direct", "direct_arms.cpp")
ESTIMATE_FROM = os.path.join(REPO, "records", "compare_reduced_O3_ade7d09.json")

PROTOCOL = {"name": "full", "factorize_repeats": 7, "factorize_calls_per_repeat": 2,
            "substitution_calls": 10000, "substitution_repeats": 7,
            "refactorize_calls": 100, "refactorize_repeats": 7,
            "compile_calls": 2, "probe_calls": 1, "cap_s": 2.0,
            "function_only_threshold_s": 0.5, "tolerance": 1e-10}
DIRECT_BACKENDS = ("SUPERLU", "EIGEN_LU", "EIGEN_LDLT")

BACKENDS = ("SUPERLU", "UMFPACK", "EIGEN_LU", "EIGEN_LDLT")
ORDERINGS = ("NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A")
REFUSALS = ("ORDERING_UNSUPPORTED", "BAD_SHAPE", "BACKEND_ABSENT")


def fail(code, text):
    sys.stderr.write("compare_record: %s\n" % text)
    sys.exit(code)


def assert_threads():
    for v in THREAD_VARIABLES:
        assert os.environ.get(v) == "1", v


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ================================================================
# the vehicles
# ================================================================

def _mtx_lines(path):
    with open(path, "r", encoding="ascii") as f:
        head = f.readline()
        lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("%")]
    return head.split(), lines


def read_matrix(path):
    """A coordinate real general .mtx as CSC (float64, int32; entries in
    column order, rows ascending, explicit zeros kept), and n."""
    head, lines = _mtx_lines(path)
    assert head[2:5] == ["coordinate", "real", "general"], head
    n_rows, n_cols, nnz = (int(t) for t in lines[0].split())
    assert n_rows == n_cols
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    vals = np.empty(nnz, dtype=np.float64)
    for k, ln in enumerate(lines[1:1 + nnz]):
        i, j, v = ln.split()
        rows[k], cols[k], vals[k] = int(i) - 1, int(j) - 1, float(v)
    order = np.lexsort((rows, cols))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    np.add.at(indptr, cols + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)
    return (np.ascontiguousarray(vals), np.ascontiguousarray(rows.astype(np.int32)),
            np.ascontiguousarray(indptr), int(n_cols))


def read_vector(path):
    head, lines = _mtx_lines(path)
    assert head[2:5] == ["array", "real", "general"], head
    n, k = (int(t) for t in lines[0].split())
    assert k == 1
    return np.array([float(t) for t in lines[1:1 + n]], dtype=np.float64)


def matrix_facts(data, indices, indptr, n):
    """n, nnz, the pattern's symmetry (explicit entries), max |M - M^T|."""
    cols = np.repeat(np.arange(n), np.diff(indptr))
    a = {(int(i), int(j)): float(v) for i, j, v in zip(indices, cols, data)}
    pattern = all((j, i) in a for (i, j) in a)
    asym = max((abs(v - a.get((j, i), 0.0)) for (i, j), v in a.items()), default=0.0)
    return {"n": n, "nnz": int(len(data)), "pattern_symmetric": bool(pattern),
            "values_max_abs_asymmetry": asym}


def vehicle_sums():
    sums = {}
    with open(os.path.join(VEHICLES, "SHA256SUMS"), "r", encoding="ascii") as f:
        for ln in f.read().splitlines():
            if ln.strip():
                digest, name = ln.split(None, 1)
                sums[name.lstrip("*")] = digest
    return sums


def load_vehicle(size, sums):
    """The vehicle at size, every file read against SHA256SUMS."""
    files = {}
    for name in ("G.mtx", "b.mtx", "S.mtx", "b_s.mtx", "vehicle.npz"):
        rel = "%s/%s" % (size, name)
        digest = sha256_file(os.path.join(VEHICLES, size, name))
        if sums.get(rel) != digest:
            fail(7, "%s: sha256 %s, SHA256SUMS %s" % (rel, digest, sums.get(rel)))
        files[rel] = digest
    d = os.path.join(VEHICLES, size)
    g = read_matrix(os.path.join(d, "G.mtx"))
    s = read_matrix(os.path.join(d, "S.mtx"))
    b = read_vector(os.path.join(d, "b.mtx"))
    b_s = read_vector(os.path.join(d, "b_s.mtx"))
    keep = np.load(os.path.join(d, "vehicle.npz"))["keep"].astype(np.int64)
    assert len(b) == g[3] and len(b_s) == s[3] == len(keep)
    return {"G": g, "S": s, "b": b, "b_s": b_s, "keep": keep, "files_sha256": files}


# ================================================================
# timing
# ================================================================

def summary(samples):
    return {"samples": [float(x) for x in samples], "median": float(np.median(samples)),
            "min": float(np.min(samples)), "max": float(np.max(samples))}


def time_factorize(factorize, free):
    """The first call timed: over the threshold the arm is function-only
    and that handle is returned with no samples. Otherwise that call is
    repeat 1's warm-up; REPEATS x (a warm-up call freed, then the timed
    call); the last handle kept. factorize() -> (handle, info). Returns
    (handle, info, summary or None, first call us)."""
    t0 = time.perf_counter()
    h, info = factorize()
    first_us = (time.perf_counter() - t0) * 1e6
    if info != 0:
        return None, info, None, first_us
    if first_us > PROTOCOL["function_only_threshold_s"] * 1e6:
        return h, 0, None, first_us
    samples, handle = [], None
    for rep in range(PROTOCOL["factorize_repeats"]):
        if rep > 0:
            h, info = factorize()
            if info != 0:
                return None, info, None, first_us
        free(h)
        t0 = time.perf_counter()
        h, info = factorize()
        samples.append((time.perf_counter() - t0) * 1e6)
        if info != 0:
            return None, info, None, first_us
        if rep + 1 < PROTOCOL["factorize_repeats"]:
            free(h)
        else:
            handle = h
    return handle, 0, summary(samples), first_us


def capped_calls(probe_us, calls):
    """THE PER-ARM CAP: a repeat's calls cut to fit cap_s from the probe."""
    if probe_us <= 0.0:
        return calls
    return max(1, min(calls, int(PROTOCOL["cap_s"] * 1e6 // probe_us)))


def time_kernel(run, calls, repeats):
    """run(calls) -> status; compiled by a two-call run, one timed probe
    call, then repeats x N calls (N capped); us per call. Returns (status,
    summary, the loop's facts)."""
    status = run(PROTOCOL["compile_calls"])
    t0 = time.perf_counter()
    status |= run(PROTOCOL["probe_calls"])
    probe_us = (time.perf_counter() - t0) * 1e6
    n = capped_calls(probe_us, calls)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        status |= run(n)
        samples.append((time.perf_counter() - t0) / n * 1e6)
    return int(status), summary(samples), {"nominal_calls": calls, "calls_actual": n,
                                          "probe_us": probe_us, "capped": n < calls}


# ================================================================
# the update's arms (sparse_numba.api of this checkout)
# ================================================================

def solver_text(backend, ordering, versions):
    if backend == "SUPERLU":
        return ("SuperLU %s dgstrf / dgstrs (the vendored libsuperlu-7.dll)" % versions["superlu"],
                "options.ColPerm = %s" % ordering)
    if backend == "UMFPACK":
        pairing = {"NATURAL": ("UMFPACK_ORDERING_NONE", "UMFPACK_STRATEGY_UNSYMMETRIC"),
                   "COLAMD": ("UMFPACK_ORDERING_AMD", "UMFPACK_STRATEGY_UNSYMMETRIC (COLAMD)"),
                   "AMD": ("UMFPACK_ORDERING_AMD", "UMFPACK_STRATEGY_SYMMETRIC")}.get(ordering)
        param = ("Control[UMFPACK_ORDERING] = %s, Control[UMFPACK_STRATEGY] = %s, "
                 "Control[UMFPACK_PIVOT_TOLERANCE] = 1.0" % pairing) if pairing else "--"
        return ("UMFPACK %s umfpack_di_symbolic / numeric / solve(UMFPACK_A)" % versions["umfpack"],
                param)
    eigen_ordering = {"NATURAL": "Eigen::NaturalOrdering<int>", "COLAMD": "Eigen::COLAMDOrdering<int>",
                      "AMD": "Eigen::AMDOrdering<int>"}.get(ordering, "--")
    if backend == "EIGEN_LU":
        return ("Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, %s>"
                % eigen_ordering, "%s, setPivotThreshold(1.0)" % eigen_ordering)
    return ("Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, "
            "Eigen::Lower, Eigen::AMDOrdering<int>>",
            "Eigen::AMDOrdering<int>" if ordering == "AMD" else "--")


def make_kernels():
    from numba import njit
    from sparse_numba import api

    @njit(nogil=True)
    def substitution(handle, b, out, calls):
        status = 0
        for _ in range(calls):
            status |= api.solve_into(handle, b, out)
        return status

    @njit(nogil=True)
    def refactorization(handle, data, calls):
        status = 0
        for _ in range(calls):
            status |= api.refactorize(handle, data)
        return status

    return substitution, refactorization


def warm_dispatch(api, kernels):
    """Every njit entry the arms call compiled (or loaded from the cache) on
    a 3 x 3 matrix before any timing, so no first factorize call carries a
    compile (the function-only rule times it)."""
    data = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
    indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 5, 7], dtype=np.int32)
    b, out = np.ones(3), np.zeros(3)
    h, info = api.factorize(data, indices, indptr, 3, api.SUPERLU, api.NATURAL)
    assert info == 0, info
    status = int(api.solve_into(h, b, out)) | int(api.refactorize(h, data))
    substitution, refactorization = kernels
    status |= int(substitution(h, b, out, 1)) | int(refactorization(h, data, 1))
    api.handle_info(h)
    status |= int(api.free(h))
    assert status == 0, status


def run_arm(api, kernels, names, matrix_name, matrix, b, backend, ordering, versions):
    data, indices, indptr, n = matrix
    solver, param = solver_text(backend, ordering, versions)
    arm = {"id": "%s.%s.%s" % (backend, ordering, matrix_name), "matrix": matrix_name,
           "backend": backend, "ordering": ordering, "solver": solver,
           "ordering_parameter": param}
    bid, oid = getattr(api, backend), getattr(api, ordering)

    def factorize():
        return api.factorize(data, indices, indptr, n, bid, oid)

    handle, info, factor, first_us = time_factorize(factorize, api.free)
    arm["first_factorize_us"] = first_us
    if handle is None:
        name = names.get(int(info), str(info))
        arm.update({"status": "not_applicable" if name in REFUSALS else "error",
                    "code": int(info), "code_name": name})
        return arm, None
    out = np.zeros(n)
    if factor is None:
        # FUNCTION-ONLY: that factorization, one solve, one refactorize and
        # one solve; no timing protocol
        solve_info = int(api.solve_into(handle, b, out))
        x = out.copy()
        t0 = time.perf_counter()
        r_info = int(api.refactorize(handle, data))
        refac_single = (time.perf_counter() - t0) * 1e6
        after = np.zeros(n)
        after_info = int(api.solve_into(handle, b, after))
        arm.update({"status": "function_only",
                    "function_only_trigger": "the first factorize call took %.1f us > %g s"
                                             % (first_us, PROTOCOL["function_only_threshold_s"]),
                    "solve_info": solve_info, "refactorize_info": r_info,
                    "refactorize_single_us": refac_single})
    else:
        substitution, refactorization = kernels
        status, subst, s_loop = time_kernel(lambda calls: substitution(handle, b, out, calls),
                                            PROTOCOL["substitution_calls"],
                                            PROTOCOL["substitution_repeats"])
        x = out.copy()
        r_status, refac, r_loop = time_kernel(lambda calls: refactorization(handle, data, calls),
                                              PROTOCOL["refactorize_calls"],
                                              PROTOCOL["refactorize_repeats"])
        after = np.zeros(n)
        after_info = int(api.solve_into(handle, b, after))
        arm.update({"status": "measured", "factorize_us": factor, "substitution_us": subst,
                    "refactorize_us": refac, "substitution_loop": s_loop, "refactorize_loop": r_loop,
                    "substitution_status": status, "refactorize_status": r_status})
    info_t = api.handle_info(handle)
    free_info = int(api.free(handle))
    arm.update({
        "x_after_refactorize_max_rel_diff": rel_diff(after, x), "solve_after_refactorize_info": after_info,
        "handle_info": {"n": int(info_t[0]), "nnz_L": int(info_t[1]), "nnz_U": int(info_t[2]),
                        "backend_id": int(info_t[3]), "ordering_id": int(info_t[4]),
                        "last_status": int(info_t[5]), "pivot_tolerance": float(info_t[6]),
                        "info": int(info_t[7])},
        "free_info": free_info})
    return arm, x


# ================================================================
# the C++ direct-call arms (tools/cpp_direct/direct_arms.cpp)
# ================================================================

def split_line(line):
    """A logged command line split as the shell would (shlex first, so a
    quoted argument holding escaped quotes stays one token), then each
    token's backslashes written as slashes. An unquoted backslash is an
    escape to shlex, so an unquoted Windows path loses its separators; only
    the flags are read from these tokens, never a path."""
    return [t.replace("\\", "/") for t in shlex.split(line)]


def direct_flags(env):
    """The compile flags of the Eigen extension as built (the build log's
    compile line of eigen_wrapper.cpp), else the flags compiled into the
    Eigen shim (cache_key()); and the link options of its link line."""
    bl = env.get("build_log")
    if bl:
        entry = bl["extensions"]["cy_eigen_wrapper"]
        lines = [c["line"] for c in entry["compile"] if c["source"].endswith("eigen_wrapper.cpp")
                 and not c["source"].endswith("cy_eigen_wrapper.cpp")]
        if lines:
            tokens = split_line(lines[-1])[1:]
            flags, skip = [], False
            for t in tokens:
                if skip:
                    skip = False
                elif t in ("-c", "-o"):
                    skip = True
                elif t != "-shared" and not t.startswith("-DSN_BUILD_FLAGS"):
                    flags.append(t)
            link = []
            for ln in entry["link"]:
                link = [t for t in split_line(ln)
                        if t == "-s" or t.startswith("-static") or t.startswith("-Wl,")]
            return flags, link, "the build log's compile line of sparse_numba/sparse_eigen/eigen_wrapper.cpp"
    text = (env["flags_compiled_in"]["eigen_shim"] or "").split("flags=", 1)[-1]
    link = ["-static-libstdc++", "-static-libgcc"] if platform.system() == "Windows" else []
    return shlex.split(text), link, "cache_key()'s eigen= flags (no build log given)"


def build_direct(env, scratch):
    flags, link, source = direct_flags(env)
    exe = os.path.join(scratch, "direct_arms" + (".dll" if platform.system() == "Windows" else ".so"))
    cmd = (["g++", "-shared"] + flags + ["-I" + os.path.join(REPO, "third_party", "eigen-3.4.0"),
                              "-I" + os.path.join(REPO, "vendor", "superlu", "include"),
                              DIRECT_SOURCE, "-o", exe,
                              "-L" + os.path.join(REPO, "vendor", "superlu", "lib"),
                              "-L" + os.path.join(REPO, "vendor", "openblas", "lib"),
                              "-lsuperlu", "-lopenblas"] + link)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        fail(9, "the direct-call driver did not build (exit %d)" % proc.returncode)
    package = os.path.dirname(os.path.abspath(sys.modules["sparse_numba"].__file__))
    dll_dirs = [os.path.join(package, "vendor", sub, "bin") for sub in ("superlu", "openblas")]
    import ctypes
    for d in dll_dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)
    lib = ctypes.CDLL(exe)
    lib.direct_arms_run.restype = ctypes.c_int
    lib.direct_arms_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p,
                                    ctypes.c_int]
    return {"_lib": lib, "source": os.path.relpath(DIRECT_SOURCE, REPO).replace("\\", "/"),
            "source_sha256": sha256_file(DIRECT_SOURCE), "flags": flags, "link_options": link,
            "flags_from": source,
            "compile_line": " ".join(shlex.quote(t.replace("\\", "/")) for t in cmd),
            "compiler": first_line(["g++", "--version"]), "build_wall_s": wall,
            "compiler_stderr": proc.stderr[-2000:], "binary": exe.replace("\\", "/"),
            "binary_sha256": sha256_file(exe),
            "loaded_in_process_after": [os.path.relpath(d, REPO).replace("\\", "/") for d in dll_dirs]}


def run_direct(direct, scratch, size, matrix_name, matrix, b, backend, ordering, function_only):
    data, indices, indptr, n = matrix
    stem = os.path.join(scratch, "direct_%s_%s_%s_%s" % (size, matrix_name, backend, ordering))
    with open(stem + ".bin", "wb") as f:
        np.array([n, len(data)], dtype="<i4").tofile(f)
        np.ascontiguousarray(indptr, dtype="<i4").tofile(f)
        np.ascontiguousarray(indices, dtype="<i4").tofile(f)
        np.ascontiguousarray(data, dtype="<f8").tofile(f)
        np.ascontiguousarray(b, dtype="<f8").tofile(f)
    p = PROTOCOL
    import ctypes
    args = ["direct_arms", stem + ".bin", backend, ordering, "function" if function_only else "auto",
            stem + ".x.bin", str(p["factorize_repeats"]), str(p["substitution_calls"]),
            str(p["substitution_repeats"]), str(p["refactorize_calls"]), str(p["refactorize_repeats"]),
            repr(p["cap_s"]), repr(p["function_only_threshold_s"])]
    argv = (ctypes.c_char_p * len(args))(*[a.encode("utf-8") for a in args])
    buf = ctypes.create_string_buffer(1 << 20)
    code = direct["_lib"].direct_arms_run(len(args), argv, buf, len(buf))
    text = buf.value.decode("ascii", errors="replace")
    if code != 0:
        fail(9, "the direct arm %s %s %s returned %d: %s" % (size, backend, ordering, code, text))
    res = json.loads(text)
    arm = {"id": "DIRECT.%s.%s.%s" % (backend, ordering, matrix_name), "matrix": matrix_name,
           "backend": backend, "ordering": ordering, "solver": res["solver"],
           "status": res["status"], "first_factorize_us": res["first_factorize_us"]}
    if res["status"] == "error":
        arm["factorize_info"] = res["factorize_info"]
        return arm, None
    xx = np.fromfile(stem + ".x.bin", dtype="<f8")
    x, after = xx[:n].copy(), xx[n:].copy()
    if res["status"] == "function_only":
        arm.update({k: res[k] for k in ("function_only_reason", "solve_info", "refactorize_info",
                                        "refactorize_single_us", "solve_after_refactorize_info",
                                        "nnz_L", "nnz_U")})
    else:
        arm.update({
            "factorize_us": summary(res["factorize_samples_us"]),
            "substitution_us": summary(res["substitution_samples_us"]),
            "refactorize_us": summary(res["refactorize_samples_us"]),
            "substitution_loop": {"nominal_calls": p["substitution_calls"],
                                  "calls_actual": res["substitution_calls_actual"],
                                  "probe_us": res["substitution_probe_us"],
                                  "capped": res["substitution_calls_actual"] < p["substitution_calls"]},
            "refactorize_loop": {"nominal_calls": p["refactorize_calls"],
                                 "calls_actual": res["refactorize_calls_actual"],
                                 "probe_us": res["refactorize_probe_us"],
                                 "capped": res["refactorize_calls_actual"] < p["refactorize_calls"]},
            "factorize_info": res["factorize_info"], "substitution_status": res["substitution_status"],
            "refactorize_status": res["refactorize_status"],
            "solve_after_refactorize_info": res["solve_after_refactorize_info"],
            "nnz_L": res["nnz_L"], "nnz_U": res["nnz_U"]})
    arm["x_after_refactorize_max_rel_diff"] = rel_diff(after, x)
    return arm, x


def overhead(arms, direct_arms):
    """Per pair: the sparse_numba median / the direct median of the
    factorize, substitution and refactorize (both measured), else the
    status of each side."""
    by = {a["id"]: a for a in arms}
    out = []
    for d in direct_arms:
        a = by.get(d["id"][len("DIRECT."):])
        row = {"pair": d["id"][len("DIRECT."):], "sparse_numba_status": a["status"] if a else None,
               "direct_status": d["status"]}
        for what in ("factorize_us", "substitution_us", "refactorize_us"):
            ok = a and a["status"] == "measured" and d["status"] == "measured"
            row[what.replace("_us", "")] = (a[what]["median"] / d[what]["median"]) if ok else None
        out.append(row)
    return out


# ================================================================
# the expected total wall
# ================================================================

def arm_wall_s(f_us, s_us, r_us, refactorize=True):
    """The wall of one arm by the rule and the cap from medians (us)."""
    p = PROTOCOL
    if f_us > p["function_only_threshold_s"] * 1e6:
        return (f_us + 2 * s_us + (r_us if refactorize else 0.0)) / 1e6
    wall = p["factorize_calls_per_repeat"] * p["factorize_repeats"] * f_us
    for us, calls, reps, on in ((s_us, p["substitution_calls"], p["substitution_repeats"], True),
                                (r_us, p["refactorize_calls"], p["refactorize_repeats"], refactorize)):
        if on:
            wall += (p["compile_calls"] + p["probe_calls"]) * us + reps * capped_calls(us, calls) * us
    return wall / 1e6


def expected_wall(prior_path, sizes, direct):
    """The expected total wall (s) from a prior record's medians; an arm
    absent there at the cap's bound; a fixed 120 s for the imports, the
    vehicles, the 0.1.11 child's start and the driver's build."""
    p = PROTOCOL
    bound = (p["factorize_calls_per_repeat"] * p["factorize_repeats"] * p["function_only_threshold_s"]
             + (p["substitution_repeats"] + p["refactorize_repeats"]) * p["cap_s"])
    prior = None
    if prior_path and os.path.exists(prior_path):
        with open(prior_path, "r", encoding="ascii") as f:
            prior = json.load(f)
    total, per_size = 120.0, {}
    for size in sizes:
        arms = (prior or {}).get("per_size", {}).get(size, {}).get("arms", [])
        wall = 0.0
        for a in arms:
            if a["status"] not in ("measured", "cited", "function_only"):
                continue
            if a["status"] == "function_only":
                w = (2 * a["first_factorize_us"] + a.get("refactorize_single_us", 0.0)) / 1e6
            else:
                f = a["factorize_us"]["median"]
                s = a["substitution_us"]["median"]
                r = (a.get("refactorize_us") or {}).get("median", 0.0)
                w = arm_wall_s(f, s, r, refactorize=a.get("refactorize_us") is not None)
            wall += w
            if direct and a["backend"] in DIRECT_BACKENDS:
                wall += w  # the direct arm, taken at the sparse_numba arm's wall
        if not arms:
            wall = 14 * bound * (2 if direct else 1)
        per_size[size] = wall
        total += wall
    return {"expected_total_wall_s": total, "per_size_s": per_size,
            "from": os.path.relpath(prior_path, REPO).replace("\\", "/") if prior else None,
            "target_s": 2400.0,
            "method": "per arm by the function-only rule and the per-arm cap from the medians of "
                      "the record named (the factorization the first call's estimate); each "
                      "direct arm taken at its sparse_numba arm's wall; an arm absent there at "
                      "the cap's bound; plus 120 s fixed"}


def rel_diff(x, ref):
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def bits_equal(x, y):
    x, y = np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)
    return x.shape == y.shape and bool(np.array_equal(x.view(np.int64), y.view(np.int64)))


# ================================================================
# the 0.1.11 arm (a child with the installed copy first on sys.path)
# ================================================================

def copy_release(site, dest):
    """The files RECORD lists with a hash, copied from site into dest, each
    read back against RECORD. Returns the per-file check."""
    record = os.path.join(site, RELEASE_DIST_INFO, "RECORD")
    rows = []
    with open(record, "r", encoding="utf-8", newline="") as f:
        for ln in f.read().splitlines():
            path, digest, size = ln.rsplit(",", 2)
            if digest:
                rows.append((path, digest, int(size)))
    checks = []
    for path, digest, size in rows:
        algo, value = digest.split("=", 1)
        assert algo == "sha256", digest
        target = os.path.join(dest, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(site, *path.split("/")), target)
        with open(target, "rb") as f:
            raw = f.read()
        got = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode("ascii")
        checks.append({"path": path, "record_sha256": value, "copy_sha256": got,
                       "size": len(raw), "equal": got == value and len(raw) == size})
    bad = [c["path"] for c in checks if not c["equal"]]
    if bad:
        fail(6, "RECORD sha256 differs: %s" % ", ".join(bad))
    return {"record": record.replace("\\", "/"), "record_sha256": sha256_file(record),
            "files": len(checks), "equal": len(checks) - len(bad), "checks": checks}


def child_0111(copy_dir, out_dir, sizes):
    """In the child: the 0.1.11 arm on each size's G; x to out_dir."""
    assert_threads()
    sys.path.insert(0, copy_dir)
    # 0.1.11's __init__ finds its DLLs under sys.prefix/Lib/site-packages/
    # sparse_numba/vendor/*/bin (its package-relative path misses them); the
    # child's prefix is not the install's, so the same two directories of the
    # copy are added here, as 0.1.11 adds them where it is installed
    for sub in ("superlu", "openblas"):
        dll_dir = os.path.join(copy_dir, "sparse_numba", "vendor", sub, "bin")
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
    import sparse_numba
    from numba import njit
    package = os.path.dirname(os.path.abspath(sparse_numba.__file__))
    assert os.path.normcase(package).startswith(os.path.normcase(os.path.abspath(copy_dir))), package
    from sparse_numba.sparse_superlu import superlu_numba_interface as sn
    from importlib import metadata
    dist = [d for d in metadata.distributions(path=[copy_dir]) if d.metadata["Name"] == "sparse_numba"]
    version = dist[0].version if dist else "unknown"

    @njit(nogil=True)
    def substitution(handle, b, calls):
        status = 0
        x = np.zeros(1)
        for _ in range(calls):
            x, info = sn.superlu_solve_factored(handle, b)
            status |= info
        return status, x

    result = {"package_file": sparse_numba.__file__.replace("\\", "/"), "version": version,
              "sizes": {}}
    # the dispatchers compiled on a 3 x 3 matrix before any timing
    w_data = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
    w_indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    w_indptr = np.array([0, 2, 5, 7], dtype=np.int32)
    h, info = sn.superlu_factorize_csc(w_data, w_indices, w_indptr)
    assert info == 0, info
    substitution(h, np.ones(3), 1)
    sn.superlu_free_factors(h)
    for size in sizes:
        d = os.path.join(VEHICLES, size)
        data, indices, indptr, n = read_matrix(os.path.join(d, "G.mtx"))
        b = read_vector(os.path.join(d, "b.mtx"))

        def factorize():
            return sn.superlu_factorize_csc(data, indices, indptr)

        handle, info, factor, first_us = time_factorize(factorize, sn.superlu_free_factors)
        assert handle is not None, info
        held = {}

        def run(calls):
            status, x = substitution(handle, b, calls)
            held["x"] = x
            return status

        if factor is None:
            # function-only by rule: one solve
            status, subst, loop = run(1), None, None
        else:
            status, subst, loop = time_kernel(run, PROTOCOL["substitution_calls"],
                                              PROTOCOL["substitution_repeats"])
        np.save(os.path.join(out_dir, "x_0111_%s.npy" % size), held["x"])
        sn.superlu_free_factors(handle)
        result["sizes"][size] = {"factorize_us": factor, "substitution_us": subst,
                                 "substitution_loop": loop, "first_factorize_us": first_us,
                                 "substitution_status": int(status)}
    with open(os.path.join(out_dir, "child_0111.json"), "w", encoding="ascii") as f:
        json.dump(result, f)


def arm_0111(versions, size, measured, cited_from=None):
    return {"id": "SPARSE_NUMBA_0111.NATURAL.G", "matrix": "G", "backend": "sparse_numba 0.1.11",
            "ordering": "NATURAL",
            "solver": "sparse_numba 0.1.11 superlu_factorize_csc / superlu_solve_factored "
                      "(SuperLU dgstrf / dgstrs, the 0.1.11 wheel's libsuperlu-7.dll)",
            "ordering_parameter": "options.ColPerm = NATURAL (fixed in 0.1.11)",
            "status": ("cited" if cited_from is not None else
                       "function_only" if measured["factorize_us"] is None else "measured"),
            "cited_from": cited_from, "factorize_us": measured["factorize_us"],
            "first_factorize_us": measured.get("first_factorize_us"),
            "substitution_us": measured["substitution_us"],
            "substitution_loop": measured.get("substitution_loop"), "refactorize_us": None,
            "refactorize": "not_applicable (0.1.11 has no refactorize)",
            "substitution_status": measured["substitution_status"]}


# ================================================================
# the environment
# ================================================================

def cpu_name():
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        with open("/proc/cpuinfo", "r") as f:
            for ln in f:
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def first_line(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return out.stdout.splitlines()[0].strip() if out.stdout else "absent"
    except (OSError, subprocess.SubprocessError):
        return "absent"


def loaded_library_path(name):
    """The file a loaded DLL came from (Windows), else ''."""
    if platform.system() != "Windows":
        return ""
    import ctypes
    k = ctypes.WinDLL("kernel32")
    k.GetModuleHandleW.restype = ctypes.c_void_p
    h = k.GetModuleHandleW(name)
    if not h:
        return ""
    buf = ctypes.create_unicode_buffer(1024)
    k.GetModuleFileNameW(ctypes.c_void_p(h), buf, 1024)
    return buf.value


EXTENSION_SOURCES = {
    "cy_superlu_wrapper": "sparse_numba/sparse_superlu/",
    "cy_umfpack_wrapper": "sparse_numba/sparse_umfpack/",
    "cy_eigen_wrapper": "sparse_numba/sparse_eigen/",
}


def read_build_log(path):
    """Every compile (-c) and link line of the log per extension, the -I /
    -L arguments dropped; NDEBUG defined or not; the last -O."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out = {name: {"compile": [], "link": []} for name in EXTENSION_SOURCES}
    for ln in lines:
        if not re.match(r"^(gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(\.exe)?\s", ln):
            continue
        text = re.sub(r'\s-[IL]("[^"]*"|\S+)', "", ln)
        m = re.search(r"\s-c\s+(\S+)", ln)
        if m:
            source = m.group(1).replace("\\", "/")
            for name, prefix in EXTENSION_SOURCES.items():
                if source.startswith(prefix):
                    tokens = split_line(text)
                    opt = [t for t in tokens if re.match(r"^-O\w*$", t)]
                    out[name]["compile"].append({
                        "source": source, "line": text,
                        "ndebug_defined": any(t == "-DNDEBUG" or t.startswith("-DNDEBUG=")
                                              for t in tokens),
                        "optimization_last": opt[-1] if opt else None})
        else:
            for name in EXTENSION_SOURCES:
                if name in ln and "-shared" in ln:
                    out[name]["link"].append(text)
    for name, entry in out.items():
        entry["ndebug_defined_any"] = any(c["ndebug_defined"] for c in entry["compile"])
    return {"path": os.path.abspath(path).replace("\\", "/"), "sha256": sha256_file(path),
            "extensions": out}


def environment(build_log):
    import numba
    import llvmlite
    import sparse_numba
    from importlib import metadata
    from sparse_numba import _symbols
    key = sparse_numba.cache_key()
    parts = dict(p.split("=", 1) for p in key.split(";"))
    code, umf_version, umf_source = _symbols.umfpack_library()
    umf_path = loaded_library_path(umf_source) if code == 0 else ""
    package = os.path.dirname(os.path.abspath(sparse_numba.__file__))
    extensions = {}
    for module in _symbols._EXTENSIONS:
        spec = importlib.util.find_spec(module)
        extensions[module] = {"file": os.path.relpath(spec.origin, REPO).replace("\\", "/"),
                              "sha256": sha256_file(spec.origin)}
    dlls = {}
    for sub in ("vendor/superlu/bin", "vendor/openblas/bin"):
        d = os.path.join(package, *sub.split("/"))
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".dll"):
                    dlls["sparse_numba/%s/%s" % (sub, f)] = sha256_file(os.path.join(d, f))
    with open(os.path.join(REPO, "third_party", "eigen-3.4.0", "SHA256SUMS"), "r",
              encoding="ascii") as f:
        eigen_head = f.readline().strip()

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "absent"

    return {
        "machine": platform.platform(), "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "executable": sys.executable.replace("\\", "/"),
        "numpy": np.__version__, "numba": numba.__version__, "llvmlite": llvmlite.__version__,
        "cython": version("Cython"), "setuptools": version("setuptools"),
        "compiler_on_path": {"gcc": first_line(["gcc", "--version"]),
                             "g++": first_line(["g++", "--version"])},
        "threads": {v: os.environ.get(v) for v in THREAD_VARIABLES},
        "sparse_numba_file": os.path.relpath(sparse_numba.__file__, REPO).replace("\\", "/"),
        "cache_key": key,
        "flags_compiled_in": {"superlu_shim": parts.get("superlu"), "eigen_shim": parts.get("eigen"),
                              "umfpack_shim": "not exposed by the shim (cache_key()'s umfpack= is "
                                              "the user's library version); see build_log"},
        "superlu_version": parts.get("superlu", "").split(",")[0],
        "umfpack": {"code": int(code), "version": umf_version, "source": umf_source,
                    "loaded_from": umf_path.replace("\\", "/"),
                    "sha256": sha256_file(umf_path) if umf_path else None},
        "eigen_version": parts.get("eigen", "").split(",")[0],
        "eigen_sha256sums_head": eigen_head,
        "extensions": extensions, "vendored_dlls_sha256": dlls,
        "prebuilt_libraries": "libsuperlu-7.dll, libopenblas.dll and the user's SuiteSparse keep "
                              "their own build flags (they are prebuilt, not rebuilt here); only the "
                              "shims are compiled by setup.py",
        "extra_cflags_env": os.environ.get("SPARSE_NUMBA_EXTRA_CFLAGS"),
        "debug_env": os.environ.get("SPARSE_NUMBA_DEBUG"),
        "power_plan": first_line(["powercfg", "/getactivescheme"]) if platform.system() == "Windows"
                      else "not read (not Windows)",
        "build_log": read_build_log(build_log) if build_log else None,
    }


# ================================================================
# the record
# ================================================================

def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True,
                          check=True).stdout


def ratios(arms, ref_arm):
    """THE RATIOS on the substitution medians (and on the
    factorization medians where named)."""
    by = {a["id"]: a for a in arms if a["status"] == "measured"}

    def med(arm_id, what="substitution_us"):
        a = by.get(arm_id)
        return a[what]["median"] if a else None

    def ratio(num, den):
        return num / den if num is not None and den else None

    out = []
    ref = ref_arm["substitution_us"]["median"] if ref_arm["substitution_us"] else None
    for a in arms:
        if a["status"] == "measured" and a["matrix"] == "G" and a["id"] != ref_arm["id"]:
            out.append({"family": "against_0111", "num": a["id"], "den": ref_arm["id"],
                        "substitution": ratio(med(a["id"]), ref)})
    for be in BACKENDS:
        base = "%s.NATURAL.G" % be
        for o in ORDERINGS[1:]:
            arm_id = "%s.%s.G" % (be, o)
            if arm_id in by and base in by:
                out.append({"family": "ordering_against_natural", "num": arm_id, "den": base,
                            "substitution": ratio(med(arm_id), med(base)),
                            "factorize": ratio(med(arm_id, "factorize_us"), med(base, "factorize_us"))})
    for o in ORDERINGS:
        num, den = "EIGEN_LU.%s.G" % o, "SUPERLU.%s.G" % o
        if num in by and den in by:
            out.append({"family": "eigen_lu_against_superlu", "num": num, "den": den,
                        "substitution": ratio(med(num), med(den)),
                        "factorize": ratio(med(num, "factorize_us"), med(den, "factorize_us"))})
    for den in ("SUPERLU.COLAMD.S", "EIGEN_LU.COLAMD.S"):
        num = "EIGEN_LDLT.AMD.S"
        if num in by and den in by:
            out.append({"family": "ldlt_on_s_against_lu_on_s", "num": num, "den": den,
                        "substitution": ratio(med(num), med(den)),
                        "factorize": ratio(med(num, "factorize_us"), med(den, "factorize_us"))})
    return out


def fmt(v, digits=3):
    return "--" if v is None else ("%.*f" % (digits, v))


def fmt_us(s, digits=1):
    return "%s (%s / %s)" % (fmt(s["median"], digits), fmt(s["min"], digits), fmt(s["max"], digits))


def loop_n(loop):
    return "--" if not loop else "{:,}".format(loop["calls_actual"])


def write_md(path, rec):
    env = rec["environment"]
    p = rec["protocol"]
    w = []
    w.append("# compare_%s (%s, %s)" % (rec["name"], rec["git_sha"], rec["written"]))
    w.append("")
    w.append("Produced by tools/compare_record.py (the method in its docstring) at the committed "
             "tree %s; the data in the json beside this file. Protocol: %s -- an arm whose first "
             "factorize call takes over %g s is FUNCTION-ONLY (one factorize, one solve_into, one "
             "refactorize and one solve_into, the checks; no timing); otherwise the factorization "
             "the second of %d factorize calls in each of %d repeats; substitution %s calls a repeat "
             "x %d repeats from a numba kernel on a held out; refactorize %d calls x %d repeats; THE "
             "PER-ARM CAP: a repeat's calls cut to fit %g s from one timed probe call (the actual N "
             "in the tables); per arm the median (min / max; the samples in the json), in us. The "
             "same rhs every call. No verdict word; the ratios are quotients of medians."
             % (rec["git_sha"], p["name"], p["function_only_threshold_s"],
                p["factorize_calls_per_repeat"], p["factorize_repeats"],
                "{:,}".format(p["substitution_calls"]), p["substitution_repeats"],
                p["refactorize_calls"], p["refactorize_repeats"], p["cap_s"]))
    if rec.get("label"):
        w.append("")
        w.append("LABEL: %s" % rec["label"])
    est = rec["expected_wall"]
    w.append("")
    w.append("THE WALL: expected %.0f s (%.1f m; target %.0f m; from %s), actual %.0f s (%.1f m)."
             % (est["expected_total_wall_s"], est["expected_total_wall_s"] / 60, est["target_s"] / 60,
                est["from"] or "the cap's bound", rec["wall_s"], rec["wall_s"] / 60))
    w.append("")
    w.append("## The environment")
    w.append("")
    w.append("- machine: %s; cpu: %s; logical cpus %s" % (env["machine"], env["cpu"], env["logical_cpus"]))
    w.append("- power plan: %s" % env.get("power_plan"))
    w.append("- Python %s, numpy %s, numba %s, llvmlite %s, Cython %s; threads: %s"
             % (env["python"], env["numpy"], env["numba"], env["llvmlite"], env["cython"],
                ", ".join("%s=%s" % kv for kv in env["threads"].items())))
    w.append("- compiler on PATH: %s" % env["compiler_on_path"]["gcc"])
    w.append("- SPARSE_NUMBA_EXTRA_CFLAGS at this run: %s; SPARSE_NUMBA_DEBUG at this run: %s"
             % (env["extra_cflags_env"] or "unset", env.get("debug_env") or "unset"))
    w.append("- cache_key(): `%s`" % env["cache_key"])
    w.append("- SuperLU %s (vendored DLL, its own flags); UMFPACK %s loaded from %s (sha256 %s; the "
             "user's library, its own flags); Eigen %s (headers compiled in; %s)"
             % (env["superlu_version"], env["umfpack"]["version"], env["umfpack"]["loaded_from"] or "--",
                (env["umfpack"]["sha256"] or "--")[:16], env["eigen_version"], env["eigen_sha256sums_head"]))
    for name, digest in env["vendored_dlls_sha256"].items():
        if "superlu-7" in name or "openblas" in name:
            w.append("- %s sha256 %s" % (name, digest))
    for module, e in env["extensions"].items():
        w.append("- %s sha256 %s" % (e["file"], e["sha256"]))
    bl = env["build_log"]
    if bl:
        w.append("- THE COMPILE FLAGS AS BUILT, read from the build's own output %s (sha256 %s; "
                 "banked beside this file as compare_%s_%s.build.log), the -I / -L arguments dropped:"
                 % (bl["path"], bl["sha256"][:16], rec["name"], rec["git_sha"]))
        for name, entry in bl["extensions"].items():
            for c in entry["compile"]:
                w.append("  - %s: `%s` (NDEBUG defined: %s; last -O: %s)"
                         % (c["source"], c["line"], c["ndebug_defined"], c["optimization_last"]))
            for ln in entry["link"]:
                w.append("  - %s link: `%s`" % (name, ln))
    else:
        w.append("- no build log given: the flags are those compiled in (cache_key() above)")
    rel = rec["release_0111"]
    if rel.get("copy"):
        w.append("- THE 0.1.11 ARM: %d files of %s copied, %d sha256 equal to RECORD; run from %s "
                 "(version %s) in a child of %s"
                 % (rel["copy"]["files"], rel["copy"]["record"], rel["copy"]["equal"],
                    rel["package_file"], rel["version"], env["executable"]))
    else:
        w.append("- THE 0.1.11 ARM: cited from %s (sha256 %s), not re-run"
                 % (rel["cited_from"]["path"], rel["cited_from"]["sha256"][:16]))
    d = rec.get("direct")
    if d:
        w.append("- THE C++ DIRECT-CALL ARMS: %s (sha256 %s) built by %s with the flags of %s, as a "
                 "shared library (sha256 %s) loaded in the harness's process by ctypes after %s. "
                 "The compile line: `%s`"
                 % (d["source"], d["source_sha256"][:16], d["compiler"], d["flags_from"],
                    d["binary_sha256"], ", ".join(d["loaded_in_process_after"]), d["compile_line"]))
    else:
        w.append("- THE C++ DIRECT-CALL ARMS: not run (--no-direct)")
    for size in rec["sizes"]:
        sz = rec["per_size"][size]
        w.append("")
        w.append("## %s" % size)
        w.append("")
        g, s = sz["facts"]["G"], sz["facts"]["S"]
        w.append("THE MATRICES: G n %d, nnz %d, pattern symmetric %s, max |G - G^T| %g; S n %d, "
                 "nnz %d, pattern symmetric %s, max |S - S^T| %g."
                 % (g["n"], g["nnz"], g["pattern_symmetric"], g["values_max_abs_asymmetry"],
                    s["n"], s["nnz"], s["pattern_symmetric"], s["values_max_abs_asymmetry"]))
        w.append("")
        w.append("| arm | matrix | solver | ordering parameter | factor us (min / max) | "
                 "substitution us (median) | min / max | N subst | refactorize us (min / max) | "
                 "N refac | check |")
        w.append("|---|---|---|---|---|---|---|---|---|---|---|")
        fo = []
        for a in sz["arms"] + sz.get("direct_arms", []):
            chk = a.get("check")
            chk_text = ("%.2e" % chk["max_rel_diff"]) if chk else "reference"
            if a["status"] in ("measured", "cited"):
                f, su, r = a["factorize_us"], a["substitution_us"], a.get("refactorize_us")
                w.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
                    a["id"] + (" (cited)" if a["status"] == "cited" else ""), a["matrix"],
                    a["solver"], a.get("ordering_parameter", "--"), fmt_us(f), fmt(su["median"]),
                    fmt(su["min"]), fmt(su["max"]), loop_n(a.get("substitution_loop")),
                    fmt_us(r) if r else "--", loop_n(a.get("refactorize_loop")), chk_text))
            elif a["status"] == "function_only":
                w.append("| %s | %s | %s | %s | function-only (first call %s) | -- | -- | -- | -- | "
                         "-- | %s |" % (a["id"], a["matrix"], a["solver"], a.get("ordering_parameter", "--"),
                                        fmt(a["first_factorize_us"], 1), chk_text))
                fo.append("%s (first factorize call %s us)" % (a["id"], fmt(a["first_factorize_us"], 1)))
        na = ["%s (%s)" % (a["id"], a.get("code_name", a.get("factorize_info"))) for a in
              sz["arms"] + sz.get("direct_arms", []) if a["status"] in ("not_applicable", "error")]
        w.append("")
        w.append("FUNCTION-ONLY (the first factorize call over %g s, or the pair's sparse_numba arm "
                 "function-only): %s." % (p["function_only_threshold_s"], "; ".join(fo) if fo else "none"))
        w.append("")
        w.append("NOT APPLICABLE (factorize's code by name): %s." % ("; ".join(na) if na else "none"))
        w.append("")
        parts = []
        for r in sz["ratios"]:
            t = "%s / %s %s" % (r["num"], r["den"], fmt(r["substitution"]))
            if r.get("factorize") is not None:
                t += " (factorize %s)" % fmt(r["factorize"])
            parts.append(t)
        w.append("THE RATIOS (substitution medians; factorization medians in brackets): %s."
                 % "; ".join(parts))
        w.append("")
        chk = ["%s %.2e (%s)" % (a["id"], a["check"]["max_rel_diff"],
                                 "within" if a["check"]["within_tolerance"] else "outside")
               for a in sz["arms"] + sz.get("direct_arms", []) if a.get("check")]
        w.append("THE CHECKS: max |x - x_ref| / max |x_ref| against %g, x_ref = the 0.1.11 arm's x "
                 "on G (on S its kept entries): %s. The update's SUPERLU.NATURAL.G x bit-equal to "
                 "the 0.1.11 arm's: %s." % (p["tolerance"], "; ".join(chk), sz["oracle_bit_equal"]))
    if d:
        w.append("")
        w.append("## The wrapper overhead against n (sparse_numba median / direct median)")
        w.append("")
        w.append("Per pair, the quotient of the sparse_numba arm's median and the C++ direct-call "
                 "arm's median on the same matrix and rhs; G's n %s; S's n %s. A function-only pair "
                 "has no quotient."
                 % (" / ".join(str(rec["per_size"][s]["facts"]["G"]["n"]) for s in rec["sizes"]),
                    " / ".join(str(rec["per_size"][s]["facts"]["S"]["n"]) for s in rec["sizes"])))
        pairs = []
        for s in rec["sizes"]:
            for row in rec["per_size"][s]["overhead"]:
                if row["pair"] not in pairs:
                    pairs.append(row["pair"])
        for what in ("factorize", "substitution", "refactorize"):
            w.append("")
            w.append("| %s | %s |" % (what, " | ".join(rec["sizes"])))
            w.append("|---|%s" % ("---|" * len(rec["sizes"])))
            for pair in pairs:
                cells = []
                for s in rec["sizes"]:
                    row = [r for r in rec["per_size"][s]["overhead"] if r["pair"] == pair]
                    if not row:
                        cells.append("--")
                    elif row[0][what] is None:
                        cells.append("function-only" if "function_only" in
                                     (row[0]["sparse_numba_status"], row[0]["direct_status"]) else "--")
                    else:
                        cells.append(fmt(row[0][what]))
                w.append("| %s | %s |" % (pair, " | ".join(cells)))
    w.append("")
    with open(path, "x", encoding="ascii", newline="\n") as f:
        f.write("\n".join(w))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--label", default="")
    ap.add_argument("--build-log")
    ap.add_argument("--reference-record")
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "records"))
    ap.add_argument("--scratch")
    ap.add_argument("--release-site")
    ap.add_argument("--estimate-from", default=ESTIMATE_FROM)
    ap.add_argument("--no-direct", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--child-0111", nargs=3, metavar=("COPY_DIR", "OUT_DIR", "SIZES"))
    args = ap.parse_args(argv)
    assert_threads()
    if args.child_0111:
        copy_dir, out_dir, sizes = args.child_0111
        child_0111(copy_dir, out_dir, sizes.split(","))
        return 0
    if not args.name or not re.match(r"^[A-Za-z0-9_]+$", args.name):
        fail(2, "--name [A-Za-z0-9_]+ is required")
    sizes = args.sizes.split(",")
    for s in sizes:
        if s not in SIZES:
            fail(2, "unknown size %s" % s)
    if not args.release_site and not args.reference_record:
        fail(2, "--release-site DIR (the site-packages holding 0.1.11) or --reference-record is required")
    out_dir = os.path.abspath(args.out_dir)
    inside = os.path.normcase(out_dir).startswith(os.path.normcase(REPO + os.sep))
    if args.smoke and inside:
        fail(2, "--smoke writes outside the repository only")
    if not args.smoke and git("status", "--porcelain").strip():
        fail(4, "the tree is dirty")
    sha = git("rev-parse", "--short=7", "HEAD").strip()
    sha_full = git("rev-parse", "HEAD").strip()
    stem = os.path.join(out_dir, "compare_%s_%s" % (args.name, sha))
    outputs = (".json", ".md") + ((".build.log",) if args.build_log else ())
    for ext in outputs:
        if os.path.exists(stem + ext):
            fail(3, "%s exists" % (stem + ext))
    direct_on = not args.no_direct
    estimate = expected_wall(args.estimate_from, sizes, direct_on)
    sys.stderr.write("compare_record: the expected total wall %.0f s (%.1f m; target %.0f m), from %s\n"
                     % (estimate["expected_total_wall_s"], estimate["expected_total_wall_s"] / 60,
                        estimate["target_s"] / 60, estimate["from"] or "the cap's bound"))

    sys.path.insert(0, REPO)
    import sparse_numba
    from sparse_numba import api, errors
    assert os.path.normcase(os.path.abspath(sparse_numba.__file__)).startswith(
        os.path.normcase(os.path.join(REPO, "sparse_numba") + os.sep)), sparse_numba.__file__
    names = {getattr(errors, k): k for k in dir(errors)
             if k.isupper() and isinstance(getattr(errors, k), int)}
    sums = vehicle_sums()
    vehicles = {size: load_vehicle(size, sums) for size in sizes}
    scratch = args.scratch or tempfile.mkdtemp(prefix="compare_record_")
    os.makedirs(scratch, exist_ok=True)
    t_start = time.perf_counter()

    # the 0.1.11 arm: run in a child, or cited
    release = {}
    if args.reference_record:
        with open(args.reference_record, "r", encoding="ascii") as f:
            ref_rec = json.load(f)
        if ref_rec["git_sha_full"] != sha_full:
            fail(2, "the reference record is of tree %s, this tree %s" % (ref_rec["git_sha"], sha))
        cited = {"path": os.path.abspath(args.reference_record).replace("\\", "/"),
                 "name": os.path.basename(args.reference_record),
                 "sha256": sha256_file(args.reference_record)}
        release = {"cited_from": cited, "version": ref_rec["release_0111"]["version"],
                   "package_file": ref_rec["release_0111"]["package_file"]}
        ref_measured = {s: ref_rec["per_size"][s]["release_0111_measured"] for s in sizes}
        x_ref = {s: np.array(ref_rec["per_size"][s]["x_ref"], dtype=np.float64) for s in sizes}
    else:
        copy_dir = os.path.join(scratch, "sparse_numba_0111")
        if os.path.exists(copy_dir):
            fail(2, "%s exists" % copy_dir)
        copy = copy_release(args.release_site, copy_dir)
        child_out = os.path.join(scratch, "child_0111")
        os.makedirs(child_out)
        try:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-0111",
                                   copy_dir, child_out, ",".join(sizes)],
                                  capture_output=True, text=True, timeout=7200, cwd=scratch)
        except subprocess.TimeoutExpired:
            fail(8, "the 0.1.11 child timed out")
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            fail(8, "the 0.1.11 child exited %d" % proc.returncode)
        with open(os.path.join(child_out, "child_0111.json"), "r", encoding="ascii") as f:
            child = json.load(f)
        release = {"copy": copy, "version": child["version"], "package_file": child["package_file"],
                   "child_stdout_lines": len(proc.stdout.splitlines()),
                   "child_stderr": proc.stderr[-2000:]}
        ref_measured = child["sizes"]
        x_ref = {s: np.load(os.path.join(child_out, "x_0111_%s.npy" % s)) for s in sizes}

    env = environment(args.build_log)
    versions = {"superlu": env["superlu_version"], "umfpack": env["umfpack"]["version"]}
    kernels = make_kernels()
    warm_dispatch(api, kernels)
    direct = build_direct(env, scratch) if direct_on else None
    per_size = {}
    for size in sizes:
        v = vehicles[size]
        arms, direct_arms = [], []
        ref_arm = arm_0111(versions, size, ref_measured[size],
                           release.get("cited_from"))
        arms.append(ref_arm)
        xs = {}
        pairs = [("G", be, o) for be in BACKENDS for o in ORDERINGS]
        pairs += [("S", "EIGEN_LDLT", o) for o in ORDERINGS]
        pairs += [("S", "SUPERLU", "COLAMD"), ("S", "EIGEN_LU", "COLAMD")]

        def check(arm, x, matrix_name):
            ref = x_ref[size] if matrix_name == "G" else x_ref[size][v["keep"]]
            d = rel_diff(x, ref)
            arm["check"] = {"max_rel_diff": d, "within_tolerance": bool(d <= PROTOCOL["tolerance"]),
                            "against": "x_ref" if matrix_name == "G" else "x_ref[keep]"}

        for matrix_name, be, o in pairs:
            matrix = v[matrix_name]
            b = v["b"] if matrix_name == "G" else v["b_s"]
            arm, x = run_arm(api, kernels, names, matrix_name, matrix, b, be, o, versions)
            if x is not None:
                check(arm, x, matrix_name)
                xs[arm["id"]] = x
            arms.append(arm)
            sys.stderr.write("%s %s %s %s\n" % (size, arm["id"], arm["status"],
                                                 fmt(arm.get("substitution_us", {}).get("median")
                                                     if arm.get("substitution_us") else None)))
            if direct and be in DIRECT_BACKENDS and x is not None:
                d_arm, d_x = run_direct(direct, scratch, size, matrix_name, matrix, b, be, o,
                                        arm["status"] == "function_only")
                if d_x is not None:
                    check(d_arm, d_x, matrix_name)
                direct_arms.append(d_arm)
                sys.stderr.write("%s %s %s %s\n" % (size, d_arm["id"], d_arm["status"],
                                                     fmt(d_arm["substitution_us"]["median"]
                                                         if d_arm.get("substitution_us") else None)))
        oracle = bits_equal(xs.get("SUPERLU.NATURAL.G", np.zeros(0)), x_ref[size])
        if not oracle:
            fail(5, "%s: the update's SUPERLU NATURAL x is not bit-equal to the 0.1.11 arm's" % size)
        g, s = v["G"], v["S"]
        per_size[size] = {
            "files_sha256": v["files_sha256"],
            "facts": {"G": matrix_facts(*g), "S": matrix_facts(*s)},
            "release_0111_measured": ref_measured[size],
            "x_ref": [float(t) for t in x_ref[size]],
            "x_ref_sha256": hashlib.sha256(np.ascontiguousarray(x_ref[size]).tobytes()).hexdigest(),
            "oracle_bit_equal": oracle, "arms": arms, "ratios": ratios(arms, ref_arm),
            "direct_arms": direct_arms, "overhead": overhead(arms, direct_arms)}
    if not args.smoke and git("status", "--porcelain").strip():
        fail(4, "the tree became dirty during the run")
    if git("rev-parse", "HEAD").strip() != sha_full:
        fail(4, "HEAD moved during the run")
    if direct:
        direct = {k: val for k, val in direct.items() if not k.startswith("_")}
    rec = {"tool": "tools/compare_record.py", "name": args.name, "label": args.label,
           "git_sha": sha, "git_sha_full": sha_full, "smoke": bool(args.smoke),
           "written": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "wall_s": time.perf_counter() - t_start, "expected_wall": estimate,
           "protocol": PROTOCOL, "sizes": sizes, "release_0111": release, "direct": direct,
           "environment": env, "per_size": per_size}
    os.makedirs(out_dir, exist_ok=True)
    for ext in outputs:
        if os.path.exists(stem + ext):
            fail(3, "%s exists" % (stem + ext))
    if args.build_log:
        with open(args.build_log, "rb") as src:
            raw = src.read().replace(b"\r\n", b"\n").decode("ascii", errors="replace")
        with open(stem + ".build.log", "x", encoding="ascii", errors="replace", newline="\n") as dst:
            dst.write(raw.replace(chr(0xfffd), "?"))
    with open(stem + ".json", "x", encoding="ascii", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    if os.path.exists(stem + ".md"):
        fail(3, "%s exists" % (stem + ".md"))
    write_md(stem + ".md", rec)
    for ext in outputs:
        print(stem + ext)
    return 0


if __name__ == "__main__":
    sys.exit(main())
