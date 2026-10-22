# Sparse_Numba

A Numba-compatible sparse linear solver: SuperLU, Eigen (SparseLU and
SimplicialLDLT) and a user-supplied UMFPACK behind one interface that runs
inside `@njit(nogil=True)` code and keeps its numba cache across processes.

- New to it? The [user guide](docs/USER_GUIDE.md) walks through the
  backends, the orderings, the handle's life, numba use and the cache.
- Scripts to run and read: [examples/](examples/README.md).
- What 0.2.0 adds and changes over 0.1.11:
  [the release notes](docs/RELEASE_NOTES_0.2.0.md).

[![PyPI version](https://badge.fury.io/py/sparse-numba.svg)](https://badge.fury.io/py/sparse-numba)
[![Build Status](https://github.com/th1275/sparse_numba/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/th1275/sparse_numba/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/sparse-numba.svg)](https://pypi.org/project/sparse-numba/)

## Why Sparse_Numba?

Sparse linear solvers in Python (SciPy, CVXOPT, KVXOPT) are called from the
interpreter. Sparse_Numba's functions are `njit(nogil=True, cache=True)`:
they are called from Numba-compiled code, release the GIL, and so run in
parallel threads (`prange`) without a round trip through Python.

## Installation

```bash
pip install sparse-numba
```

One wheel per Python (CPython 3.10 - 3.13) and platform, each built and
tested on its own Python. Requires numba >= 0.61 (the first numba with
Python 3.13; it needs Python 3.10) and numpy >= 1.24. Python 3.9 users stay
on 0.1.11.

| Platform | Python | Wheels | SuperLU / OpenBLAS | UMFPACK |
|---|---|---|---|---|
| Windows (x86_64) | 3.10 - 3.13 | yes | vendored in the wheel | the user's SuiteSparse |
| Linux (x86_64, manylinux) | 3.10 - 3.13 | yes | the system's (libsuperlu, libopenblas) | the user's SuiteSparse |
| macOS (Apple Silicon) | 3.10 - 3.13 | yes | Homebrew's (superlu, openblas) | the user's SuiteSparse |
| macOS (Intel) | 3.10 - 3.13 | build from source | Homebrew's | the user's SuiteSparse |

Intel Mac users: `pip install sparse-numba --no-binary=sparse-numba`.

### UMFPACK: bring your own SuiteSparse

No SuiteSparse binary ships with sparse_numba (UMFPACK and CHOLMOD are
GPL / LGPL; see Licences below). The UMFPACK backend loads the user's
library at its first call:

- `SPARSE_NUMBA_UMFPACK_LIBRARY=/path/to/libumfpack.{dll,so,dylib}` names
  it (only that path is tried); otherwise
- the platform's library names are tried through the platform loader:
  `libumfpack.dll`, `umfpack.dll` (Windows: the default DLL directories,
  among them those added with `os.add_dll_directory`; sparse_numba adds
  `SUITESPARSE_BIN` when it is set); `libumfpack.so.6`, `.so.5`, `.so`
  (Linux); `libumfpack.6.dylib`, `.5.dylib`, `.dylib` (macOS).

Not found: every UMFPACK call returns `BACKEND_ABSENT` (-20), `backends()`
leaves it out, and `require(UMFPACK)` raises with this hint. Nothing is
printed at import.

### Building from source

Windows needs MinGW-w64 (x86_64-posix-seh) on PATH; setup.py selects the
`mingw32` compiler itself. Then:

```bash
python -m build --wheel
pip install dist/sparse_numba-<VERSION>-<TAGS>.whl
```

Linux and macOS need the SuperLU and OpenBLAS development packages (and
SuiteSparse's headers for the UMFPACK interface). Build switches (setup.py,
read at build time):

| variable | effect |
|---|---|
| `SPARSE_NUMBA_SKIP_UMFPACK=1` | the UMFPACK interface extension is not built (no SuiteSparse headers needed) |
| `SPARSE_NUMBA_DEBUG=1` | every extension is compiled WITHOUT `-DNDEBUG` (the asserts on, for development); never set for a release wheel |
| `SPARSE_NUMBA_EXTRA_CFLAGS="..."` | appended to every extension's compile flags (a comparison build, e.g. `-march=native`); never set for a release wheel |

The release flags: `-O3 -DNDEBUG` on every extension and platform (plus
`-DMS_WIN64` on 64-bit Windows, `-fPIC` on Linux / macOS); the Eigen
extension adds `-std=c++17 -DEIGEN_MPL2_ONLY -DEIGEN_DONT_PARALLELIZE`;
MinGW links with `-Wl,--no-insert-timestamp` (a rebuild of unchanged
source gives the same extension bytes). The flags compiled in are part of
`cache_key()`.

More: the [user guide](docs/USER_GUIDE.md) (its parts 1, Install, and 8, UMFPACK's library).

## The API

Every function below is `njit(nogil=True, cache=True)` unless marked
python-level. Input: CSC of an n x n matrix, `data` float64, `indices`
and `indptr` int32; `b` / `out` float64, 1-D, length n, C-contiguous;
`B` / `OUT` (n, k) C-contiguous.

### ADDED in 0.2.0: the backend-neutral interface

| function | returns |
|---|---|
| `factorize(data, indices, indptr, n, backend, ordering)` | `(handle, info)` |
| `solve_into(handle, b, out)` | `info` (x written into `out`; no allocation on SuperLU / UMFPACK) |
| `solve_many_into(handle, B, OUT)` | `info` |
| `refactorize(handle, data)` | `info` (new values, the same pattern, the analysis kept) |
| `free(handle)` | `info` |
| `solve(data, indices, indptr, n, b, backend, ordering)` | `(x, info)` (factorize + solve_into + free) |
| `handle_info(handle)` | `(n, nnz_L, nnz_U, backend_id, ordering_id, last_status, pivot_tolerance, info)` |
| `Factors(data, indices, indptr, n, backend, ordering)` | python-level context manager: `.solve(b)`, `.refactorize(data)`, `.free()`; raises `SparseNumbaError` |
| `backends()`, `require(backend)`, `is_eigen_available()` | python-level: the available backend ids; raise with an install hint; Eigen present |
| `explain(info)`, `SparseNumbaError`, `SymbolConflictError` | python-level: a code in words; the exceptions |
| `cache_key()` | python-level: a string for the caller's own cache key (below) |
| `sparse_numba.symbols.TABLE`, `declared(name)` | the LLVM symbol table and its signatures, as data |

Backends: `SUPERLU`, `UMFPACK`, `EIGEN_LU`, `EIGEN_LDLT`. Orderings:
`NATURAL`, `COLAMD`, `AMD`, `MMD_AT_PLUS_A`. There is no default backend
and no default ordering: both are arguments of `factorize`.

```python
import numpy as np
from numba import njit
import sparse_numba as sn

# a 3 x 3 CSC matrix
data = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
indptr = np.array([0, 2, 5, 7], dtype=np.int32)
n = 3

handle, info = sn.factorize(data, indices, indptr, n, sn.SUPERLU, sn.COLAMD)
assert info == 0, sn.explain(info)

@njit(nogil=True)
def many_solves(handle, rhs, out):
    status = 0
    for i in range(rhs.shape[0]):
        status |= sn.solve_into(handle, rhs[i], out[i])
    return status

rhs = np.random.rand(1000, n)
out = np.empty_like(rhs)
assert many_solves(handle, rhs, out) == 0

data2 = data * 1.5                         # new values, the same pattern
assert sn.refactorize(handle, data2) == 0
assert sn.free(handle) == 0
```

The same with Eigen's SparseLU at AMD, or SimplicialLDLT on a symmetric
positive definite matrix: `sn.factorize(..., sn.EIGEN_LU, sn.AMD)`,
`sn.factorize(..., sn.EIGEN_LDLT, sn.AMD)`.

### What each backend supports

"refused" = the call returns `ORDERING_UNSUPPORTED` or `BAD_SHAPE` by name,
never a silent fallback.

| | SUPERLU | UMFPACK | EIGEN_LU | EIGEN_LDLT |
|---|---|---|---|---|
| unsymmetric general | yes | yes | yes | refused (BAD_SHAPE) |
| symmetric positive definite | yes (as general) | yes (as general) | yes (as general) | yes (the only input) |
| NATURAL | yes (ColPerm NATURAL) | yes (UMFPACK_ORDERING_NONE; refused by a library without umfpack_version) | NaturalOrdering | refused |
| COLAMD | yes | yes (STRATEGY_UNSYMMETRIC) | COLAMDOrdering | refused |
| AMD | refused | yes (STRATEGY_SYMMETRIC) | AMDOrdering | yes (its only one) |
| MMD_AT_PLUS_A | yes | refused | refused | refused |
| refactorize on the same pattern | yes (SamePattern) | yes (numeric again) | yes (factorize after analyzePattern) | yes |
| several right-hand sides in one call | yes (dgstrs nrhs) | a loop per column | yes (a dense block) | yes |
| solve into a held out | yes | yes | yes | yes |
| what it needs at run time | the vendored DLLs (Windows) / the system's (Linux, macOS) | the user's SuiteSparse | nothing (compiled in) | nothing (compiled in) |
| threads | single | single | single (EIGEN_DONT_PARALLELIZE) | single |
| handle_info's nnz_L / nnz_U | L (SCformat), U (NCformat) | Info[UMFPACK_LNZ], Info[UMFPACK_UNZ] | nnzL(), nnzU() | the strictly lower L; 0 |
| pivot tolerance (handle_info) | DiagPivotThresh (1.0) | 1.0 | 1.0 | 0.0 (no pivoting) |

Notes:

- EIGEN_LDLT checks that the PATTERN is symmetric (the stored entries,
  explicit zeros among them), not the values; it reads the lower triangle.
  An indefinite matrix is refused after the factorization: `SINGULAR` (= 1)
  when an entry of D is <= 0.
- A row index repeated inside one column: the Eigen backends sum the
  entries; UMFPACK refuses it (BACKEND_ERROR); SuperLU passes it to dgstrf
  as given.
- SINGULAR: SuperLU and UMFPACK return k > 0, the zero pivot's 1-based
  index; the Eigen backends return 1.

### The status codes

`info == 0` is success; `info != 0` is failure (the 0.1.11 contract).
`sparse_numba.errors` names every code and `explain(info)` says it in
words; nothing is printed by any call.

| code | value |
|---|---|
| OK | 0 |
| BAD_DTYPE | -1 |
| BAD_SHAPE | -2 |
| BAD_INDPTR | -3 |
| BAD_INDEX | -4 |
| NOT_SQUARE | -5 |
| HANDLE_INVALID | -10 |
| HANDLE_FREED | -11 (a double free or a use after free) |
| HANDLE_WRONG_BACKEND | -12 |
| BACKEND_ABSENT | -20 |
| ORDERING_UNSUPPORTED | -21 |
| ALLOC_FAILED | -30 |
| BACKEND_ERROR | -31 (the backend's own status in handle_info's last_status) |
| SINGULAR | k > 0 |

The handle is an int64 registry index: the caller who factorized owns it
and frees it; it is process-local (it never crosses processes).

### KEPT from 0.1.11 (same names, signatures and return shapes)

`superlu_solve_csc / _coo / _csr`, `superlu_factorize_csc / _coo / _csr`,
`superlu_solve_factored`, `superlu_free_factors`, and the same `umfpack_*`
set; `convert_coo_to_csc`, `convert_csr_to_csc`, `convert_coo_to_csr`,
`sparse_matvec_csr`, `validate_sparse_matrix`, `check_matrix_properties`,
`validate_csc_matrix`; `is_slu_available`, `is_umf_available`. They are
thin shims over the interface above; SuperLU's give bit-identical
solutions to 0.1.11 (NATURAL ordering). The values some codes carry
changed (the 0.2.0 release notes, docs/RELEASE_NOTES_0.2.0.md).

```python
from sparse_numba import superlu_factorize_csc, superlu_solve_factored, superlu_free_factors

b = np.ones(n)
handle, info = superlu_factorize_csc(data, indices, indptr)
x, info = superlu_solve_factored(handle, b)
superlu_free_factors(handle)
```

### DEPRECATED (kept through 0.2.x, removed at 0.3)

`*_solve_factored(handle, b) -> (x, info)` (allocates x) in favour of
`solve_into`; the `*_factorize_*` forms without an ordering in favour of
`factorize(..., backend, ordering)`; `*_free_factors` in favour of `free`.
The deprecation is in the documentation only (njit functions do not warn).

## The numba cache across processes

- Every function is `cache=True`; the C entry points are registered with
  llvmlite under stable names (`sparse_numba.symbols.TABLE`) at import, so
  a cached caller loads in a new process without recompiling.
- No cache file ships in a wheel (no `__pycache__`, `*.nbi`, `*.nbc`).
- A read-only install (site-packages not writable): numba writes its
  cache where `NUMBA_CACHE_DIR` points.
- Upgrades: the symbol names are stable across releases; the contract
  for a caller that caches its own compiled code is to fold
  `sparse_numba.cache_key()` into its cache key (it names the version, the symbol table, each
  extension's sha256, the SuperLU / UMFPACK / Eigen versions and the
  compile flags). For example, a downstream DAE solver that keeps
  factorizations across steps records the key in its generated modules'
  manifest and regenerates them when it changes.

## Performance: the comparison record

The numbers below are one run of `tools/compare_record.py` at tree cb0617e on one machine (its json record, holding every sample, sha256 ff2cc551f5347012b8aa88870be4b88afd36834445dde17f4d923364f51bf1fe; the record is kept with the development tree). Machine: Intel(R) Core(TM) Ultra 7 258V, 8 logical CPUs, Windows power plan Balanced; Python 3.11.9, numpy 2.4.6, numba 0.67.0; the wheel's build (`-O3 -DNDEBUG`, MinGW gcc 14.2.0). Single-threaded (OPENBLAS / OMP / MKL / NUMEXPR threads = 1; Eigen with EIGEN_DONT_PARALLELIZE). The protocol: per arm the factorization is the second of two calls, 7 repeats; the substitution is `solve_into` on a held `out` from a numba loop, 10,000 calls a repeat x 7 repeats; refactorize 100 calls x 7; a repeat's calls cut to fit 2 s (the actual N in the record); an arm whose first factorization takes over 0.5 s is run as a function test only (its solution checked, no timing). Every value is the median of the 7 repeats, in microseconds; min and max are in the record. Every arm's solution is within 1e-10 (relative, max norm) of the 0.1.11 SuperLU solution; SuperLU NATURAL's is bit-identical to it at every size.

The matrices: G, the unsymmetric chain matrix of a power-system model (n = 688 / 2068 / 6898 / 20698); S, its symmetric positive definite nodal reduction (n = 207 / 621 / 2070 / 6210). The same right-hand side every call.

### Substitution (us per solve_into) against n

| arm | G n 688 | G n 2068 | G n 6898 | G n 20698 |
|---|---|---|---|---|
| 0.1.11 superlu_solve_factored (NATURAL) | 45.76 | 161.15 | 1138.73 | 3969.46 |
| SUPERLU.NATURAL | 36.09 | 141.69 | 1017.48 | 3935.80 |
| SUPERLU.COLAMD | 50.15 | 149.28 | 405.07 | 1425.87 |
| SUPERLU.MMD_AT_PLUS_A | 37.53 | 137.98 | 398.14 | 1529.57 |
| UMFPACK.NATURAL | 20.01 | 127.23 | 372.89 | 1575.99 |
| UMFPACK.COLAMD | 18.83 | 122.54 | 376.74 | 1552.94 |
| UMFPACK.AMD | 33.74 | 131.27 | 367.57 | 1330.19 |
| EIGEN_LU.NATURAL | 21.16 | 97.38 | 804.06 | 3739.60 |
| EIGEN_LU.COLAMD | 17.30 | 65.52 | 212.30 | 755.29 |
| EIGEN_LU.AMD | 59.09 | 540.57 | function-only | function-only |

| arm (on S) | S n 207 | S n 621 | S n 2070 | S n 6210 |
|---|---|---|---|---|
| EIGEN_LDLT.AMD | 2.69 | 9.52 | 30.61 | 108.76 |
| SUPERLU.COLAMD | 12.53 | 38.17 | 126.68 | 423.44 |
| EIGEN_LU.COLAMD | 8.32 | 22.62 | 73.16 | 251.75 |

### Factorization and refactorization (us) against n

| arm | G n 688 | G n 2068 | G n 6898 | G n 20698 |
|---|---|---|---|---|
| 0.1.11 superlu_factorize_csc (NATURAL) | 641 / -- | 3363 / -- | 17766 / -- | 68961 / -- |
| SUPERLU.NATURAL | 514 / 742 | 2731 / 2426 | 13116 / 13635 | 61935 / 60793 |
| SUPERLU.COLAMD | 327 / 351 | 1114 / 1281 | 3541 / 2987 | 16450 / 10138 |
| SUPERLU.MMD_AT_PLUS_A | 457 / 281 | 1191 / 879 | 4391 / 3473 | 16470 / 13413 |
| UMFPACK.NATURAL | 699 / 456 | 2856 / 1530 | 13124 / 5589 | 95642 / 29681 |
| UMFPACK.COLAMD | 369 / 271 | 1285 / 960 | 5247 / 4286 | 18697 / 13265 |
| UMFPACK.AMD | 324 / 235 | 1080 / 823 | 4047 / 3536 | 15834 / 11110 |
| EIGEN_LU.NATURAL | 326 / 302 | 2293 / 2045 | 15699 / 13895 | 91095 / 91893 |
| EIGEN_LU.COLAMD | 268 / 135 | 829 / 566 | 4278 / 2788 | 12585 / 8437 |
| EIGEN_LU.AMD | 3716 / 3592 | 62310 / 63306 | function-only | function-only |

Each cell: factorize / refactorize (`--`: 0.1.11 has no refactorize). UMFPACK is the user's library (6.3.5, SuiteSparse built by MSYS2), loaded at run time.

### The wrapper's overhead against n: sparse_numba / a C++ direct call

A C++ driver (`tools/cpp_direct/direct_arms.cpp`) calls the same SuperLU and Eigen builds directly -- the same DLLs, the extensions' own compile flags, the shims' options and template parameters, no sparse_numba in between -- in the same process, on the same matrix and right-hand side, under the same protocol. Each cell is the sparse_numba median divided by the direct-call median (1.000 = the same time). Single runs: the spread between repeats is in the record's min / max.

| substitution | n69 (G 688 / S 207) | n207 (G 2068 / S 621) | n690 (G 6898 / S 2070) | n2070 (G 20698 / S 6210) |
|---|---|---|---|---|
| SUPERLU.NATURAL.G | 0.969 | 1.006 | 0.974 | 0.920 |
| SUPERLU.COLAMD.G | 1.120 | 1.062 | 0.988 | 0.915 |
| SUPERLU.MMD_AT_PLUS_A.G | 0.933 | 0.977 | 0.980 | 0.941 |
| EIGEN_LU.NATURAL.G | 0.953 | 1.087 | 1.017 | 0.992 |
| EIGEN_LU.COLAMD.G | 1.015 | 1.098 | 0.961 | 0.980 |
| EIGEN_LU.AMD.G | 0.909 | 1.012 | function-only | function-only |
| EIGEN_LDLT.AMD.S | 1.021 | 1.083 | 1.123 | 1.060 |
| SUPERLU.COLAMD.S | 0.925 | 0.951 | 0.824 | 0.889 |
| EIGEN_LU.COLAMD.S | 1.105 | 1.060 | 1.129 | 1.052 |

| factorize | n69 (G 688 / S 207) | n207 (G 2068 / S 621) | n690 (G 6898 / S 2070) | n2070 (G 20698 / S 6210) |
|---|---|---|---|---|
| SUPERLU.NATURAL.G | 0.964 | 1.220 | 1.023 | 1.109 |
| SUPERLU.COLAMD.G | 0.941 | 0.798 | 0.919 | 1.234 |
| SUPERLU.MMD_AT_PLUS_A.G | 1.404 | 1.189 | 1.103 | 0.963 |
| EIGEN_LU.NATURAL.G | 1.094 | 0.816 | 0.953 | 0.931 |
| EIGEN_LU.COLAMD.G | 1.204 | 0.770 | 1.046 | 0.921 |
| EIGEN_LU.AMD.G | 1.049 | 1.603 | function-only | function-only |
| EIGEN_LDLT.AMD.S | 1.184 | 1.094 | 1.067 | 1.059 |
| SUPERLU.COLAMD.S | 0.948 | 0.969 | 0.990 | 0.928 |
| EIGEN_LU.COLAMD.S | 0.808 | 1.053 | 1.156 | 1.159 |

| refactorize | n69 (G 688 / S 207) | n207 (G 2068 / S 621) | n690 (G 6898 / S 2070) | n2070 (G 20698 / S 6210) |
|---|---|---|---|---|
| SUPERLU.NATURAL.G | 1.713 | 0.972 | 1.094 | 0.851 |
| SUPERLU.COLAMD.G | 1.285 | 1.538 | 1.017 | 0.975 |
| SUPERLU.MMD_AT_PLUS_A.G | 0.952 | 0.978 | 1.052 | 0.985 |
| EIGEN_LU.NATURAL.G | 0.969 | 1.026 | 0.960 | 0.991 |
| EIGEN_LU.COLAMD.G | 1.011 | 1.188 | 1.148 | 0.979 |
| EIGEN_LU.AMD.G | 1.007 | 0.901 | function-only | function-only |
| EIGEN_LDLT.AMD.S | 1.409 | 1.742 | 1.349 | 1.039 |
| SUPERLU.COLAMD.S | 2.167 | 0.902 | 0.901 | 1.090 |
| EIGEN_LU.COLAMD.S | 0.948 | 1.289 | 1.129 | 1.065 |

Column headers: the vehicle, then the n of G (the `.G` rows) and of S (the `.S` rows).

The direct-call arms' own medians, the substitution (us):

| direct arm | n69 (G 688 / S 207) | n207 (G 2068 / S 621) | n690 (G 6898 / S 2070) | n2070 (G 20698 / S 6210) |
|---|---|---|---|---|
| SUPERLU.NATURAL.G | 37.25 | 140.88 | 1044.71 | 4275.84 |
| SUPERLU.COLAMD.G | 44.76 | 140.53 | 409.84 | 1557.72 |
| SUPERLU.MMD_AT_PLUS_A.G | 40.24 | 141.22 | 406.38 | 1625.87 |
| EIGEN_LU.NATURAL.G | 22.21 | 89.61 | 790.64 | 3770.93 |
| EIGEN_LU.COLAMD.G | 17.04 | 59.67 | 221.02 | 770.65 |
| EIGEN_LU.AMD.G | 64.97 | 533.93 | function-only | function-only |
| EIGEN_LDLT.AMD.S | 2.64 | 8.79 | 27.26 | 102.64 |
| SUPERLU.COLAMD.S | 13.54 | 40.15 | 153.67 | 476.10 |
| EIGEN_LU.COLAMD.S | 7.52 | 21.34 | 64.78 | 239.35 |

Function-only by the 0.5 s rule: EIGEN_LU at AMD on G at n 6898 and 20698 (Eigen's AMDOrdering on the symmetrized pattern of the unsymmetric G; first factorizations 2.7 s and 39.5 s), on both sides.

## Architecture

```
User code (@njit)
    |
    v
sparse_numba.api / the kept interface modules (njit, cache=True; numba
ExternalFunctions on stable LLVM symbol names)
    |
    v
Cython layer (cdef api entry points)
    |
    v
the shims: superlu_wrapper.c, umfpack_wrapper.c (the user's library loaded
at run time), eigen_wrapper.cpp; one handle registry per backend
    |
    v
SuperLU 7.0.0 + OpenBLAS (vendored on Windows) / Eigen 3.4.0 (compiled
in) / the user's UMFPACK
```

Every layer releases the GIL.

## Licences

sparse_numba is BSD 3-Clause. The components it ships or compiles in
(docs/LICENSE_AUDIT.md holds each one's version, licence file and sha256):

| component | licence | how it ships |
|---|---|---|
| SuperLU 7.0.0 | BSD 3-Clause | Windows wheel: libsuperlu-7.dll, its licence beside it |
| OpenBLAS 0.3.29 (with reference LAPACK) | BSD 3-Clause | Windows wheel: libopenblas.dll, its licences beside it |
| GCC 14.2.0 runtime (libgcc_s_seh-1, libgomp-1, libgfortran-5) | GPL-3.0+ with the GCC Runtime Library Exception | Windows wheel, the licence files beside them |
| libquadmath-0 (GCC 14.2.0) | LGPL-2.1+ | Windows wheel, unmodified and replaceable, its licence text beside it |
| mingw-w64 winpthreads | MIT / BSD | Windows wheel, its licence beside it |
| Eigen 3.4.0 | MPL-2.0 (compiled with `-DEIGEN_MPL2_ONLY`) | compiled into the Eigen extension; the source form at https://eigen.tuxfamily.org and in this repository's third_party/eigen-3.4.0 |
| SuiteSparse (UMFPACK, CHOLMOD, ...) | GPL / LGPL | not shipped: interface-only, the user's library |

On Linux and macOS SuperLU and OpenBLAS come from the system's packages;
what each wheel bundles is read from that wheel by the release's audit.

## Citation

```bibtex
@software{hong2025sparse_numba,
  author = {Hong, Tianqi},
  title = {Sparse_Numba: A Numba-Compatible Sparse Solver},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/th1275/sparse_numba}
}
```

## Release process

1. Set the version in `setup.py`, `setup.cfg` and `sparse_numba/__init__.py`
   (`cache_key()` reads `sparse_numba.__version__`).
2. Open a pull request to main: CI builds one wheel per Python (cp310 -
   cp313) on Windows, manylinux x86_64 and macOS arm64 and tests each on
   its own Python (on Windows the full test suite from the wheel, with no
   UMFPACK library on the runner, and the wheel's contents; on Linux and
   macOS the import check and the wheel's contents); no wheel may carry a
   SuiteSparse library; no upload.
3. When CI is green, merge, tag the merge commit `vX.Y.Z` and push the
   tag: the `upload_pypi` job builds from the tagged tree and publishes
   the wheels and the sdist (trusted publishing).
4. A failed tag push before `upload_pypi` ran: delete the tag, fix, re-tag.
   A version uploaded to PyPI is never re-uploaded: bump the version.

## Contributing

Contributions are welcome. Please open an issue or pull request on
[GitHub](https://github.com/th1275/sparse_numba).
