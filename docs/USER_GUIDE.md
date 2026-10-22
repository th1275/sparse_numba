# sparse_numba 0.2.0 -- user guide

sparse_numba solves sparse linear systems A x = b from inside
`@njit(nogil=True)` code: SuperLU, Eigen's SparseLU and SimplicialLDLT,
and a user-supplied UMFPACK, behind one interface. This guide walks
through installing it, choosing a backend and an ordering, the life of a
factorization handle, calling it from numba, the numba cache across
processes, the status codes, and how the UMFPACK library is found. The
scripts in [examples/](../examples/README.md) show each part running.

Contents: [1 Install](#1-install) - [2 Backends](#2-backends) -
[3 Orderings](#3-orderings) - [4 The handle's life](#4-the-handles-life) -
[5 Calling from numba](#5-calling-from-numba) -
[6 The numba cache](#6-the-numba-cache-across-processes) -
[7 Status codes](#7-status-codes) - [8 UMFPACK's library](#8-umfpacks-library) -
[9 The 0.1.11 names](#9-the-0111-names)

## 1 Install

```bash
pip install sparse-numba
```

Python 3.10 - 3.13 (CPython), numba >= 0.61, numpy >= 1.24. One wheel per
Python and platform:

| platform | SuperLU / OpenBLAS | Eigen | UMFPACK |
|---|---|---|---|
| Windows x86_64 | in the wheel | compiled in | the user's SuiteSparse |
| Linux x86_64 (manylinux) | the system's (`libsuperlu`, `libopenblas`) | compiled in | the user's SuiteSparse |
| macOS arm64 | Homebrew's (`brew install superlu openblas`) | compiled in | the user's SuiteSparse |

Check what is available in your process:

```python
import sparse_numba as sn
print(sn.backends())            # e.g. (1, 3, 4): SUPERLU, EIGEN_LU, EIGEN_LDLT
print(sn.is_eigen_available())  # True
sn.require(sn.UMFPACK)          # raises SparseNumbaError with an install hint if absent
```

Importing sparse_numba prints nothing and loads no optional library; an
absent backend is reported by its calls (`BACKEND_ABSENT`) and by
`backends()` / `require()`. Building from source and the platform
packages: the [installation guide](Installation_Guide_for_sparse_numba.md).

## 2 Backends

| id | name | what it is | input |
|---|---|---|---|
| 1 | `SUPERLU` | SuperLU 7.0.0, `dgstrf` / `dgstrs` | any square nonsingular matrix |
| 2 | `UMFPACK` | the user's UMFPACK (SuiteSparse), loaded at the first call | any square nonsingular matrix |
| 3 | `EIGEN_LU` | Eigen 3.4.0 `SparseLU` | any square nonsingular matrix |
| 4 | `EIGEN_LDLT` | Eigen 3.4.0 `SimplicialLDLT` | a symmetric positive definite matrix |

There is no default backend: you name one in every `factorize` call.
Every backend is single-threaded (Eigen is compiled with
`EIGEN_DONT_PARALLELIZE`); run several factorizations in parallel threads
(`prange`) instead -- every call releases the GIL.

EIGEN_LDLT checks the PATTERN (the stored entries) for symmetry, not the
values, and reads the lower triangle: an unsymmetric pattern is refused
with `BAD_SHAPE`; a symmetric pattern whose lower triangle defines a
matrix that is not positive definite is refused after the factorization
with `SINGULAR` (= 1), when an entry of D is <= 0.

## 3 Orderings

The fill-reducing ordering is the second argument you always pass. A pair
the backend does not offer returns `ORDERING_UNSUPPORTED`; nothing falls
back silently.

| ordering | SUPERLU | UMFPACK | EIGEN_LU | EIGEN_LDLT |
|---|---|---|---|---|
| `NATURAL` (0) | yes (the 0.1.11 factorization) | yes (`UMFPACK_ORDERING_NONE`; needs a library with `umfpack_version`) | yes | no |
| `COLAMD` (1) | yes | yes (strategy unsymmetric) | yes | no |
| `AMD` (2) | no | yes (strategy symmetric) | yes | yes (its only one) |
| `MMD_AT_PLUS_A` (3) | yes | no | no | no |

Which ordering fills least depends on the matrix; `handle_info` reports
nnz(L) and nnz(U), so you can measure it.
[examples/orderings_compare.py](../examples/orderings_compare.py) prints
the factorization time, the fill and the solve time of every pair on one
matrix.

## 4 The handle's life

The matrix is CSC: `data` float64, `indices` and `indptr` int32, n x n.

```python
import numpy as np
import sparse_numba as sn

data = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
indptr = np.array([0, 2, 5, 7], dtype=np.int32)
n = 3

handle, info = sn.factorize(data, indices, indptr, n, sn.SUPERLU, sn.COLAMD)
if info != 0:
    raise RuntimeError(sn.explain(info))

b = np.ones(n)
x = np.empty(n)
info = sn.solve_into(handle, b, x)          # x written in place; b read only

B = np.ones((n, 4))                         # (n, k), C-contiguous
X = np.empty_like(B)
info = sn.solve_many_into(handle, B, X)     # k right-hand sides in one call

info = sn.refactorize(handle, data * 2.0)   # new values, the same pattern
n_, nnz_l, nnz_u, backend, ordering, last_status, pivot_tol, info = sn.handle_info(handle)

info = sn.free(handle)                      # the caller who factorized frees
```

- `factorize` returns `(handle, info)`; the handle is nonzero when info is
  0 and is yours to free. It is an int64 index into the backend's
  registry: process-local (never send it to another process) and checked
  on every call -- a freed handle returns `HANDLE_FREED`, another
  backend's `HANDLE_WRONG_BACKEND`, never a crash.
- `solve_into` writes into an `out` you hold; on SuperLU and UMFPACK it
  allocates nothing per call.
- `refactorize` keeps the analysis (the ordering, the symbolic
  factorization) and factors new values on the same pattern, in the
  pattern's order: the step a Newton iteration repeats.
- `solve(data, indices, indptr, n, b, backend, ordering) -> (x, info)` is
  factorize + solve_into + free in one call.
- For Python code, `Factors` wraps the handle as a context manager and
  raises `SparseNumbaError` on a failed call:

```python
with sn.Factors(data, indices, indptr, n, sn.EIGEN_LU, sn.COLAMD) as f:
    x = f.solve(b)
    f.refactorize(data * 2.0)
```

[examples/superlu_walk.py](../examples/superlu_walk.py),
[examples/umfpack_walk.py](../examples/umfpack_walk.py) and
[examples/eigen_lu_ldlt.py](../examples/eigen_lu_ldlt.py) walk this life on
each backend.

## 5 Calling from numba

`factorize`, `solve_into`, `solve_many_into`, `refactorize`, `free`,
`solve` and `handle_info` are `njit(nogil=True, cache=True)` functions:
call them from your own njit code, with the backend and ordering as
integers.

```python
from numba import njit, prange

@njit(nogil=True)
def solve_rows(handle, rhs, out):
    status = 0
    for i in range(rhs.shape[0]):
        status |= sn.solve_into(handle, rhs[i], out[i])
    return status

@njit(nogil=True, parallel=True)
def factor_many(datas, indices, indptr, n, handles, infos):
    for i in prange(datas.shape[0]):       # one factorization per thread
        handle, info = sn.factorize(datas[i], indices, indptr, n,
                                    sn.SUPERLU, sn.COLAMD)
        handles[i] = handle
        infos[i] = info
```

Give each thread its own handle for factorize / refactorize / solve /
free. (On SuperLU, concurrent `solve_into` calls may also share one
handle: a second concurrent solve takes its own work buffers.) `explain`,
`backends`, `require`, `Factors` and `cache_key` are Python-level.

## 6 The numba cache across processes

Every sparse_numba function is `cache=True`, and the C entry points are
registered with llvmlite under stable names at import. A function of yours
compiled with `@njit(cache=True)` that calls sparse_numba therefore loads
from numba's cache in a new process without recompiling.
[examples/cache_two_processes.py](../examples/cache_two_processes.py)
shows it: the first process compiles, the second loads.

- No cache file ships in a wheel. On a read-only install, numba writes its
  cache where `NUMBA_CACHE_DIR` points.
- Upgrades: the symbol names are stable across releases. If you cache
  compiled code of your own (generated modules, a manifest of builds),
  fold `sparse_numba.cache_key()` into your cache key: it names the
  version, the symbol table, each extension's sha256, the SuperLU / Eigen
  versions with their compile flags, and the UMFPACK library's version
  (`absent` when it is not found). When it changes, rebuild. A downstream
  DAE solver that keeps factorizations across steps, for instance, records
  the key in its generated modules' manifest and regenerates them when it
  changes.

## 7 Status codes

Every call returns an `info`: `info == 0` is success, `info != 0` is
failure. `sparse_numba.errors` names each code; `sn.explain(info)` says it
in words. Nothing is printed by any call.

| code | value | meaning |
|---|---|---|
| `OK` | 0 | success |
| `BAD_DTYPE` | -1 | an array of the wrong dtype |
| `BAD_SHAPE` | -2 | an array of the wrong rank, length or layout; n <= 0; EIGEN_LDLT given an unsymmetric pattern |
| `BAD_INDPTR` | -3 | indptr[0] != 0, indptr[n] != nnz, or indptr decreasing |
| `BAD_INDEX` | -4 | a row index outside [0, n) |
| `NOT_SQUARE` | -5 | len(indptr) != n + 1 |
| `HANDLE_INVALID` | -10 | zero, negative, or never handed out |
| `HANDLE_FREED` | -11 | a double free or a use after free |
| `HANDLE_WRONG_BACKEND` | -12 | a handle passed to another backend's function |
| `BACKEND_ABSENT` | -20 | the backend is not available in this process |
| `ORDERING_UNSUPPORTED` | -21 | the backend does not offer this ordering |
| `ALLOC_FAILED` | -30 | an allocation failed |
| `BACKEND_ERROR` | -31 | the backend's own failure (its status in handle_info's `last_status`) |
| `SINGULAR` | k > 0 | a zero pivot: SuperLU and UMFPACK report its 1-based index k, the Eigen backends 1 |

A row index repeated inside one column: the Eigen backends sum the
entries, UMFPACK refuses it (`BACKEND_ERROR`), SuperLU passes it to
`dgstrf` as given -- sum duplicates before factorizing.

## 8 UMFPACK's library

UMFPACK is interface-only: sparse_numba ships no SuiteSparse binary
(SuiteSparse's UMFPACK and CHOLMOD are GPL / LGPL). The UMFPACK backend
loads YOUR library at its first call, once per process:

1. `SPARSE_NUMBA_UMFPACK_LIBRARY=/path/to/libumfpack.{dll,so,dylib}`: only
   that path is tried;
2. otherwise the platform's names through the platform's loader:
   `libumfpack.dll` / `umfpack.dll` on Windows (the default DLL
   directories, among them those added with `os.add_dll_directory`;
   sparse_numba adds `SUITESPARSE_BIN` when it is set),
   `libumfpack.so.6` / `.so.5` / `.so` on Linux,
   `libumfpack.6.dylib` / `.5.dylib` / `.dylib` on macOS.

Getting a library: Linux `apt install libsuitesparse-dev` (or your
distribution's suitesparse package); macOS `brew install suite-sparse`;
Windows MSYS2's `mingw-w64-x86_64-suitesparse` (then
`SUITESPARSE_BIN=C:\msys64\mingw64\bin`) or a SuiteSparse build of your
own. Set the variables before the first UMFPACK call: the outcome is held
for the process. Not found: every UMFPACK call returns `BACKEND_ABSENT`,
`backends()` leaves it out, `require(UMFPACK)` raises with the hint, and
`cache_key()` names `umfpack=absent`.

## 9 The 0.1.11 names

Code written for 0.1.11 runs unchanged: `superlu_solve_csc` / `_coo` /
`_csr`, `superlu_factorize_csc` / `_coo` / `_csr`,
`superlu_solve_factored`, `superlu_free_factors`, the same `umfpack_*`
set, the conversions (`convert_coo_to_csc`, `convert_csr_to_csc`,
`convert_coo_to_csr`, `sparse_matvec_csr`) and `is_slu_available` /
`is_umf_available`. They are thin shims over the interface above;
SuperLU's give bit-identical solutions to 0.1.11 (NATURAL ordering) on
the same CPU and BLAS kernel. The
values some codes carry changed, and `*_solve_factored`, the
`*_factorize_*` forms without an ordering and `*_free_factors` are
deprecated (kept through 0.2.x): see the
[0.2.0 release notes](RELEASE_NOTES_0.2.0.md).
[examples/kept_0111_names.py](../examples/kept_0111_names.py) runs them
beside the new API.
