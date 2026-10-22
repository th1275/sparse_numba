# sparse_numba 0.2.0 -- release notes

0.2.0 is one comprehensive update over 0.1.11. Every 0.1.11 public name
keeps its name, positional signature and return shape; SuperLU at the
NATURAL ordering gives bit-identical solutions to 0.1.11 on the same CPU
and BLAS kernel (the oracle cells of tests/; OpenBLAS selects its kernel
by CPU at run time, so CI pins OPENBLAS_CORETYPE=Haswell for them). What changed is listed below; the README holds the API, the
backend table and the measured numbers.

## Added

- A backend-neutral interface, every function `njit(nogil=True,
  cache=True)`: `factorize(data, indices, indptr, n, backend, ordering) ->
  (handle, info)`, `solve_into(handle, b, out) -> info`,
  `solve_many_into(handle, B, OUT) -> info`, `refactorize(handle, data) ->
  info`, `free(handle) -> info`, `solve(data, indices, indptr, n, b,
  backend, ordering) -> (x, info)`, `handle_info(handle)`; the Python-level
  `Factors` (a context manager), `explain(info)`, `SparseNumbaError`,
  `SymbolConflictError`, `require`, `backends()`, `is_eigen_available()`,
  `cache_key()`, `sparse_numba.symbols.TABLE` / `declared()`, and the
  enums `SUPERLU`, `UMFPACK`, `EIGEN_LU`, `EIGEN_LDLT`, `NATURAL`,
  `COLAMD`, `AMD`, `MMD_AT_PLUS_A`.
- The ordering is an argument (no default): SuperLU NATURAL / COLAMD /
  MMD_AT_PLUS_A; UMFPACK NATURAL / COLAMD / AMD; EIGEN_LU NATURAL /
  COLAMD / AMD; EIGEN_LDLT AMD. A pair a backend does not offer returns
  ORDERING_UNSUPPORTED by name, never a silent fallback.
- Eigen 3.4.0 as two backends compiled into the package (header-only,
  `-DEIGEN_MPL2_ONLY`, `-DEIGEN_DONT_PARALLELIZE`): EIGEN_LU
  (`Eigen::SparseLU`) and EIGEN_LDLT (`Eigen::SimplicialLDLT`, symmetric
  positive definite input only).
- The handle is an int64 registry index (generation << 32 | slot): a
  double free or a use after free returns HANDLE_FREED, another backend's
  handle HANDLE_WRONG_BACKEND, 0 HANDLE_INVALID -- the process stays alive.
- The numba cache works across processes (`cache=True` on every njit
  function; stable LLVM symbol names registered at import). `cache_key()`
  gives a string a caller folds into its own cache key (the version, the
  symbol table, each extension's sha256, the library versions and the
  compile flags compiled into the shims).
- Typed status codes shared by every backend (sparse_numba_codes.h,
  mirrored by `sparse_numba.errors`); nothing is printed by any call.
- `import sparse_numba` is silent (no banner); an absent backend is
  reported at its first use (BACKEND_ABSENT) and by `backends()` /
  `require()`.
- A user guide, docs/USER_GUIDE.md (install, the backends, the orderings,
  the handle's life, numba use, the cache, the status codes, UMFPACK's
  library).
- examples/ at the repository's root: scripts a person runs and reads, each
  printing its results (a walk per backend, the orderings side by side,
  the cache across two processes, the 0.1.11 names beside the new API).

## Changed

- UMFPACK is INTERFACE-ONLY: no SuiteSparse binary ships in any wheel.
  The UMFPACK shim loads the user's library at the first call:
  `SPARSE_NUMBA_UMFPACK_LIBRARY` (a path) when set, else the platform's
  library names (libumfpack.dll / umfpack.dll; libumfpack.so.6 / .5 / .so;
  libumfpack.6.dylib / .5.dylib / .dylib) through the platform loader (on
  Windows the default DLL directories, among them those added with
  `os.add_dll_directory` -- the package adds `SUITESPARSE_BIN` when set).
  Not found: BACKEND_ABSENT.
- The Windows wheel vendors the SuperLU / OpenBLAS link chain only
  (libsuperlu-7, libopenblas, libgcc_s_seh-1, libgfortran-5, libgomp-1,
  libquadmath-0, libwinpthread-1), each DLL with its licence file beside
  it; libsuperlu_fortran-7.dll (imported by nothing) is no longer shipped.
- Release builds compile every extension with `-DNDEBUG` on every
  platform; `SPARSE_NUMBA_DEBUG=1` at build time leaves the asserts on (for
  development; never set for a wheel). MinGW links with
  `-Wl,--no-insert-timestamp`, so a rebuild of unchanged source gives the
  same extension bytes (and the same `cache_key()`).
- EIGEN_LDLT refuses an indefinite matrix: SINGULAR (= 1) when a D entry
  of the factorization is <= 0; an unsymmetric pattern is BAD_SHAPE.
- The DLL search on Windows: the package's own `vendor/<name>/bin` is
  searched first, before 0.1.11's list (an install by `pip install
  --target` or into a user site now finds its DLLs; 0.1.11's list had no
  package-relative entry); `SUITESPARSE_BIN` is added after them when set.
- Python 3.10 - 3.13: `python_requires >= 3.10`; Python 3.9 is DROPPED
  (numba 0.61, the first numba with Python 3.13, needs 3.10); requires
  numba >= 0.61 and numpy >= 1.24; wheels cp310 - cp313 (0.1.11's were
  cp39 - cp312).
- The Linux wheels are manylinux_2_28 (0.1.11's were manylinux2014):
  they need glibc >= 2.28 (Ubuntu 18.10+, Debian 10+, RHEL / AlmaLinux /
  Rocky 8+). Current numpy publishes no manylinux2014 wheels to build
  against; on an older glibc pip falls back to the sdist.
- The Eigen extension's licence text ships beside it in the wheel
  (sparse_numba/sparse_eigen/COPYING.MPL2 and COPYING.README, Eigen
  3.4.0's own files).
- The Windows wheels built by CI carry the UMFPACK interface (compiled
  against SuiteSparse's headers only; no SuiteSparse binary).

## Deprecated (kept through 0.2.x, removed at 0.3)

- `*_solve_factored(handle, b) -> (x, info)` (allocates x) in favour of
  `solve_into(handle, b, out)`;
- the `*_factorize_*` forms without an ordering in favour of
  `factorize(..., backend, ordering)`;
- `*_free_factors(handle)` in favour of `free(handle)`.

The deprecation is documented, not warned at run time (the functions are
njit-compiled).

## THE STATUS CODES: 0.1.11 TO 0.2.0 (the KEPT names' values)

The shapes of the kept functions did not change; some of the values their
`info` carries did. A caller that checks `info != 0` (the 0.1.11 contract)
is unaffected.

| case | 0.1.11 | 0.2.0 |
|---|---|---|
| `indptr[0] != 0` or `indptr[n] != nnz` (SuperLU / UMFPACK) | -1 / -2 | BAD_INDPTR = -3 |
| UMFPACK singular matrix | 1 | k > 0, the first zero of U's diagonal (1-based, UMFPACK's pivot order) |
| SuperLU singular matrix | k (dgstrf's info) | k (unchanged) |
| a raw UMFPACK status (any other negative) | the UMFPACK status itself | BACKEND_ERROR = -31 (UMFPACK's own status in `handle_info`'s last_status) |
| `info`'s dtype | intc (int32) | int64 |
| a double free / use after free | undefined (a crash) | HANDLE_FREED = -11 |

THE CODE MAPPING for the inputs 0.1.11 gave no code (one rule for every
backend):

| input | code |
|---|---|
| `len(indptr) != n + 1` | NOT_SQUARE = -5 |
| a wrong rank; a non-C-contiguous data / indices / indptr / b / out; `n <= 0`; `len(indices) != nnz`; a wrong-length b / out / data | BAD_SHAPE = -2 |
| a decreasing indptr | BAD_INDPTR = -3 |
| a row index outside [0, n) | BAD_INDEX = -4 |
| a NULL pointer reaching C | BAD_SHAPE = -2 |
| SuperLU dgstrf info > n (memory) | ALLOC_FAILED = -30 |
| SuperLU dgstrf info < 0 | BACKEND_ERROR = -31 |
| an unknown backend id at factorize | BACKEND_ABSENT = -20 |
| a handle whose backend byte is none of 1-4 | HANDLE_INVALID = -10 |
| Eigen's NumericalIssue; EIGEN_LDLT with a D entry <= 0 | SINGULAR = 1 |
| `out` aliasing `b` | not checked |

The full list is `sparse_numba.errors` (`explain(info)` says each in
words): OK 0, BAD_DTYPE -1, BAD_SHAPE -2, BAD_INDPTR -3, BAD_INDEX -4,
NOT_SQUARE -5, HANDLE_INVALID -10, HANDLE_FREED -11, HANDLE_WRONG_BACKEND
-12, BACKEND_ABSENT -20, ORDERING_UNSUPPORTED -21, ALLOC_FAILED -30,
BACKEND_ERROR -31; a positive k SINGULAR.

## Breaks (disclosed)

- The memory layout behind a handle: 0.1.11's handle was a pointer to the
  SuperLU factors struct, which a caller could read through ctypes; 0.2.0's
  handle is a registry index. `handle_info(handle)` gives n, the L / U
  nonzeros, the backend, the ordering, the last status and the pivot
  tolerance instead.

## Not in 0.2.0's wheels

- No SuiteSparse binary (above); no tests; no `__pycache__`, `*.nbi` or
  `*.nbc` file.
