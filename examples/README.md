# sparse_numba examples

Scripts to run and read, each printing its results (they are not tests).
Install sparse_numba first (`pip install sparse-numba`), then from the
repository's root:

```bash
python examples/superlu_walk.py
```

| script | what it shows |
|---|---|
| `superlu_walk.py` | the life of a factorization through the backend-neutral API with SuperLU: `factorize`, `handle_info`, `solve_into` (once, and 1000 times from a numba loop), `solve_many_into`, `refactorize` on new values, `free`, and the code a call on a freed handle returns |
| `umfpack_walk.py` | the same walk with UMFPACK at each of its orderings, when the user's SuiteSparse library is found (`SPARSE_NUMBA_UMFPACK_LIBRARY` or, on Windows, `SUITESPARSE_BIN`); otherwise a printed note and `require()`'s hint |
| `eigen_lu_ldlt.py` | Eigen's SparseLU (`EIGEN_LU`) at its orderings and SimplicialLDLT (`EIGEN_LDLT`) on a symmetric positive definite matrix, and the two inputs LDLT refuses by name (an unsymmetric pattern, an indefinite matrix) |
| `orderings_compare.py` | every backend at every ordering on one matrix, side by side: factorization time, fill (nnz(L) + nnz(U)), solve time, residual; the pairs a backend does not offer refused by name |
| `cache_two_processes.py` | an `@njit(cache=True)` function of yours that calls sparse_numba, compiled in one process and loaded from the numba cache in a second; `cache_key()` |
| `kept_0111_names.py` | the 0.1.11 names (`superlu_solve_csc` / `_coo` / `_csr`, `superlu_factorize_csc` / `superlu_solve_factored` / `superlu_free_factors`, the `umfpack_*` set, the conversions), unchanged in 0.2.0, their x compared with the new API's |

The matrices are built in each script (a convection-diffusion grid, or a
small hand-written one); nothing is downloaded. The times printed are the
machine's own, from a few repeats: an illustration, not a benchmark.

More: the [user guide](../docs/USER_GUIDE.md) and the
[0.2.0 release notes](../docs/RELEASE_NOTES_0.2.0.md).
