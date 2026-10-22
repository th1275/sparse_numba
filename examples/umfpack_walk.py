"""
The life of a UMFPACK factorization through the backend-neutral API:
factorize -> handle_info -> solve_into -> refactorize -> free, at each of
UMFPACK's orderings.

UMFPACK is interface-only: sparse_numba ships no SuiteSparse binary, and
the UMFPACK backend loads the user's library at its first call
(SPARSE_NUMBA_UMFPACK_LIBRARY = the library's path; or, on Windows,
SUITESPARSE_BIN = the folder holding libumfpack.dll; or the platform's
loader path). When it is not found this script prints a note and stops.

    python examples/umfpack_walk.py
"""

import numpy as np

import sparse_numba as sn
from sparse_numba import errors


def grid_matrix(k, wind=0.5):
    """The 5-point Laplacian on a k x k grid plus an upwind convection term
    in x (wind > 0 makes it unsymmetric). Returns the CSC arrays (data,
    indices, indptr) and n."""
    n = k * k
    columns = [[] for _ in range(n)]
    for i in range(k):
        for j in range(k):
            p = i * k + j
            row = {p: 4.0 + wind}
            if j > 0:
                row[p - 1] = -1.0 - wind
            if j < k - 1:
                row[p + 1] = -1.0
            if i > 0:
                row[p - k] = -1.0
            if i < k - 1:
                row[p + k] = -1.0
            for q, value in row.items():
                columns[q].append((p, value))
    indptr = np.zeros(n + 1, dtype=np.int32)
    indices, data = [], []
    for q, entries in enumerate(columns):
        entries.sort()
        indices.extend(r for r, _ in entries)
        data.extend(v for _, v in entries)
        indptr[q + 1] = len(indices)
    return np.array(data), np.array(indices, dtype=np.int32), indptr, n


def residual(data, indices, indptr, x, b):
    y = np.zeros(len(b))
    for q in range(len(indptr) - 1):
        y[indices[indptr[q]:indptr[q + 1]]] += data[indptr[q]:indptr[q + 1]] * x[q]
    return np.max(np.abs(y - b)) / np.max(np.abs(b))


def main():
    if sn.UMFPACK not in sn.backends():
        print("UMFPACK: the user's library was not found, so the UMFPACK backend is absent.")
        print("Every UMFPACK call returns BACKEND_ABSENT (%d); require() says how to fix it:"
              % errors.BACKEND_ABSENT)
        try:
            sn.require(sn.UMFPACK)
        except sn.SparseNumbaError as e:
            print("  %s" % e)
        return

    print("UMFPACK found; cache_key() names its version: %s"
          % [p for p in sn.cache_key().split(";") if p.startswith("umfpack=")][0])
    data, indices, indptr, n = grid_matrix(40)
    print("matrix: %d x %d, %d nonzeros (a 40 x 40 grid, unsymmetric)" % (n, n, len(data)))
    b = np.ones(n)
    x = np.empty(n)
    for name in ("AMD", "COLAMD", "NATURAL", "MMD_AT_PLUS_A"):
        ordering = getattr(sn, name)
        handle, info = sn.factorize(data, indices, indptr, n, sn.UMFPACK, ordering)
        if info != 0:
            print("%-14s factorize: info %d (%s)" % (name, info, sn.explain(info)))
            continue
        nnz_l, nnz_u = sn.handle_info(handle)[1:3]
        info_solve = sn.solve_into(handle, b, x)
        r1 = residual(data, indices, indptr, x, b)
        data2 = data * 2.0
        info_re = sn.refactorize(handle, data2)
        sn.solve_into(handle, b, x)
        r2 = residual(data2, indices, indptr, x, b)
        info_free = sn.free(handle)
        print("%-14s nnz(L) %6d, nnz(U) %6d; solve_into info %d residual %.2e; "
              "refactorize info %d residual %.2e; free info %d"
              % (name, nnz_l, nnz_u, info_solve, r1, info_re, r2, info_free))


if __name__ == "__main__":
    main()
