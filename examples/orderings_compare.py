"""
Every backend at every ordering on one matrix, side by side: the
factorization time, the fill (nnz(L) + nnz(U) from handle_info), the time
of one solve_into, and the residual. A pair the backend does not offer is
refused by name (ORDERING_UNSUPPORTED or BAD_SHAPE), never a silent
fallback. EIGEN_LDLT reads the lower triangle of a symmetric pattern: on
the unsymmetric matrix (a symmetric pattern, unsymmetric values) the
symmetric matrix that triangle defines is not positive definite, and it
returns SINGULAR. UMFPACK appears when the user's library is found.

The times are this machine's, single runs of a few repeats (the median
printed): an illustration, not a benchmark.

    python examples/orderings_compare.py
"""

import time

import numpy as np
from numba import njit

import sparse_numba as sn

BACKENDS = ("SUPERLU", "UMFPACK", "EIGEN_LU", "EIGEN_LDLT")
ORDERINGS = ("NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A")


def grid_matrix(k, wind=0.5):
    """The 5-point Laplacian on a k x k grid plus an upwind convection term
    in x (wind > 0 makes it unsymmetric; wind = 0 gives a symmetric positive
    definite matrix). Returns the CSC arrays (data, indices, indptr) and n."""
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


@njit(nogil=True)
def solve_many_times(handle, b, out, calls):
    status = 0
    for _ in range(calls):
        status |= sn.solve_into(handle, b, out)
    return status


def table(label, data, indices, indptr, n):
    print("\n%s: %d x %d, %d nonzeros" % (label, n, n, len(data)))
    print("%-11s %-14s %12s %10s %12s %10s" % ("backend", "ordering", "factorize ms",
                                              "fill", "solve us", "residual"))
    available = sn.backends()
    b = np.ones(n)
    x = np.empty(n)
    for backend_name in BACKENDS:
        backend = getattr(sn, backend_name)
        if backend not in available:
            print("%-11s %-14s %s" % (backend_name, "-", "absent (BACKEND_ABSENT)"))
            continue
        for ordering_name in ORDERINGS:
            walls = []
            info, handle = 0, 0
            for _ in range(3):
                t0 = time.perf_counter()
                handle, info = sn.factorize(data, indices, indptr, n, backend,
                                            getattr(sn, ordering_name))
                walls.append(time.perf_counter() - t0)
                if info != 0:
                    break
                if len(walls) < 3:
                    sn.free(handle)
            if info != 0:
                print("%-11s %-14s %s" % (backend_name, ordering_name,
                                          "info %d: %s" % (info, sn.explain(info).split(":")[0])))
                continue
            nnz_l, nnz_u = sn.handle_info(handle)[1:3]
            solve_many_times(handle, b, x, 1)
            t0 = time.perf_counter()
            solve_many_times(handle, b, x, 200)
            solve_us = (time.perf_counter() - t0) / 200 * 1e6
            r = residual(data, indices, indptr, x, b)
            sn.free(handle)
            print("%-11s %-14s %12.3f %10d %12.1f %10.1e"
                  % (backend_name, ordering_name, sorted(walls)[1] * 1e3, nnz_l + nnz_u,
                     solve_us, r))


def main():
    data, indices, indptr, n = grid_matrix(60, wind=0.5)
    table("an unsymmetric 60 x 60 grid matrix", data, indices, indptr, n)
    data, indices, indptr, n = grid_matrix(60, wind=0.0)
    table("its symmetric positive definite counterpart", data, indices, indptr, n)


if __name__ == "__main__":
    main()
