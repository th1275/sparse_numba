"""
The life of a SuperLU factorization through the backend-neutral API:
factorize -> handle_info -> solve_into (one right-hand side, and many from
a numba loop) -> solve_many_into -> refactorize (new values, the same
pattern) -> free, with the residual of every solution printed.

    python examples/superlu_walk.py
"""

import time

import numpy as np
from numba import njit

import sparse_numba as sn


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


def csc_matvec(data, indices, indptr, x):
    y = np.zeros(len(indptr) - 1)
    for q in range(len(indptr) - 1):
        y[indices[indptr[q]:indptr[q + 1]]] += data[indptr[q]:indptr[q + 1]] * x[q]
    return y


def residual(data, indices, indptr, x, b):
    return np.max(np.abs(csc_matvec(data, indices, indptr, x) - b)) / np.max(np.abs(b))


@njit(nogil=True)
def solve_rows(handle, rhs, out):
    """Solve every row of rhs into the same row of out; returns the OR of
    the codes (0 when every solve succeeded)."""
    status = 0
    for i in range(rhs.shape[0]):
        status |= sn.solve_into(handle, rhs[i], out[i])
    return status


def main():
    data, indices, indptr, n = grid_matrix(40)
    print("matrix: %d x %d, %d nonzeros (a 40 x 40 grid, unsymmetric)" % (n, n, len(data)))

    t0 = time.perf_counter()
    handle, info = sn.factorize(data, indices, indptr, n, sn.SUPERLU, sn.COLAMD)
    t1 = time.perf_counter()
    print("factorize(SUPERLU, COLAMD): info %d (%s), %.3f s (the first call compiles)"
          % (info, sn.explain(info), t1 - t0))
    if info != 0:
        return

    n_, nnz_l, nnz_u, backend, ordering, last, pivot_tol, info = sn.handle_info(handle)
    print("handle_info: n %d, nnz(L) %d, nnz(U) %d, backend %d, ordering %d, "
          "pivot tolerance %.1f" % (n_, nnz_l, nnz_u, backend, ordering, pivot_tol))

    b = np.ones(n)
    x = np.empty(n)
    info = sn.solve_into(handle, b, x)
    print("solve_into: info %d, residual %.2e" % (info, residual(data, indices, indptr, x, b)))

    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((1000, n))
    out = np.empty_like(rhs)
    solve_rows(handle, rhs[:1], out[:1])          # compile the loop once
    t0 = time.perf_counter()
    status = solve_rows(handle, rhs, out)
    t1 = time.perf_counter()
    worst = max(residual(data, indices, indptr, out[i], rhs[i]) for i in range(0, 1000, 100))
    print("1000 solve_into calls from a numba loop: status %d, %.1f us per solve, "
          "worst sampled residual %.2e" % (status, (t1 - t0) / 1000 * 1e6, worst))

    B = np.ascontiguousarray(rng.standard_normal((n, 8)))
    X = np.empty_like(B)
    info = sn.solve_many_into(handle, B, X)
    print("solve_many_into (8 columns in one call): info %d, residual of column 0 %.2e"
          % (info, residual(data, indices, indptr, X[:, 0].copy(), B[:, 0].copy())))

    data2 = data * 1.5
    info = sn.refactorize(handle, data2)
    info_solve = sn.solve_into(handle, b, x)
    print("refactorize (values x 1.5, the same pattern): info %d; solve_into: info %d, "
          "residual %.2e" % (info, info_solve, residual(data2, indices, indptr, x, b)))

    print("free: info %d" % sn.free(handle))
    info = sn.solve_into(handle, b, x)
    print("solve_into after free: info %d (%s)" % (info, sn.explain(info)))


if __name__ == "__main__":
    main()
