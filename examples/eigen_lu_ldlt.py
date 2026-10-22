"""
Eigen's two backends, compiled into sparse_numba (nothing to install):
EIGEN_LU (Eigen::SparseLU) on an unsymmetric matrix at each of its
orderings, and EIGEN_LDLT (Eigen::SimplicialLDLT, AMD ordering) on a
symmetric positive definite one -- with the two inputs LDLT refuses by
name: an unsymmetric pattern (BAD_SHAPE) and an indefinite matrix
(SINGULAR, a D entry <= 0).

    python examples/eigen_lu_ldlt.py
"""

import numpy as np

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


def residual(data, indices, indptr, x, b):
    y = np.zeros(len(b))
    for q in range(len(indptr) - 1):
        y[indices[indptr[q]:indptr[q + 1]]] += data[indptr[q]:indptr[q + 1]] * x[q]
    return np.max(np.abs(y - b)) / np.max(np.abs(b))


def walk(label, data, indices, indptr, n, backend, ordering):
    """factorize -> solve_into -> refactorize -> solve_into -> free, printed."""
    handle, info = sn.factorize(data, indices, indptr, n, backend, ordering)
    if info != 0:
        print("%-22s factorize: info %d (%s)" % (label, info, sn.explain(info)))
        return
    nnz_l, nnz_u = sn.handle_info(handle)[1:3]
    b = np.arange(1.0, n + 1.0)
    x = np.empty(n)
    info_solve = sn.solve_into(handle, b, x)
    r1 = residual(data, indices, indptr, x, b)
    data2 = data * 3.0
    info_re = sn.refactorize(handle, data2)
    sn.solve_into(handle, b, x)
    r2 = residual(data2, indices, indptr, x, b)
    print("%-22s nnz(L) %6d, nnz(U) %6d; solve residual %.2e; refactorize info %d, "
          "residual %.2e; free info %d"
          % (label, nnz_l, nnz_u, r1, info_re, r2, sn.free(handle)))


def main():
    print("Eigen available: %s" % sn.is_eigen_available())
    print("cache_key()'s eigen= entry: %s"
          % [p for p in sn.cache_key().split(";") if p.startswith("eigen=")][0])

    data, indices, indptr, n = grid_matrix(40, wind=0.5)
    print("\nEIGEN_LU on an unsymmetric %d x %d matrix (%d nonzeros):" % (n, n, len(data)))
    for name in ("NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A"):
        walk("EIGEN_LU %s" % name, data, indices, indptr, n, sn.EIGEN_LU, getattr(sn, name))

    sdata, sindices, sindptr, sn_n = grid_matrix(40, wind=0.0)
    print("\nEIGEN_LDLT on a symmetric positive definite %d x %d matrix (%d nonzeros):"
          % (sn_n, sn_n, len(sdata)))
    for name in ("AMD", "COLAMD"):
        walk("EIGEN_LDLT %s" % name, sdata, sindices, sindptr, sn_n, sn.EIGEN_LDLT,
             getattr(sn, name))

    print("\nWhat EIGEN_LDLT refuses:")
    # [[4, 1, 0], [0, 4, 1], [0, 0, 4]]: the entries above the diagonal have
    # no mirror below it
    walk("unsymmetric pattern", np.array([4.0, 1.0, 4.0, 1.0, 4.0]),
         np.array([0, 0, 1, 1, 2], dtype=np.int32), np.array([0, 1, 3, 5], dtype=np.int32),
         3, sn.EIGEN_LDLT, sn.AMD)
    # the grid matrix above has a symmetric PATTERN with unsymmetric values:
    # LDLT checks the pattern only and reads the lower triangle, so it
    # factors the symmetric matrix that triangle defines -- here not
    # positive definite, so a D entry <= 0
    walk("unsymmetric values", data, indices, indptr, n, sn.EIGEN_LDLT, sn.AMD)
    # the same SPD pattern with the diagonal negated in its second half: indefinite
    indefinite = sdata.copy()
    for q in range(sn_n // 2, sn_n):
        for pos in range(sindptr[q], sindptr[q + 1]):
            if sindices[pos] == q:
                indefinite[pos] = -indefinite[pos]
    walk("indefinite matrix", indefinite, sindices, sindptr, sn_n, sn.EIGEN_LDLT, sn.AMD)


if __name__ == "__main__":
    main()
