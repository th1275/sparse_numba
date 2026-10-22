"""
The child of the njit-caller cache cell (run by test_njit_callers.py with
NUMBA_CACHE_DIR set; never collected by pytest).

    python njit_callers_child.py <out.npz>

A @njit(cache=True) caller defined here calls factorize, handle_info,
solve_into, solve_many_into (k = 3), refactorize (values x 1.5), solve_into,
free and solve_into after the free on SUPERLU (COLAMD) on the fixed 60x60
system, and writes the infos and each x to <out.npz>.
"""

import sys

import numpy as np
from numba import njit

import sparse_numba as sn

import _sn1_common as common

OUT = sys.argv[1]


@njit(cache=True, nogil=True)
def cached_caller(data, indices, indptr, n, backend, ordering, b, B, data2):
    infos = np.zeros(8, dtype=np.int64)
    x1 = np.empty(n)
    X = np.empty_like(B)
    x2 = np.empty(n)
    x3 = np.empty(n)
    handle, info = sn.factorize(data, indices, indptr, n, backend, ordering)
    infos[0] = info
    n_, nnz_l, nnz_u, backend_, ordering_, last_status, pivot_tol, info = sn.handle_info(handle)
    infos[1] = info
    infos[2] = sn.solve_into(handle, b, x1)
    infos[3] = sn.solve_many_into(handle, B, X)
    infos[4] = sn.refactorize(handle, data2)
    infos[5] = sn.solve_into(handle, b, x2)
    infos[6] = sn.free(handle)
    infos[7] = sn.solve_into(handle, b, x3)
    return infos, x1, X, x2


def main():
    a, b = common.fixed_system()
    n = a.shape[0]
    data, indices, indptr = common.dense_to_csc(a)
    rng = np.random.default_rng(common.SEED_FIXED + 1)
    B = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n, 3)))
    infos, x1, X, x2 = cached_caller(data, indices, indptr, n, sn.SUPERLU, sn.COLAMD,
                                     b, B, data * 1.5)
    np.savez(OUT, infos=infos, x1=x1, X=X, x2=x2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
