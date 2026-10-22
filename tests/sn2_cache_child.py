"""
The child of the API cache round trip (run by test_sn2_api.py with
NUMBA_CACHE_DIR set; never collected by pytest).

    python sn2_cache_child.py <backend> <out.npz>

Calls factorize / solve_into / solve_many_into / handle_info / refactorize /
free and solve of sparse_numba.api on the fixed 60x60 system (its SPD form
for eigen_ldlt) at the backend's reference ordering, plus a cache=True caller of factorize / solve_into
/ free defined here, and writes each x and the walls to <out.npz>.
"""

import sys
import time

t_start = time.perf_counter()

import numpy as np
from numba import njit

import _sn1_common as common
import _sn2_common as c2
import sparse_numba.api as api

BACKEND = sys.argv[1]
OUT = sys.argv[2]
BE, ORDER = c2.api_ids(api, BACKEND)


@njit(cache=True, nogil=True)
def caller_factor_solve_free(data, indices, indptr, n, b, backend, ordering):
    x = np.zeros(n)
    handle, info = api.factorize(data, indices, indptr, n, backend, ordering)
    if info != 0:
        return x, info
    info = api.solve_into(handle, b, x)
    info_free = api.free(handle)
    if info != 0:
        return x, info
    return x, info_free


def _check(what, info):
    if info != 0:
        raise RuntimeError(f"{BACKEND} {what}: info {info}")


def main():
    a, b = c2.system_for(BACKEND)
    data, indices, indptr = common.dense_to_csc(a)
    n = a.shape[0]
    t_import = time.perf_counter() - t_start
    t0 = time.perf_counter()
    out = {}
    handle, info = api.factorize(data, indices, indptr, n, BE, ORDER)
    _check("factorize", info)
    x = np.empty(n)
    _check("solve_into", api.solve_into(handle, b, x))
    out["x__solve_into"] = x
    B = np.ascontiguousarray(np.stack([b, 2.0 * b], axis=1))
    M = np.empty((n, 2))
    _check("solve_many_into", api.solve_many_into(handle, B, M))
    out["m__solve_many_into"] = M
    _check("handle_info", api.handle_info(handle)[-1])
    _check("refactorize", api.refactorize(handle, data))
    xr = np.empty(n)
    _check("solve_into after refactorize", api.solve_into(handle, b, xr))
    out["x__refactorized"] = xr
    _check("free", api.free(handle))
    x1, info = api.solve(data, indices, indptr, n, b, BE, ORDER)
    _check("solve", info)
    out["x__solve"] = x1
    xc, info = caller_factor_solve_free(data, indices, indptr, n, b, BE, ORDER)
    _check("caller_factor_solve_free", info)
    out["x__caller"] = xc
    t_calls = time.perf_counter() - t0
    np.savez(OUT, wall_import=np.float64(t_import), wall_calls=np.float64(t_calls), **out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
