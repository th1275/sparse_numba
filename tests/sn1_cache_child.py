"""
The child of the cache round trip (run by
test_sn1_cache.py with NUMBA_CACHE_DIR set; never collected by pytest).

    python sn1_cache_child.py <backend> <out.npz>

Calls every public function of <backend> on the fixed 60x60 system, plus a
cache=True caller of factorize / solve_factored / free defined here (the
caller side), and writes each x and the walls to <out.npz>.
"""

import sys
import time

t_start = time.perf_counter()

import numpy as np
from numba import njit

import _sn1_common as common

BACKEND = sys.argv[1]
OUT = sys.argv[2]

if BACKEND == "superlu":
    from sparse_numba.sparse_superlu.superlu_numba_interface import (
        superlu_factorize_csc as _factorize_csc,
        superlu_solve_factored as _solve_factored,
        superlu_free_factors as _free_factors,
    )
else:
    from sparse_numba.sparse_umfpack.umfpack_numba_interface import (
        umfpack_factorize_csc as _factorize_csc,
        umfpack_solve_factored as _solve_factored,
        umfpack_free_factors as _free_factors,
    )


@njit(cache=True, nogil=True)
def caller_factor_solve_free(data, indices, indptr, b):
    handle, info = _factorize_csc(data, indices, indptr)
    if info != 0:
        return np.zeros_like(b), info
    x, info = _solve_factored(handle, b)
    info_free = _free_factors(handle)
    if info != 0:
        return x, info
    return x, info_free


def main():
    a, b = common.fixed_system()
    t_import = time.perf_counter() - t_start
    t0 = time.perf_counter()
    xs = common.run_all(BACKEND, a, b)
    data, indices, indptr = common.dense_to_csc(a)
    x, info = caller_factor_solve_free(data, indices, indptr, b)
    if info != 0:
        raise RuntimeError(f"caller_factor_solve_free: info {info}")
    xs["caller_factor_solve_free"] = np.asarray(x)
    t_calls = time.perf_counter() - t0
    np.savez(OUT, wall_import=np.float64(t_import), wall_calls=np.float64(t_calls),
             **{"x__" + k: v for k, v in xs.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
