"""
The numba cache across processes. A function of yours, compiled with
@njit(cache=True), calls sparse_numba's factorize / solve_into / free. The
first process compiles it and writes the cache; a second process loads it
from the cache instead of compiling -- sparse_numba's C entry points are
registered under stable names at import, so the cached code links in a new
process.

This script runs itself twice as a child process on one fresh cache
directory (NUMBA_CACHE_DIR) and prints the wall of each child's first call.
It also prints cache_key(): the string a caller folds into its own cache
key so that an upgrade of sparse_numba (or of its libraries, or the user's
UMFPACK) invalidates what the caller cached.

    python examples/cache_two_processes.py
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from numba import njit

import sparse_numba as sn


@njit(nogil=True, cache=True)
def tridiagonal_solve(n, backend, ordering):
    """Factorize the n x n tridiagonal (-1, 4, -1) matrix, solve it against
    ones, free; returns (x[0], info)."""
    nnz = 3 * n - 2
    data = np.empty(nnz)
    indices = np.empty(nnz, dtype=np.int32)
    indptr = np.empty(n + 1, dtype=np.int32)
    k = 0
    for q in range(n):
        indptr[q] = k
        for r in range(q - 1, q + 2):
            if 0 <= r < n:
                indices[k] = r
                data[k] = 4.0 if r == q else -1.0
                k += 1
    indptr[n] = k
    handle, info = sn.factorize(data, indices, indptr, n, backend, ordering)
    if info != 0:
        return 0.0, info
    b = np.ones(n)
    x = np.empty(n)
    info = sn.solve_into(handle, b, x)
    sn.free(handle)
    return x[0], info


def child():
    t0 = time.perf_counter()
    x0, info = tridiagonal_solve(1000, sn.SUPERLU, sn.COLAMD)
    first_call = time.perf_counter() - t0
    t0 = time.perf_counter()
    tridiagonal_solve(1000, sn.SUPERLU, sn.COLAMD)
    second_call = time.perf_counter() - t0
    print("  first call %.3f s; second call %.4f s; x[0] = %.6f, info %d"
          % (first_call, second_call, x0, info))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child()
        return
    cache_dir = tempfile.mkdtemp(prefix="sparse_numba_cache_example_")
    env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir)
    print("NUMBA_CACHE_DIR = %s (fresh)" % cache_dir)
    for label in ("process 1 (compiles, writes the cache)",
                  "process 2 (loads the cache)"):
        print(label)
        t0 = time.perf_counter()
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env,
                             capture_output=True, text=True)
        sys.stdout.write(out.stdout)
        if out.returncode != 0:
            sys.stdout.write(out.stderr)
            return
        files = sum(len(f) for _, _, f in os.walk(cache_dir))
        print("  the process's wall %.2f s; %d files in the cache directory"
              % (time.perf_counter() - t0, files))
    print("cache_key(): %s" % sn.cache_key())


if __name__ == "__main__":
    main()
