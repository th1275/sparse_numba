"""
Producer of tests/data/oracle_0111.npz: x from the 0.1.11 ctypes path
(run once at the 0.1.11 src).

Keys: <backend>__<function key> for the fixed 60x60 system (the cache
round trip's) through every public function of each backend, and
n69_superlu__<key> for the 69-bus G at SuperLU NATURAL. Prints the npz's
sha256 and, for information, n69 x against the vehicle's x_A1.txt.

Usage: .venv311/Scripts/python.exe tests/make_oracle_0111.py
"""

import hashlib
import os
import sys
import time

import numpy as np

import _sn1_common as common

sys.path.insert(0, common.REPO_DIR)


def main():
    t0 = time.perf_counter()
    a, b = common.fixed_system()
    arrays = {}
    for backend in common.BACKENDS:
        for key, x in common.run_all(backend, a, b).items():
            arrays[f"{backend}__{key}"] = x
    for key, x in common.run_n69_superlu().items():
        arrays[f"n69_superlu__{key}"] = x

    os.makedirs(os.path.dirname(common.ORACLE_PATH), exist_ok=True)
    np.savez(common.ORACLE_PATH, **arrays)
    with open(common.ORACLE_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(common.N69_DIR, "x_A1.txt"), "r", encoding="ascii") as f:
        x_a1 = np.array([float(t) for t in f.read().split()], dtype=np.float64)
    for key in ("solve_csc", "factored_csc"):
        x = arrays[f"n69_superlu__{key}"]
        diff = float(np.max(np.abs(x - x_a1))) if x.shape == x_a1.shape else float("nan")
        print(f"n69 {key} vs x_A1.txt: bit-equal {common.bits_equal(x, x_a1)}, max |diff| {diff:.3e}")

    print(f"keys {len(arrays)}: {', '.join(sorted(arrays))}")
    print(f"oracle {common.ORACLE_PATH}")
    print(f"sha256 {sha}")
    print(f"wall {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
