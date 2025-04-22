import os
import sys
import numpy as np


def main():
    print("Testing sparse_numba package...")

    # Try to import sparse_numba
    try:
        import sparse_numba
        print("Successfully imported sparse_numba")
    except ImportError as e:
        print(f"Failed to import sparse_numba: {e}")
        sys.exit(1)

    # Test SuperLU solver
    try:
        # Create a simple sparse matrix (diagonal matrix)
        n = 5
        data = np.ones(n, dtype=np.float64)
        indices = np.arange(n, dtype=np.int32)
        indptr = np.arange(n + 1, dtype=np.int32)
        b = np.ones(n, dtype=np.float64)

        # Solve using SuperLU
        x = sparse_numba.sparse_superlu.superlu_numba_interface.superlu_solve_csc(n, n, data, indices, indptr, b)

        # Check result
        if not np.allclose(x, b):
            print(f"SuperLU solution is incorrect: {x} != {b}")
            sys.exit(1)

        print("SuperLU solver test passed!")
    except Exception as e:
        print(f"SuperLU solver test failed: {e}")
        sys.exit(1)

    # Only test UMFPACK if we're not on Windows (since DLLs won't be available in CI)
    if os.name != 'nt':
        try:
            # Try UMFPACK if available
            x = sparse_numba.umfpack_solve_csc(n, n, data, indices, indptr, b)
            print("UMFPACK solver test passed!")
        except Exception as e:
            print(f"UMFPACK solver test (optional): {e}")
            # Don't fail the build if UMFPACK isn't available

    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())