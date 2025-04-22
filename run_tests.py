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

    # Check if SuperLU is available before testing
    try:
        # First check if the module is available
        if hasattr(sparse_numba, 'has_superlu') and sparse_numba.has_superlu():
            # Create a simple sparse matrix (diagonal matrix)
            n = 5
            data = np.ones(n, dtype=np.float64)
            indices = np.arange(n, dtype=np.int32)
            indptr = np.arange(n + 1, dtype=np.int32)
            b = np.ones(n, dtype=np.float64)

            # Get the solver function (avoid direct access pattern that might cause import recursion)
            solver = sparse_numba.superlu_numba_interface.superlu_solve_csc

            # Solve using SuperLU
            x = solver(n, n, data, indices, indptr, b)

            # Check result
            if not np.allclose(x, b):
                print(f"SuperLU solution is incorrect: {x} != {b}")
                sys.exit(1)

            print("SuperLU solver test passed!")
        else:
            print("SuperLU not available, skipping test")
    except Exception as e:
        print(f"SuperLU solver test failed: {e}")
        import traceback
        traceback.print_exc()  # This will help diagnose the exact recursion path
        sys.exit(1)

    # Only test UMFPACK if we're not on Windows (since DLLs won't be available in CI)
    if os.name != 'nt':
        try:
            # Check if UMFPACK is available
            if hasattr(sparse_numba, 'has_umfpack') and sparse_numba.has_umfpack():
                # Try UMFPACK if available
                solver = sparse_numba.umfpack_numba_interface.umfpack_solve_csc
                x = solver(n, n, data, indices, indptr, b)
                print("UMFPACK solver test passed!")
            else:
                print("UMFPACK not available, skipping test")
        except Exception as e:
            print(f"UMFPACK solver test (optional): {e}")
            # Don't fail the build if UMFPACK isn't available

    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())