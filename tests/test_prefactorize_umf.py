"""
Tests for UMFPACK pre-factorization API.
Tests factorize once / solve many times pattern.
"""

import numpy as np
import scipy.sparse as sp
from sparse_numba.sparse_umfpack.umfpack_numba_interface import (
    umfpack_solve_csc,
    umfpack_factorize_csc,
    umfpack_factorize_coo,
    umfpack_factorize_csr,
    umfpack_solve_factored,
    umfpack_free_factors,
)


def _make_test_matrix(n=100, density=0.05, seed=42):
    """Create a well-conditioned sparse test matrix."""
    np.random.seed(seed)
    A_coo = sp.random(n, n, density=density, format='coo', dtype=np.float64)
    # Add diagonal dominance
    diag = np.arange(n)
    A_coo.data = np.concatenate([A_coo.data, np.ones(n) * 10.0])
    A_coo.row = np.concatenate([A_coo.row, diag])
    A_coo.col = np.concatenate([A_coo.col, diag])
    return A_coo


def test_basic_factorize_and_solve():
    """Test basic factorize + solve matches direct solve."""
    print("Test: basic factorize + solve")
    A_coo = _make_test_matrix()
    A_csc = A_coo.tocsc()

    x_true = np.ones(A_csc.shape[0])
    b = A_csc @ x_true

    # Direct solve
    x_direct, info_direct = umfpack_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )
    assert info_direct == 0, f"Direct solve failed: info={info_direct}"

    # Pre-factored solve
    handle, info_fact = umfpack_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info_fact == 0, f"Factorize failed: info={info_fact}"
    assert handle != 0, "Handle should be nonzero"

    x_factored, info_solve = umfpack_solve_factored(handle, b)
    assert info_solve == 0, f"Factored solve failed: info={info_solve}"

    err = np.linalg.norm(x_factored - x_direct)
    print(f"  Error vs direct: {err:.2e}")
    assert err < 1e-10, f"Results differ: error={err}"

    info_free = umfpack_free_factors(handle)
    assert info_free == 0
    print("  PASSED")


def test_multiple_solves():
    """Test factorize once, solve with multiple RHS vectors."""
    print("Test: multiple solves with same factors")
    A_coo = _make_test_matrix()
    A_csc = A_coo.tocsc()
    n = A_csc.shape[0]

    handle, info = umfpack_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info == 0

    np.random.seed(123)
    for i in range(5):
        x_true = np.random.randn(n)
        b = A_csc @ x_true

        x_solved, info = umfpack_solve_factored(handle, b)
        assert info == 0

        err = np.linalg.norm(x_solved - x_true) / np.linalg.norm(x_true)
        print(f"  RHS {i}: relative error = {err:.2e}")
        assert err < 1e-10, f"Solution {i} has large error: {err}"

    umfpack_free_factors(handle)
    print("  PASSED")


def test_factorize_coo():
    """Test COO format factorization."""
    print("Test: factorize from COO format")
    A_coo = _make_test_matrix()
    A_csc = A_coo.tocsc()
    n = A_csc.shape[0]

    x_true = np.ones(n)
    b = A_csc @ x_true

    handle, info = umfpack_factorize_coo(
        A_coo.row.astype(np.int32),
        A_coo.col.astype(np.int32),
        A_coo.data,
        A_coo.shape,
    )
    assert info == 0, f"COO factorize failed: info={info}"

    x_solved, info = umfpack_solve_factored(handle, b)
    assert info == 0

    err = np.linalg.norm(x_solved - x_true)
    print(f"  Error: {err:.2e}")
    assert err < 1e-8

    umfpack_free_factors(handle)
    print("  PASSED")


def test_factorize_csr():
    """Test CSR format factorization."""
    print("Test: factorize from CSR format")
    A_coo = _make_test_matrix()
    A_csr = A_coo.tocsr()
    A_csc = A_coo.tocsc()
    n = A_csc.shape[0]

    x_true = np.ones(n)
    b = A_csc @ x_true

    handle, info = umfpack_factorize_csr(
        A_csr.data, A_csr.indices.astype(np.int32), A_csr.indptr.astype(np.int32)
    )
    assert info == 0, f"CSR factorize failed: info={info}"

    x_solved, info = umfpack_solve_factored(handle, b)
    assert info == 0

    err = np.linalg.norm(x_solved - x_true)
    print(f"  Error: {err:.2e}")
    assert err < 1e-8

    umfpack_free_factors(handle)
    print("  PASSED")


def test_free_factors():
    """Test that free_factors doesn't crash."""
    print("Test: free factors")
    A_coo = _make_test_matrix(n=10)
    A_csc = A_coo.tocsc()

    handle, info = umfpack_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info == 0

    info_free = umfpack_free_factors(handle)
    assert info_free == 0
    print("  PASSED")


def test_comparison_with_direct_solve():
    """Verify factored solve matches direct solve exactly."""
    print("Test: factored vs direct solve comparison")
    n = 200
    A_coo = _make_test_matrix(n=n, density=0.02)
    A_csc = A_coo.tocsc()

    np.random.seed(99)
    b = np.random.randn(n)

    x_direct, info1 = umfpack_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )
    assert info1 == 0

    handle, info2 = umfpack_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info2 == 0

    x_factored, info3 = umfpack_solve_factored(handle, b)
    assert info3 == 0

    err = np.linalg.norm(x_factored - x_direct)
    print(f"  Error between direct and factored: {err:.2e}")
    assert err < 1e-10

    umfpack_free_factors(handle)
    print("  PASSED")


def run_all_tests():
    print("=" * 60)
    print("UMFPACK Pre-Factorization Tests")
    print("=" * 60)
    test_basic_factorize_and_solve()
    test_multiple_solves()
    test_factorize_coo()
    test_factorize_csr()
    test_free_factors()
    test_comparison_with_direct_solve()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
