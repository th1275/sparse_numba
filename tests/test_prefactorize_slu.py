"""
Tests for SuperLU pre-factorization API.
Tests factorize once / solve many times pattern.
"""

import numpy as np
import scipy.sparse as sp
from sparse_numba.sparse_superlu.superlu_numba_interface import (
    superlu_solve_csc,
    superlu_factorize_csc,
    superlu_factorize_coo,
    superlu_factorize_csr,
    superlu_solve_factored,
    superlu_free_factors,
)
from sparse_numba.conversion.matrix_conversion_numba import (
    convert_coo_to_csr,
    sparse_matvec_csr,
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
    x_direct, info_direct = superlu_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )
    assert info_direct == 0, f"Direct solve failed: info={info_direct}"

    # Pre-factored solve
    handle, info_fact = superlu_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info_fact == 0, f"Factorize failed: info={info_fact}"
    assert handle != 0, "Handle should be nonzero"

    x_factored, info_solve = superlu_solve_factored(handle, b)
    assert info_solve == 0, f"Factored solve failed: info={info_solve}"

    # Results should match
    err = np.linalg.norm(x_factored - x_direct)
    print(f"  Error vs direct: {err:.2e}")
    assert err < 1e-10, f"Results differ: error={err}"

    # Cleanup
    info_free = superlu_free_factors(handle)
    assert info_free == 0, f"Free failed: info={info_free}"
    print("  PASSED")


def test_multiple_solves():
    """Test factorize once, solve with multiple RHS vectors."""
    print("Test: multiple solves with same factors")
    A_coo = _make_test_matrix()
    A_csc = A_coo.tocsc()
    n = A_csc.shape[0]

    handle, info = superlu_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info == 0

    # Solve with 5 different RHS vectors
    np.random.seed(123)
    for i in range(5):
        x_true = np.random.randn(n)
        b = A_csc @ x_true

        x_solved, info = superlu_solve_factored(handle, b)
        assert info == 0

        err = np.linalg.norm(x_solved - x_true) / np.linalg.norm(x_true)
        print(f"  RHS {i}: relative error = {err:.2e}")
        assert err < 1e-10, f"Solution {i} has large error: {err}"

    superlu_free_factors(handle)
    print("  PASSED")


def test_factorize_coo():
    """Test COO format factorization."""
    print("Test: factorize from COO format")
    A_coo = _make_test_matrix()
    A_csc = A_coo.tocsc()
    n = A_csc.shape[0]

    x_true = np.ones(n)
    b = A_csc @ x_true

    handle, info = superlu_factorize_coo(
        A_coo.row.astype(np.int32),
        A_coo.col.astype(np.int32),
        A_coo.data,
        A_coo.shape,
    )
    assert info == 0, f"COO factorize failed: info={info}"

    x_solved, info = superlu_solve_factored(handle, b)
    assert info == 0

    err = np.linalg.norm(x_solved - x_true)
    print(f"  Error: {err:.2e}")
    assert err < 1e-8, f"COO factorize solution error: {err}"

    superlu_free_factors(handle)
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

    handle, info = superlu_factorize_csr(
        A_csr.data, A_csr.indices.astype(np.int32), A_csr.indptr.astype(np.int32)
    )
    assert info == 0, f"CSR factorize failed: info={info}"

    x_solved, info = superlu_solve_factored(handle, b)
    assert info == 0

    err = np.linalg.norm(x_solved - x_true)
    print(f"  Error: {err:.2e}")
    assert err < 1e-8, f"CSR factorize solution error: {err}"

    superlu_free_factors(handle)
    print("  PASSED")


def test_free_factors():
    """Test that free_factors doesn't crash."""
    print("Test: free factors")
    A_coo = _make_test_matrix(n=10)
    A_csc = A_coo.tocsc()

    handle, info = superlu_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info == 0

    info_free = superlu_free_factors(handle)
    assert info_free == 0, f"Free failed: info={info_free}"
    print("  PASSED")


def test_sparse_matvec_csr():
    """Test sparse matrix-vector product."""
    print("Test: sparse_matvec_csr")
    n = 50
    np.random.seed(42)
    A_dense = np.random.randn(n, n)
    A_dense[np.abs(A_dense) < 1.0] = 0.0  # Sparsify
    A_csr = sp.csr_matrix(A_dense)

    x = np.random.randn(n)

    # Reference: dense matmul
    y_ref = A_dense @ x

    # Sparse matvec
    y_sparse = sparse_matvec_csr(
        A_csr.data, A_csr.indices.astype(np.int32),
        A_csr.indptr.astype(np.int32), x
    )

    err = np.linalg.norm(y_sparse - y_ref)
    print(f"  Error: {err:.2e}")
    assert err < 1e-10, f"sparse_matvec_csr error: {err}"
    print("  PASSED")


def test_convert_coo_to_csr():
    """Test COO to CSR conversion against scipy."""
    print("Test: convert_coo_to_csr")
    n = 30
    np.random.seed(42)
    A_coo = sp.random(n, n, density=0.2, format='coo', dtype=np.float64)

    # scipy reference
    A_csr_ref = A_coo.tocsr()

    # Our conversion
    data, indices, indptr = convert_coo_to_csr(
        A_coo.row.astype(np.int32),
        A_coo.col.astype(np.int32),
        A_coo.data,
        n, n
    )

    # Build scipy CSR from our output for comparison
    A_csr_ours = sp.csr_matrix((data, indices, indptr), shape=(n, n))

    # Compare dense representations
    diff = np.abs(A_csr_ref.toarray() - A_csr_ours.toarray()).max()
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-14, f"convert_coo_to_csr error: {diff}"
    print("  PASSED")


def test_comparison_with_direct_solve():
    """Verify factored solve matches direct solve exactly."""
    print("Test: factored vs direct solve comparison")
    n = 200
    A_coo = _make_test_matrix(n=n, density=0.02)
    A_csc = A_coo.tocsc()

    np.random.seed(99)
    b = np.random.randn(n)

    # Direct
    x_direct, info1 = superlu_solve_csc(
        A_csc.data, A_csc.indices, A_csc.indptr, b
    )
    assert info1 == 0

    # Factored
    handle, info2 = superlu_factorize_csc(
        A_csc.data, A_csc.indices, A_csc.indptr
    )
    assert info2 == 0

    x_factored, info3 = superlu_solve_factored(handle, b)
    assert info3 == 0

    err = np.linalg.norm(x_factored - x_direct)
    print(f"  Error between direct and factored: {err:.2e}")
    assert err < 1e-10

    superlu_free_factors(handle)
    print("  PASSED")


def run_all_tests():
    print("=" * 60)
    print("SuperLU Pre-Factorization Tests")
    print("=" * 60)
    test_basic_factorize_and_solve()
    test_multiple_solves()
    test_factorize_coo()
    test_factorize_csr()
    test_free_factors()
    test_sparse_matvec_csr()
    test_convert_coo_to_csr()
    test_comparison_with_direct_solve()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
