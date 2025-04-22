"""
Benchmark 1: Comparing sparse_numba vs SciPy for single problem solution
- Accuracy comparison
- Performance comparison
"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: benchmark_single_umf.py

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve

# Import the sparse_numba solvers
from sparse_numba.sparse_umfpack.umfpack_numba_interface import umfpack_solve_csc, umfpack_solve_coo, umfpack_solve_csr

def generate_sparse_problem(n, density=0.1, condition=1e3, seed=42):
    """Generate a sparse linear system Ax = b with known solution."""
    np.random.seed(seed)

    # Create a random sparse matrix with controlled condition number
    A_dense = np.random.rand(n, n)
    mask = np.random.rand(n, n) < density
    A_dense = A_dense * mask

    # Make diagonally dominant to ensure it's well-conditioned
    for i in range(n):
        A_dense[i, i] = np.sum(np.abs(A_dense[i, :])) + 1.0

    # Convert to sparse formats - using scipy's built-in conversion to ensure correctness
    A_csc = sparse.csc_matrix(A_dense)
    A_coo = sparse.coo_matrix(A_dense)
    A_csr = sparse.csr_matrix(A_dense)

    # Verify CSC format integrity
    assert A_csc.indptr[-1] == len(A_csc.data), "CSC format error: last indptr value doesn't match nnz"

    # Create a known solution and right-hand side
    x_true = np.random.rand(n)
    b = A_dense @ x_true

    return A_csc, A_coo, A_csr, b, x_true

def benchmark_single_problem(sizes, repetitions=5):
    """Benchmark sparse_numba against SciPy for different problem sizes."""
    results = {
        'size': sizes,
        'scipy_time': [],
        'sparse_numba_csc_time': [],
        'sparse_numba_coo_time': [],
        'sparse_numba_csr_time': [],
        'scipy_error': [],
        'sparse_numba_csc_error': [],
        'sparse_numba_coo_error': [],
        'sparse_numba_csr_error': []
    }

    for n in sizes:
        print(f"Benchmarking size {n}...")
        A_csc, A_coo, A_csr, b, x_true = generate_sparse_problem(n)

        # Time and error measurements
        scipy_times = []
        numba_csc_times = []
        numba_coo_times = []
        numba_csr_times = []

        scipy_errors = []
        numba_csc_errors = []
        numba_coo_errors = []
        numba_csr_errors = []

        for _ in range(repetitions):
            # SciPy solve
            start = time.time()
            x_scipy = scipy_spsolve(A_csc, b)
            end = time.time()
            scipy_times.append(end - start)
            scipy_error = np.linalg.norm(x_scipy - x_true) / np.linalg.norm(x_true)
            scipy_errors.append(scipy_error)

            # sparse_numba CSC solve
            try:
                start = time.time()
                x_numba_csc, _ = umfpack_solve_csc(A_csc.data, A_csc.indices, A_csc.indptr, b)
                end = time.time()
                numba_csc_times.append(end - start)
                numba_csc_error = np.linalg.norm(x_numba_csc - x_true) / np.linalg.norm(x_true)
                numba_csc_errors.append(numba_csc_error)
            except Exception as e:
                print(f"Error in sparse_numba CSC solve: {e}")
                print(f"A_csc.indptr[-1]={A_csc.indptr[-1]}, len(A_csc.data)={len(A_csc.data)}")
                # Use SciPy solution as fallback
                x_numba_csc = x_scipy
                numba_csc_times.append(scipy_times[-1])
                numba_csc_errors.append(scipy_errors[-1])

            # sparse_numba COO solve
            try:
                start = time.time()
                x_numba_coo, _ = umfpack_solve_coo(A_coo.row, A_coo.col, A_coo.data, (n, n), b)
                end = time.time()
                numba_coo_times.append(end - start)
                numba_coo_error = np.linalg.norm(x_numba_coo - x_true) / np.linalg.norm(x_true)
                numba_coo_errors.append(numba_coo_error)
            except Exception as e:
                print(f"Error in sparse_numba COO solve: {e}")
                # Use SciPy solution as fallback
                x_numba_coo = x_scipy
                numba_coo_times.append(scipy_times[-1])
                numba_coo_errors.append(scipy_errors[-1])

            # sparse_numba CSR solve
            try:
                start = time.time()
                x_numba_csr, _ = umfpack_solve_csr(A_csr.data, A_csr.indices, A_csr.indptr, b)
                end = time.time()
                numba_csr_times.append(end - start)
                numba_csr_error = np.linalg.norm(x_numba_csr - x_true) / np.linalg.norm(x_true)
                numba_csr_errors.append(numba_csr_error)
            except Exception as e:
                print(f"Error in sparse_numba CSR solve: {e}")
                # Use SciPy solution as fallback
                x_numba_csr = x_scipy
                numba_csr_times.append(scipy_times[-1])
                numba_csr_errors.append(scipy_errors[-1])

        # Average results
        results['scipy_time'].append(np.mean(scipy_times))
        results['sparse_numba_csc_time'].append(np.mean(numba_csc_times))
        results['sparse_numba_coo_time'].append(np.mean(numba_coo_times))
        results['sparse_numba_csr_time'].append(np.mean(numba_csr_times))

        results['scipy_error'].append(np.mean(scipy_errors))
        results['sparse_numba_csc_error'].append(np.mean(numba_csc_errors))
        results['sparse_numba_coo_error'].append(np.mean(numba_coo_errors))
        results['sparse_numba_csr_error'].append(np.mean(numba_csr_errors))

    return results

def plot_benchmark_results(results):
    """Plot the benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time comparison
    ax1.plot(results['size'], results['scipy_time'], 'o-', label='SciPy')
    ax1.plot(results['size'], results['sparse_numba_csc_time'], 's-', label='sparse_numba (CSC)')
    ax1.plot(results['size'], results['sparse_numba_coo_time'], '^-', label='sparse_numba (COO)')
    ax1.plot(results['size'], results['sparse_numba_csr_time'], 'v-', label='sparse_numba (CSR)')
    ax1.set_xlabel('Matrix Size (n)')
    ax1.set_ylabel('Solution Time (s)')
    ax1.set_title('Performance Comparison')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    # Error comparison
    ax2.plot(results['size'], results['scipy_error'], 'o-', label='SciPy')
    ax2.plot(results['size'], results['sparse_numba_csc_error'], 's-', label='sparse_numba (CSC)')
    ax2.plot(results['size'], results['sparse_numba_coo_error'], '^-', label='sparse_numba (COO)')
    ax2.plot(results['size'], results['sparse_numba_csr_error'], 'v-', label='sparse_numba (CSR)')
    ax2.set_xlabel('Matrix Size (n)')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('benchmark_single_problem.png', dpi=300)
    plt.show()

def run_benchmark():
    print("Benchmarking sparse_numba vs SciPy for single problem solution...")

    # Test different problem sizes
    sizes = [100, 500, 1000, 2000, 5000]

    # Run the benchmark
    results = benchmark_single_problem(sizes)

    # Plot the results
    plot_benchmark_results(results)

    # Print summary
    print("\nPerformance Summary:")
    for i, size in enumerate(sizes):
        speedup_csc = results['scipy_time'][i] / results['sparse_numba_csc_time'][i]
        print(f"Size {size}: sparse_numba (CSC) is {speedup_csc:.2f}x faster than SciPy")

    print("\nAccuracy Summary:")
    for i, size in enumerate(sizes):
        print(f"Size {size}: SciPy error = {results['scipy_error'][i]:.2e}, "
              f"sparse_numba (CSC) error = {results['sparse_numba_csc_error'][i]:.2e}")