"""
Benchmark 2: Demonstrating the advantages of sparse_numba's Numba compatibility
- Focus on solving multiple sparse systems with data exchange between iterations
- Compare sequential SciPy approach vs. parallel sparse_numba with nogil

"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: benchmark_parallel_umf.py

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve
from numba.typed import List
from numba import njit, prange
from numba.core import types

# Import the sparse_numba solvers
from sparse_numba.sparse_umfpack.umfpack_numba_interface import umfpack_solve_csc


# Configure Numba to use all available CPU cores
# import multiprocessing
# cpu_count = multiprocessing.cpu_count()
# numba.set_num_threads(cpu_count)
# print(f"Configured Numba to use {cpu_count} threads")

def generate_multiple_sparse_problems(num_problems, n, density=0.1, seed=42):
    """Generate multiple sparse linear systems with varying coefficients."""
    np.random.seed(seed)

    A_list = []
    b_list = []
    x_true_list = []

    for i in range(num_problems):
        # Create a random sparse matrix
        A_dense = np.random.rand(n, n)
        mask = np.random.rand(n, n) < density
        A_dense = A_dense * mask

        # Make diagonally dominant
        for j in range(n):
            A_dense[j, j] = np.sum(np.abs(A_dense[j, :])) + 1.0 + 0.1 * i  # Slight variation between problems

        # Convert to CSC format
        A_csc = sparse.csc_matrix(A_dense)

        # Create a known solution and right-hand side
        x_true = np.random.rand(n) * (1.0 + 0.05 * i)  # Slight variation between problems
        b = A_dense @ x_true

        A_list.append(A_csc)
        b_list.append(b)
        x_true_list.append(x_true)

    return A_list, b_list, x_true_list


def sequential_scipy_solver(A_list, b_list):
    """Solve multiple sparse systems sequentially with SciPy."""
    num_problems = len(A_list)
    n = A_list[0].shape[0]

    # Initialize solutions array
    solutions = np.zeros((num_problems, n))

    # Solve each system sequentially
    for i in range(num_problems):
        solutions[i] = scipy_spsolve(A_list[i], b_list[i])

    return solutions


@njit(nogil=True, parallel=True)
def parallel_sparse_numba_solver(A_data_list, A_indices_list, A_indptr_list, b_list, n):
    """Solve multiple sparse systems in parallel with sparse_numba."""
    num_problems = len(A_data_list)

    # Initialize solutions array
    solutions = np.zeros((num_problems, n), dtype=np.float64)

    # Solve each system in parallel
    for i in prange(num_problems):
        # Extract the sparse matrix data for this problem
        # data = A_data_list[i].astype(np.float64)
        data = A_data_list[i]
        indices = A_indices_list[i].astype(np.int32)
        indptr = A_indptr_list[i].astype(np.int32)
        b = b_list[i].astype(np.float64)

        # Call sparse_numba solver - this is the key part that needs to be Numba-compatible
        x, info = umfpack_solve_csc(data, indices, indptr, b)

        # Store the solution
        solutions[i] = x

    return solutions


def benchmark_solvers(num_problems_list, n, repeat=3):
    """
    Benchmark both approaches (sequential SciPy vs. parallel sparse_numba) for different numbers of problems.
    """
    results = {
        'num_problems': num_problems_list,
        'scipy_time': [],
        'sparse_numba_time': []
    }

    for num_problems in num_problems_list:
        print(f"Benchmarking with {num_problems} problems...")

        # Generate the test problems
        A_list, b_list, x_true_list = generate_multiple_sparse_problems(num_problems, n)

        # Prepare data for sparse_numba
        A_data_list = [A.data for A in A_list]
        A_indices_list = [A.indices for A in A_list]
        A_indptr_list = [A.indptr for A in A_list]

        # # Convert lists to arrays for Numba
        # A_data_arr = np.array(A_data_list, dtype=np.float64)
        # A_indices_arr = np.array(A_indices_list, dtype=np.int64)
        # A_indptr_arr = np.array(A_indptr_list, dtype=np.int64)
        # b_arr = np.array(b_list)

        # Create empty typed lists with type specification
        A_data_arr = List.empty_list(types.float64[:])  # For 1D arrays of float64
        A_indices_arr = List.empty_list(types.int64[:])  # For 1D arrays of int32
        A_indptr_arr = List.empty_list(types.int64[:])  # For 1D arrays of int32
        b_arr = List.empty_list(types.float64[:])  # For 1D arrays of float64

        # Populate the typed lists, ensuring each element is converted to a numpy array
        # with the desired dtype (optional if they are already numpy arrays).
        for d in A_data_list:
            A_data_arr.append(np.asarray(d, dtype=np.float64))

        for idx in A_indices_list:
            A_indices_arr.append(np.asarray(idx, dtype=np.int64))

        for ptr in A_indptr_list:
            A_indptr_arr.append(np.asarray(ptr, dtype=np.int64))

        for bb in b_list:
            b_arr.append(np.asarray(bb, dtype=np.float64))

        # Run SciPy benchmark multiple times and take the best time
        scipy_times = []
        for r in range(repeat):
            start = time.time()
            x_scipy = sequential_scipy_solver(A_list, b_list)
            end = time.time()
            scipy_times.append(end - start)

        scipy_time = min(scipy_times)  # Use the best time
        print(f"  SciPy time: {scipy_time:.4f}s")

        # Compile the Numba function with a small problem first (to avoid including compilation time)
        if num_problems == num_problems_list[0]:
            _ = parallel_sparse_numba_solver(
                A_data_arr[:1], A_indices_arr[:1], A_indptr_arr[:1], b_arr[:1], n
            )
            print("  Numba function compiled")

        # Run sparse_numba benchmark multiple times and take the best time
        sparse_numba_times = []
        for r in range(repeat):
            start = time.time()
            x_numba = parallel_sparse_numba_solver(
                A_data_arr, A_indices_arr, A_indptr_arr, b_arr, n
            )
            end = time.time()
            sparse_numba_times.append(end - start)

        sparse_numba_time = min(sparse_numba_times)  # Use the best time
        print(f"  sparse_numba time: {sparse_numba_time:.4f}s")

        # Calculate speedup
        speedup = scipy_time / sparse_numba_time
        print(f"  Speedup: {speedup:.2f}x")

        # Store results
        results['scipy_time'].append(scipy_time)
        results['sparse_numba_time'].append(sparse_numba_time)

    return results


def plot_benchmark_results(results):
    """Plot the results of the benchmark."""
    plt.figure(figsize=(10, 6))

    plt.plot(results['num_problems'], results['scipy_time'], 'o-', label='SciPy (Sequential)')
    plt.plot(results['num_problems'], results['sparse_numba_time'], 's-', label='sparse_numba (Parallel)')

    plt.xlabel('Number of Sparse Problems')
    plt.ylabel('Total Solution Time (s)')
    plt.title('Performance Comparison: SciPy vs sparse_numba')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_parallel_umfpack.png', dpi=300)
    plt.show()

    # Calculate speedup
    speedup = [scipy / numba for scipy, numba in zip(results['scipy_time'], results['sparse_numba_time'])]

    plt.figure(figsize=(10, 6))
    plt.plot(results['num_problems'], speedup, 'o-')
    plt.xlabel('Number of Sparse Problems')
    plt.ylabel('Speedup Factor (SciPy Time / sparse_numba Time)')
    plt.title('Speedup of Parallel sparse_numba over Sequential SciPy')
    plt.grid(True)

    # Add horizontal line at y=1 (no speedup)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('speedup_parallel_umfpack.png', dpi=300)
    plt.show()


def run_benchmark():
    print("Benchmarking sparse_numba vs SciPy for multiple problems...")

    # Parameters
    n = 1000  # Size of each problem
    num_problems_list = [1, 2, 4, 8, 16, 32]  # Different numbers of problems to solve

    # Run the benchmark
    results = benchmark_solvers(num_problems_list, n, repeat=3)

    # Plot the results
    plot_benchmark_results(results)

    # Print summary
    print("\nPerformance Summary:")
    for i, num_problems in enumerate(num_problems_list):
        speedup = results['scipy_time'][i] / results['sparse_numba_time'][i]
        print(f"{num_problems} problems: sparse_numba is {speedup:.2f}x faster than SciPy")