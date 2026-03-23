"""
Benchmark: Pre-Factorization API with Parallel Multi-Thread Solving
=====================================================================

Demonstrates the full advantage of pre-factorization combined with
Numba's nogil parallel execution:

1. Factorize ONCE, solve MANY times with different RHS vectors
   - Compares pre-factored vs combined factorize+solve
   - Shows the speedup from avoiding redundant factorization

2. Parallel multi-thread solving with prange
   - Multiple independent systems solved in parallel across all CPU cores
   - Compares sequential SciPy vs parallel sparse_numba (existing)
   - Compares sequential SciPy vs parallel sparse_numba pre-factored (new)

3. Constant-matrix repeated solve (the target use case)
   - Same matrix, many RHS vectors (e.g., linear ODE time-stepping)
   - Shows the combined speedup of pre-factorization + parallelism
"""

#  [sparse_numba] (C)2025 Tianqi Hong
#
#  BSD License

import numpy as np
import time
import platform
import multiprocessing
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve
from numba.typed import List
from numba import njit, prange, set_num_threads, get_num_threads
from numba.core import types

# Import sparse_numba solvers
from sparse_numba.sparse_superlu.superlu_numba_interface import (
    superlu_solve_csc,
    superlu_factorize_csc,
    superlu_solve_factored,
    superlu_free_factors,
)
from sparse_numba.conversion.matrix_conversion_numba import sparse_matvec_csr


# ================================================================
# Problem generation
# ================================================================

def generate_diag_dominant_csc(n, density=0.05, seed=42):
    """Generate a well-conditioned diagonally dominant sparse matrix in CSC."""
    np.random.seed(seed)
    A_dense = np.random.rand(n, n)
    mask = np.random.rand(n, n) < density
    A_dense = A_dense * mask
    for i in range(n):
        A_dense[i, i] = np.sum(np.abs(A_dense[i, :])) + 1.0
    return sparse.csc_matrix(A_dense)


def generate_multiple_problems(num_problems, n, density=0.05, seed=42):
    """Generate multiple independent sparse linear systems."""
    np.random.seed(seed)
    A_list = []
    b_list = []
    for i in range(num_problems):
        A = generate_diag_dominant_csc(n, density, seed=seed + i)
        x_true = np.random.rand(n)
        b = A @ x_true
        A_list.append(A)
        b_list.append(b)
    return A_list, b_list


# ================================================================
# Numba-compiled parallel solvers
# ================================================================

@njit(nogil=True, parallel=True)
def parallel_solve_combined(A_data_list, A_indices_list, A_indptr_list,
                            b_list, n):
    """Solve multiple systems in parallel using combined factorize+solve."""
    num = len(A_data_list)
    solutions = np.zeros((num, n), dtype=np.float64)
    for i in prange(num):
        x, info = superlu_solve_csc(
            A_data_list[i],
            A_indices_list[i].astype(np.int32),
            A_indptr_list[i].astype(np.int32),
            b_list[i],
        )
        solutions[i] = x
    return solutions


@njit(nogil=True, parallel=True)
def parallel_solve_prefactored(handles, b_list, n):
    """Solve multiple systems in parallel using pre-computed LU factors."""
    num = len(handles)
    solutions = np.zeros((num, n), dtype=np.float64)
    for i in prange(num):
        x, info = superlu_solve_factored(handles[i], b_list[i])
        solutions[i] = x
    return solutions


@njit(nogil=True)
def repeated_solve_combined(csc_data, csc_indices, csc_indptr, b_list, n):
    """Solve the SAME matrix with many RHS vectors (combined, sequential)."""
    num = len(b_list)
    solutions = np.zeros((num, n), dtype=np.float64)
    for i in range(num):
        x, info = superlu_solve_csc(csc_data, csc_indices, csc_indptr, b_list[i])
        solutions[i] = x
    return solutions


@njit(nogil=True)
def repeated_solve_prefactored(handle, b_list, n):
    """Solve the SAME matrix with many RHS vectors (pre-factored, sequential)."""
    num = len(b_list)
    solutions = np.zeros((num, n), dtype=np.float64)
    for i in range(num):
        x, info = superlu_solve_factored(handle, b_list[i])
        solutions[i] = x
    return solutions


@njit(nogil=True, parallel=True)
def repeated_solve_prefactored_parallel(handle, b_list, n):
    """Solve the SAME matrix with many RHS vectors (pre-factored, parallel)."""
    num = len(b_list)
    solutions = np.zeros((num, n), dtype=np.float64)
    for i in prange(num):
        x, info = superlu_solve_factored(handle, b_list[i])
        solutions[i] = x
    return solutions


# ================================================================
# Benchmark 1: Repeated solve on the SAME matrix
# ================================================================

def benchmark_repeated_solve(n, num_solves_list, repeat=3):
    """
    Benchmark: same matrix A, many different RHS vectors.
    This is the key use case for pre-factorization (e.g., linear ODE stepping).
    """
    print(f"\n{'='*70}")
    print(f"Benchmark 1: Repeated Solve (same matrix, varying RHS)")
    print(f"  Matrix size: {n}x{n}")
    print(f"{'='*70}")

    A_csc = generate_diag_dominant_csc(n)
    results = {
        'num_solves': num_solves_list,
        'scipy_time': [],
        'combined_time': [],
        'prefactored_time': [],
    }

    # Warm up JIT
    b_warm = List.empty_list(types.float64[:])
    b_warm.append(np.random.rand(n))
    _ = repeated_solve_combined(
        A_csc.data, A_csc.indices.astype(np.int32),
        A_csc.indptr.astype(np.int32), b_warm, n
    )
    handle_warm, _ = superlu_factorize_csc(
        A_csc.data, A_csc.indices.astype(np.int32),
        A_csc.indptr.astype(np.int32)
    )
    _ = repeated_solve_prefactored(handle_warm, b_warm, n)
    superlu_free_factors(handle_warm)
    print("  JIT warmup complete")

    for num_solves in num_solves_list:
        print(f"\n  {num_solves} solves:")

        # Generate RHS vectors
        np.random.seed(123)
        b_np_list = [np.random.rand(n) for _ in range(num_solves)]

        b_typed = List.empty_list(types.float64[:])
        for b in b_np_list:
            b_typed.append(np.asarray(b, dtype=np.float64))

        # --- SciPy sequential ---
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for b in b_np_list:
                _ = scipy_spsolve(A_csc, b)
            times.append(time.perf_counter() - t0)
        scipy_time = min(times)
        results['scipy_time'].append(scipy_time)
        print(f"    SciPy sequential:         {scipy_time:.4f}s")

        # --- sparse_numba combined (factorize+solve each time) ---
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = repeated_solve_combined(
                A_csc.data, A_csc.indices.astype(np.int32),
                A_csc.indptr.astype(np.int32), b_typed, n
            )
            times.append(time.perf_counter() - t0)
        combined_time = min(times)
        results['combined_time'].append(combined_time)
        print(f"    sparse_numba combined:    {combined_time:.4f}s")

        # --- sparse_numba pre-factored ---
        times = []
        for _ in range(repeat):
            handle, info = superlu_factorize_csc(
                A_csc.data, A_csc.indices.astype(np.int32),
                A_csc.indptr.astype(np.int32)
            )
            t0 = time.perf_counter()
            _ = repeated_solve_prefactored(handle, b_typed, n)
            prefactored_time = time.perf_counter() - t0
            superlu_free_factors(handle)
            times.append(prefactored_time)
        prefactored_time = min(times)
        results['prefactored_time'].append(prefactored_time)
        speedup = combined_time / prefactored_time if prefactored_time > 0 else float('inf')
        print(f"    sparse_numba pre-factored: {prefactored_time:.4f}s  ({speedup:.1f}x vs combined)")

    return results


# ================================================================
# Benchmark 2: Parallel multi-system solve
# ================================================================

def benchmark_parallel_solve(n, num_problems_list, repeat=3):
    """
    Benchmark: multiple independent systems solved in parallel.
    Compares SciPy sequential vs sparse_numba parallel (combined and pre-factored).
    """
    num_threads = get_num_threads()
    print(f"\n{'='*70}")
    print(f"Benchmark 2: Parallel Multi-System Solve")
    print(f"  Matrix size: {n}x{n}, Threads: {num_threads}")
    print(f"{'='*70}")

    results = {
        'num_problems': num_problems_list,
        'scipy_time': [],
        'parallel_combined_time': [],
        'parallel_prefactored_time': [],
    }

    # Warm up
    A_warm, b_warm = generate_multiple_problems(1, n)
    A_data_w = List.empty_list(types.float64[:])
    A_idx_w = List.empty_list(types.int64[:])
    A_ptr_w = List.empty_list(types.int64[:])
    b_w = List.empty_list(types.float64[:])
    A_data_w.append(A_warm[0].data)
    A_idx_w.append(np.asarray(A_warm[0].indices, dtype=np.int64))
    A_ptr_w.append(np.asarray(A_warm[0].indptr, dtype=np.int64))
    b_w.append(b_warm[0])
    _ = parallel_solve_combined(A_data_w, A_idx_w, A_ptr_w, b_w, n)

    handles_w = np.zeros(1, dtype=np.int64)
    h, _ = superlu_factorize_csc(
        A_warm[0].data, A_warm[0].indices.astype(np.int32),
        A_warm[0].indptr.astype(np.int32)
    )
    handles_w[0] = h
    _ = parallel_solve_prefactored(handles_w, b_w, n)
    superlu_free_factors(h)
    print("  JIT warmup complete")

    for num_problems in num_problems_list:
        print(f"\n  {num_problems} problems:")

        A_list, b_list = generate_multiple_problems(num_problems, n)

        # Prepare typed lists for Numba
        A_data_arr = List.empty_list(types.float64[:])
        A_indices_arr = List.empty_list(types.int64[:])
        A_indptr_arr = List.empty_list(types.int64[:])
        b_arr = List.empty_list(types.float64[:])

        for A in A_list:
            A_data_arr.append(np.asarray(A.data, dtype=np.float64))
            A_indices_arr.append(np.asarray(A.indices, dtype=np.int64))
            A_indptr_arr.append(np.asarray(A.indptr, dtype=np.int64))
        for b in b_list:
            b_arr.append(np.asarray(b, dtype=np.float64))

        # --- SciPy sequential ---
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for i in range(num_problems):
                _ = scipy_spsolve(A_list[i], b_list[i])
            times.append(time.perf_counter() - t0)
        scipy_time = min(times)
        results['scipy_time'].append(scipy_time)
        print(f"    SciPy sequential:              {scipy_time:.4f}s")

        # --- sparse_numba parallel combined ---
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = parallel_solve_combined(A_data_arr, A_indices_arr, A_indptr_arr, b_arr, n)
            times.append(time.perf_counter() - t0)
        par_combined = min(times)
        results['parallel_combined_time'].append(par_combined)
        speedup1 = scipy_time / par_combined if par_combined > 0 else float('inf')
        print(f"    sparse_numba parallel combined: {par_combined:.4f}s  ({speedup1:.1f}x vs SciPy)")

        # --- sparse_numba parallel pre-factored ---
        # Factorize all matrices first
        handles = np.zeros(num_problems, dtype=np.int64)
        for i in range(num_problems):
            h, info = superlu_factorize_csc(
                A_list[i].data,
                A_list[i].indices.astype(np.int32),
                A_list[i].indptr.astype(np.int32),
            )
            assert info == 0
            handles[i] = h

        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = parallel_solve_prefactored(handles, b_arr, n)
            times.append(time.perf_counter() - t0)
        par_prefactored = min(times)
        results['parallel_prefactored_time'].append(par_prefactored)
        speedup2 = scipy_time / par_prefactored if par_prefactored > 0 else float('inf')
        print(f"    sparse_numba parallel prefact:  {par_prefactored:.4f}s  ({speedup2:.1f}x vs SciPy)")

        # Free handles
        for i in range(num_problems):
            superlu_free_factors(handles[i])

    return results


# ================================================================
# Benchmark 3: Thread scaling
# ================================================================

def benchmark_thread_scaling(n, num_problems, max_threads=None, repeat=3):
    """
    Measure how performance scales with the number of threads.
    Uses pre-factored parallel solve to isolate the solve phase.
    """
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()

    print(f"\n{'='*70}")
    print(f"Benchmark 3: Thread Scaling (Pre-Factored Parallel Solve)")
    print(f"  Matrix size: {n}x{n}, Problems: {num_problems}, Max threads: {max_threads}")
    print(f"{'='*70}")

    A_list, b_list = generate_multiple_problems(num_problems, n)

    # Prepare typed lists
    b_arr = List.empty_list(types.float64[:])
    for b in b_list:
        b_arr.append(np.asarray(b, dtype=np.float64))

    # Pre-factorize all matrices
    handles = np.zeros(num_problems, dtype=np.int64)
    for i in range(num_problems):
        h, info = superlu_factorize_csc(
            A_list[i].data,
            A_list[i].indices.astype(np.int32),
            A_list[i].indptr.astype(np.int32),
        )
        assert info == 0
        handles[i] = h

    # Also prepare for combined solve
    A_data_arr = List.empty_list(types.float64[:])
    A_indices_arr = List.empty_list(types.int64[:])
    A_indptr_arr = List.empty_list(types.int64[:])
    for A in A_list:
        A_data_arr.append(np.asarray(A.data, dtype=np.float64))
        A_indices_arr.append(np.asarray(A.indices, dtype=np.int64))
        A_indptr_arr.append(np.asarray(A.indptr, dtype=np.int64))

    # Warmup with max threads
    set_num_threads(max_threads)
    _ = parallel_solve_prefactored(handles, b_arr, n)
    _ = parallel_solve_combined(A_data_arr, A_indices_arr, A_indptr_arr, b_arr, n)

    thread_counts = []
    t = 1
    while t <= max_threads:
        thread_counts.append(t)
        t *= 2
    if thread_counts[-1] != max_threads:
        thread_counts.append(max_threads)

    results = {
        'threads': thread_counts,
        'combined_time': [],
        'prefactored_time': [],
    }

    for num_t in thread_counts:
        set_num_threads(num_t)
        print(f"\n  Threads: {num_t}")

        # Combined
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = parallel_solve_combined(A_data_arr, A_indices_arr, A_indptr_arr, b_arr, n)
            times.append(time.perf_counter() - t0)
        ct = min(times)
        results['combined_time'].append(ct)
        print(f"    Combined:     {ct:.4f}s")

        # Pre-factored
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = parallel_solve_prefactored(handles, b_arr, n)
            times.append(time.perf_counter() - t0)
        pt = min(times)
        results['prefactored_time'].append(pt)
        print(f"    Pre-factored: {pt:.4f}s")

    # Free handles
    for i in range(num_problems):
        superlu_free_factors(handles[i])

    # Restore max threads
    set_num_threads(max_threads)

    return results


# ================================================================
# Plotting
# ================================================================

def plot_repeated_solve(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results['num_solves'], results['scipy_time'], 'o-', label='SciPy (sequential)')
    ax1.plot(results['num_solves'], results['combined_time'], 's-', label='sparse_numba combined')
    ax1.plot(results['num_solves'], results['prefactored_time'], '^-', label='sparse_numba pre-factored')
    ax1.set_xlabel('Number of Solves (same matrix)')
    ax1.set_ylabel('Total Time (s)')
    ax1.set_title('Repeated Solve: Same Matrix, Different RHS')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    speedup_combined = [s / c for s, c in zip(results['scipy_time'], results['combined_time'])]
    speedup_prefact = [s / p for s, p in zip(results['scipy_time'], results['prefactored_time'])]
    ax2.plot(results['num_solves'], speedup_combined, 's-', label='combined vs SciPy')
    ax2.plot(results['num_solves'], speedup_prefact, '^-', label='pre-factored vs SciPy')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Solves')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup over SciPy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('benchmark_repeated_solve_prefactored.png', dpi=300)
    plt.show()


def plot_parallel_solve(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results['num_problems'], results['scipy_time'], 'o-', label='SciPy (sequential)')
    ax1.plot(results['num_problems'], results['parallel_combined_time'], 's-', label='parallel combined')
    ax1.plot(results['num_problems'], results['parallel_prefactored_time'], '^-', label='parallel pre-factored')
    ax1.set_xlabel('Number of Independent Systems')
    ax1.set_ylabel('Total Time (s)')
    ax1.set_title('Parallel Multi-System Solve')
    ax1.legend()
    ax1.grid(True)

    speedup1 = [s / c for s, c in zip(results['scipy_time'], results['parallel_combined_time'])]
    speedup2 = [s / p for s, p in zip(results['scipy_time'], results['parallel_prefactored_time'])]
    ax2.plot(results['num_problems'], speedup1, 's-', label='parallel combined vs SciPy')
    ax2.plot(results['num_problems'], speedup2, '^-', label='parallel pre-factored vs SciPy')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Independent Systems')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup over Sequential SciPy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_parallel_prefactored.png', dpi=300)
    plt.show()


def plot_thread_scaling(results, num_problems):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results['threads'], results['combined_time'], 's-', label='Combined (factorize+solve)')
    ax1.plot(results['threads'], results['prefactored_time'], '^-', label='Pre-factored (solve only)')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Total Time (s)')
    ax1.set_title(f'Thread Scaling ({num_problems} systems)')
    ax1.legend()
    ax1.grid(True)

    # Speedup relative to single thread
    base_combined = results['combined_time'][0]
    base_prefact = results['prefactored_time'][0]
    speedup_c = [base_combined / t for t in results['combined_time']]
    speedup_p = [base_prefact / t for t in results['prefactored_time']]
    ideal = [t for t in results['threads']]

    ax2.plot(results['threads'], speedup_c, 's-', label='Combined')
    ax2.plot(results['threads'], speedup_p, '^-', label='Pre-factored')
    ax2.plot(results['threads'], ideal, 'k--', alpha=0.3, label='Ideal linear')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup (vs 1 thread)')
    ax2.set_title('Thread Scaling Efficiency')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_thread_scaling_prefactored.png', dpi=300)
    plt.show()


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"System: {platform.processor()}")
    print(f"CPU cores: {cpu_count}")
    print(f"Numba threads: {get_num_threads()}")

    # Parameters
    n = 500                                      # matrix size
    num_solves_list = [10, 50, 100, 500, 1000]   # for repeated solve benchmark
    num_problems_list = [1, 2, 4, 8, 16, 32]     # for parallel benchmark
    num_problems_scaling = 32                     # for thread scaling

    # Benchmark 1: Repeated solve (same matrix, many RHS)
    results1 = benchmark_repeated_solve(n, num_solves_list)
    plot_repeated_solve(results1)

    # Benchmark 2: Parallel multi-system solve
    results2 = benchmark_parallel_solve(n, num_problems_list)
    plot_parallel_solve(results2)

    # Benchmark 3: Thread scaling
    results3 = benchmark_thread_scaling(n, num_problems_scaling, max_threads=cpu_count)
    plot_thread_scaling(results3, num_problems_scaling)

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    print("\nRepeated Solve Speedup (pre-factored vs combined):")
    for i, ns in enumerate(num_solves_list):
        sp = results1['combined_time'][i] / results1['prefactored_time'][i]
        print(f"  {ns:4d} solves: {sp:.1f}x faster")

    print(f"\nParallel Solve Speedup (vs SciPy sequential):")
    for i, np_ in enumerate(num_problems_list):
        sp1 = results2['scipy_time'][i] / results2['parallel_combined_time'][i]
        sp2 = results2['scipy_time'][i] / results2['parallel_prefactored_time'][i]
        print(f"  {np_:2d} systems: combined={sp1:.1f}x, pre-factored={sp2:.1f}x")

    print(f"\nThread Scaling (pre-factored, {num_problems_scaling} systems):")
    base = results3['prefactored_time'][0]
    for i, t in enumerate(results3['threads']):
        sp = base / results3['prefactored_time'][i]
        eff = sp / t * 100
        print(f"  {t:2d} threads: {sp:.1f}x speedup ({eff:.0f}% efficiency)")
