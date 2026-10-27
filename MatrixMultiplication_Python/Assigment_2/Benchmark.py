import csv
import os
import random
import tracemalloc

from Assigment_2.Matrix_multiplication import (
    multiply_baseline_ijk,
    multiply_loop_reordering_ikj,
    multiply_transpose_b,
)
from Assigment_2.CSR import (
    dense_to_csr,
    multiply_sparse_csr,
)


def generate_matrix(n, sparsity=0.0):
    """
    Generates an n x n square matrix with controlled sparsity.
    """
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            if sparsity > 0.0 and random.random() < sparsity:
                row.append(0.0)
            else:
                row.append(random.uniform(0.1, 10.0))
        matrix.append(row)
    return matrix


def compute_median_and_iqr(values):
    s = sorted(values)
    median = s[len(s) // 2]
    iqr = s[3] - s[1] if len(s) >= 5 else (s[-1] - s[0])
    return median, iqr


def run_benchmark_part_a(sizes=None, repetitions=5):
    """
    PART A: Dense Matrix Multiplication Optimisation Benchmark.
    """
    if sizes is None:
        sizes = [16, 32, 64, 128, 256, 384]

    random.seed(42)
    csv_rows = []

    methods = [
        ("Baseline_ijk", multiply_baseline_ijk),
        ("Loop_Reordering_ikj", multiply_loop_reordering_ikj),
        ("Transpose_B", multiply_transpose_b),
    ]

    print("=" * 78)
    print("   PART A BENCHMARK: DENSE OPTIMISATION APPROACHES (Baseline vs ikj vs B^T)")
    print("=" * 78)

    for n in sizes:
        print(f"\n--- Testing dense matrix size n = {n} x {n} (0% sparsity / 100% density) ---")

        # Warm-up run for each method (excluded from reported samples)
        A_warm = generate_matrix(n, sparsity=0.0)
        B_warm = generate_matrix(n, sparsity=0.0)
        for _, func in methods:
            func(A_warm, B_warm, n)

        # Pre-generate 5 matched input pairs (A, B) so all 3 methods see identical matrices
        input_pairs = [
            (generate_matrix(n, sparsity=0.0), generate_matrix(n, sparsity=0.0))
            for _ in range(repetitions)
        ]

        for method_name, func in methods:
            times = []
            memories = []

            # 1. Measure peak memory in an isolated run so tracemalloc hooks do not distort CPU timing
            A_mem, B_mem = input_pairs[0]
            tracemalloc.start()
            func(A_mem, B_mem, n)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak_bytes / (1024 * 1024)

            # 2. Measure pure execution time across the 5 repetitions WITHOUT tracemalloc overhead
            for rep in range(1, repetitions + 1):
                A, B = input_pairs[rep - 1]
                elapsed_time, C = func(A, B, n)

                times.append(elapsed_time)
                memories.append(peak_mb)

                csv_rows.append([
                    n,
                    method_name,
                    "float64",
                    rep,
                    elapsed_time,
                    peak_mb,
                ])

            med_t, iqr_t = compute_median_and_iqr(times)
            med_m, _ = compute_median_and_iqr(memories)
            print(
                f"  [{method_name:<20}] Median Time = {med_t:.5f} s (IQR = {iqr_t:.5f} s) | "
                f"Peak Mem = {med_m:.3f} MB"
            )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "results_part_a_dense.csv")
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "method",
            "numeric_type",
            "repetition",
            "time_seconds",
            "peak_memory_mb",
        ])
        writer.writerows(csv_rows)

    print(f"\n[OK] Part A raw measurements saved to: {output_file}")
    return output_file


def run_benchmark_part_b(n_fixed=128, sparsity_levels=None, repetitions=5):
    """
    PART B: Sparse Matrix Multiplication (CSR) vs Dense Methods across Sparsity Levels.
    """
    if sparsity_levels is None:
        # Declared sparsity levels (percentage of zero entries): 0%, 50%, 80%, 90%, 95%, 99%
        sparsity_levels = [0.0, 0.25, 0.50, 0.75, 0.95]

    random.seed(42)
    csv_rows = []
    total_entries = n_fixed * n_fixed

    print("\n" + "=" * 78)
    print(f"   PART B BENCHMARK: SPARSE (CSR) vs DENSE ACROSS SPARSITY LEVELS (n = {n_fixed})")
    print("=" * 78)

    # Warm-up run
    A_w = generate_matrix(64, sparsity=0.80)
    B_w = generate_matrix(64, sparsity=0.80)
    multiply_baseline_ijk(A_w, B_w, 64)
    multiply_loop_reordering_ikj(A_w, B_w, 64)
    _, A_w_csr = dense_to_csr(A_w, 64)
    _, B_w_csr = dense_to_csr(B_w, 64)
    multiply_sparse_csr(A_w_csr, B_w_csr, 64)

    for sp in sparsity_levels:
        sparsity_pct = round(sp * 100.0, 2)
        density_pct = round((1.0 - sp) * 100.0, 2)
        print(f"\n--- Sparsity = {sparsity_pct:5.1f}% zeros (Density = {density_pct:5.1f}% nonzeros) | n = {n_fixed} ---")

        # Pre-generate 5 matched logical input pairs for this sparsity level
        input_pairs = [
            (generate_matrix(n_fixed, sparsity=sp), generate_matrix(n_fixed, sparsity=sp))
            for _ in range(repetitions)
        ]

        # 1. Evaluate Dense Baseline (i-j-k) and Dense Optimised (i-k-j)
        for dense_name, dense_func in [
            ("Dense_Baseline_ijk", multiply_baseline_ijk),
            ("Dense_Optimised_ikj", multiply_loop_reordering_ikj),
        ]:
            d_times = []
            d_mems = []

            # Isolated memory measurement (avoids tracemalloc instrumentation overhead during timing)
            tracemalloc.start()
            dense_func(input_pairs[0][0], input_pairs[0][1], n_fixed)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak_bytes / (1024 * 1024)

            # Pure timing repetitions without tracemalloc overhead
            for rep in range(1, repetitions + 1):
                A, B = input_pairs[rep - 1]
                # Count logical nonzeros for reporting consistency
                nnz_a = sum(1 for row in A for val in row if val != 0.0)
                nnz_b = sum(1 for row in B for val in row if val != 0.0)

                k_time, C_dense = dense_func(A, B, n_fixed)
                nnz_c = sum(1 for row in C_dense for val in row if val != 0.0)

                d_times.append(k_time)
                d_mems.append(peak_mb)

                csv_rows.append([
                    n_fixed,
                    dense_name,
                    "Dense_2D",
                    sparsity_pct,
                    density_pct,
                    nnz_a,
                    nnz_b,
                    nnz_c,
                    rep,
                    0.0,       # conversion_time_seconds (not applicable to dense)
                    k_time,    # kernel_time_seconds
                    k_time,    # total_time_seconds
                    peak_mb,
                ])

            med_t, iqr_t = compute_median_and_iqr(d_times)
            med_m, _ = compute_median_and_iqr(d_mems)
            print(
                f"  [{dense_name:<20}] Kernel = {med_t:.5f} s (IQR = {iqr_t:.5f} s) | "
                f"Peak Mem = {med_m:.3f} MB"
            )

        # 2. Evaluate Sparse CSR Multiplication (measuring conversion and kernel separately)
        csr_conv_times = []
        csr_kern_times = []
        csr_tot_times = []
        csr_mems = []
        last_nnz_a = last_nnz_b = last_nnz_c = 0

        for rep in range(1, repetitions + 1):
            A, B = input_pairs[rep - 1]

            # Step 1: Convert logical dense matrices A and B to CSR (timed separately, no tracemalloc)
            conv_t_a, A_csr = dense_to_csr(A, n_fixed)
            conv_t_b, B_csr = dense_to_csr(B, n_fixed)
            conv_time = conv_t_a + conv_t_b

            # Step 2a: Measure pure CSR multiplication kernel time WITHOUT tracemalloc overhead
            kern_time, C_csr = multiply_sparse_csr(A_csr, B_csr, n_fixed)

            # Step 2b: Measure peak memory in an isolated call
            tracemalloc.start()
            multiply_sparse_csr(A_csr, B_csr, n_fixed)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak_bytes / (1024 * 1024)

            tot_time = conv_time + kern_time

            csr_conv_times.append(conv_time)
            csr_kern_times.append(kern_time)
            csr_tot_times.append(tot_time)
            csr_mems.append(peak_mb)

            last_nnz_a, last_nnz_b, last_nnz_c = A_csr.nnz, B_csr.nnz, C_csr.nnz

            csv_rows.append([
                n_fixed,
                "Sparse_CSR",
                "CSR",
                sparsity_pct,
                density_pct,
                A_csr.nnz,
                B_csr.nnz,
                C_csr.nnz,
                rep,
                conv_time,
                kern_time,
                tot_time,
                peak_mb,
            ])

        med_k, iqr_k = compute_median_and_iqr(csr_kern_times)
        med_c, _ = compute_median_and_iqr(csr_conv_times)
        med_tot, _ = compute_median_and_iqr(csr_tot_times)
        med_m, _ = compute_median_and_iqr(csr_mems)
        out_density = 100.0 * last_nnz_c / total_entries

        print(
            f"  [{'Sparse_CSR':<20}] Kernel = {med_k:.5f} s (IQR = {iqr_k:.5f} s) | "
            f"Conv = {med_c:.5f} s | Total = {med_tot:.5f} s | Peak Mem = {med_m:.3f} MB"
        )
        print(
            f"                         Nonzeros: nnz(A)={last_nnz_a}, nnz(B)={last_nnz_b} "
            f"-> nnz(C)={last_nnz_c}/{total_entries} ({out_density:.1f}% output density)"
        )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "results_part_b_sparse.csv")
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "method",
            "representation",
            "sparsity_percent",
            "density_percent",
            "nnz_A",
            "nnz_B",
            "nnz_C",
            "repetition",
            "conversion_time_seconds",
            "kernel_time_seconds",
            "total_time_seconds",
            "peak_memory_mb",
        ])
        writer.writerows(csv_rows)

    print(f"\n[OK] Part B raw measurements saved to: {output_file}")
    return output_file


def run_benchmark():
    """Runs both Part A (Dense Optimisations) and Part B (Sparse CSR) benchmarks."""
    run_benchmark_part_a()
    run_benchmark_part_b()

