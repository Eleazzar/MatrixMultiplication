import csv
import logging
import os
import random
import tracemalloc
from dask.distributed import Client, LocalCluster

from Assigment_4.Matrix_multiplication import (
    multiply_sequential,
    multiply_distributed,
)


def generate_matrix(n):

    matrix = []
    for _ in range(n):
        row = [random.uniform(0.0, 10.0) for _ in range(n)]
        matrix.append(row)
    return matrix


def compute_stats(values):

    mean = sum(values) / len(values)
    s = sorted(values)
    median = s[len(s) // 2]
    iqr = (s[3] - s[1]) if len(s) >= 5 else (s[-1] - s[0])
    return mean, median, iqr


def calculate_peak_memory_mb(traced_mb, n, num_workers):

    theoretical_mb = ((n * n * 8.0) + (num_workers * 16.0 * 1024.0)) / (1024.0 * 1024.0)
    return max(traced_mb, theoretical_mb)


def run_benchmark():
    seed = 42
    random.seed(seed)

    # Same matrix dimensions used in Assignments 1, 2, and 3
    sizes = [16, 32, 64, 128, 256, 384]
    worker_counts = [1, 2, 4, 8]
    repetitions = 5

    logical_cores = os.cpu_count() or 1
    csv_rows = []

    print("=" * 90)
    print("       PERFORMANCE BENCHMARK EXECUTION (PYTHON - SEQUENTIAL VS DASK DISTRIBUTED)")
    print("=" * 90)
    print("Detected logical processors: " + str(logical_cores))
    print("Matrix sizes: " + str(sizes))
    print("Worker counts (w): " + str(worker_counts))
    print("Repetitions per configuration: " + str(repetitions))
    print("=" * 90)

    for n in sizes:
        print("\n" + "=" * 90)
        print(f"                         TESTING MATRIX SIZE {n} x {n}")
        print("=" * 90)

        # Warm-up matrices (excluded from measured samples)
        A_warmup = generate_matrix(n)
        B_warmup = generate_matrix(n)

        # Pre-generate identical input matrices across all configurations for size n
        inputs_A = [generate_matrix(n) for _ in range(repetitions)]
        inputs_B = [generate_matrix(n) for _ in range(repetitions)]

        # 1. SEQUENTIAL BASELINE (Same as Assignment 1, with isolated tracemalloc as in Assignment 2)
        print(f"\n--- [Sequential Baseline] Matrix size {n} x {n} ---")
        multiply_sequential(A_warmup, B_warmup, n)

        # 1.1 Measure peak memory in an isolated run so tracemalloc hooks do not distort CPU timing
        tracemalloc.start()
        multiply_sequential(inputs_A[0], inputs_B[0], n)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        seq_peak_mb = calculate_peak_memory_mb(peak_bytes / (1024.0 * 1024.0), n, 1)

        # 1.2 Measure pure execution time across the 5 repetitions WITHOUT tracemalloc overhead
        seq_times = []
        seq_memories = []

        for rep in range(1, repetitions + 1):
            elapsed_time, _ = multiply_sequential(inputs_A[rep - 1], inputs_B[rep - 1], n)

            seq_times.append(elapsed_time)
            seq_memories.append(seq_peak_mb)

            print(f"  Repetition {rep}/{repetitions}: {elapsed_time:.5f} s | Peak memory: {seq_peak_mb:.3f} MB")

            csv_rows.append([
                n,
                "Sequential",
                "float64",
                1,
                rep,
                f"{elapsed_time:.6f}",
                f"{seq_peak_mb:.3f}",
            ])

        mean_seq_t, med_seq_t, iqr_seq_t = compute_stats(seq_times)
        mean_seq_m, med_seq_m, _ = compute_stats(seq_memories)

        print(
            f"  --> SUMMARY (Sequential): Mean = {mean_seq_t:.5f} s | Median = {med_seq_t:.5f} s "
            f"| IQR = {iqr_seq_t:.5f} s | Mean Mem = {mean_seq_m:.3f} MB | Peak Mem (Median) = {med_seq_m:.3f} MB"
        )

        # 2. DISTRIBUTED MULTI-WORKER CONFIGURATIONS (For the same size n)
        for w in worker_counts:
            print(f"\n--- [Distributed w = {w} workers] Matrix size {n} x {n} ---")

            with LocalCluster(
                n_workers=w,
                threads_per_worker=1,
                processes=True,
                silence_logs=logging.ERROR,
            ) as cluster, Client(cluster) as client:

                # Warm-up run on the active Dask cluster (excluded from measured samples)
                multiply_distributed(A_warmup, B_warmup, n, client, w)

                # 2.1 Measure peak memory in an isolated run so tracemalloc hooks do not distort timing
                tracemalloc.start()
                multiply_distributed(inputs_A[0], inputs_B[0], n, client, w)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                dist_peak_mb = calculate_peak_memory_mb(peak_bytes / (1024.0 * 1024.0), n, w)

                # 2.2 Measure pure execution time across the 5 repetitions WITHOUT tracemalloc overhead
                dist_times = []
                dist_memories = []

                for rep in range(1, repetitions + 1):
                    elapsed_time, _ = multiply_distributed(
                        inputs_A[rep - 1], inputs_B[rep - 1], n, client, w
                    )

                    dist_times.append(elapsed_time)
                    dist_memories.append(dist_peak_mb)

                    print(f"  Repetition {rep}/{repetitions}: {elapsed_time:.5f} s | Peak memory: {dist_peak_mb:.3f} MB")

                    csv_rows.append([
                        n,
                        "Distributed",
                        "float64",
                        w,
                        rep,
                        f"{elapsed_time:.6f}",
                        f"{dist_peak_mb:.3f}",
                    ])

                mean_d_t, med_d_t, iqr_d_t = compute_stats(dist_times)
                mean_d_m, med_d_m, _ = compute_stats(dist_memories)

                speedup_mean = mean_seq_t / mean_d_t if mean_d_t > 0 else 0.0
                eff_mean = speedup_mean / w
                speedup_med = med_seq_t / med_d_t if med_d_t > 0 else 0.0
                eff_med = speedup_med / w

                print(
                    f"  --> SUMMARY (w = {w:2d})    : Mean = {mean_d_t:.5f} s | Median = {med_d_t:.5f} s "
                    f"| IQR = {iqr_d_t:.5f} s | Mean Mem = {mean_d_m:.3f} MB | Peak Mem (Median) = {med_d_m:.3f} MB"
                )
                print(
                    f"      METRICS (w = {w:2d})    : Speedup (Mean) = {speedup_mean:.2f}x (Eff = {eff_mean * 100.0:.1f}%) "
                    f"| Speedup (Median) = {speedup_med:.2f}x (Eff = {eff_med * 100.0:.1f}%)"
                )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "results_assignment_4.csv")

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "method", "numeric_type", "workers", "repetition", "time_seconds", "peak_memory_mb"])
        writer.writerows(csv_rows)

    print("\n" + "=" * 90)
    print(f"Benchmark finished! Raw data saved to: {output_file}")
    print("==========================================================================================")

