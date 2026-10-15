import csv
import os
import random
import tracemalloc

from Assigment_1.Matrix_multiplication import multiplication


def generate_matrix(n):
    """
    Generates an n x n square matrix filled with random floating-point numbers.
    Constructed row by row for maximum readability.
    """
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            number = random.uniform(0.0, 10.0)
            row.append(number)
        matrix.append(row)
    return matrix


def run_benchmark():
    random.seed(42)
    sizes = [16, 32, 64, 128, 256, 384]
    repetitions = 5

    csv_rows = []

    print("=========================================================")
    print("        PERFORMANCE BENCHMARK EXECUTION (PYTHON)")
    print("=========================================================")

    for n in sizes:
        print(f"\n--- Testing matrix size {n} x {n} ---")

        # -------------------------------------------------------------
        # Warm-up run (excluded from measured samples)
        # Preloads Python interpreter instructions and CPU hardware cache
        # -------------------------------------------------------------
        A_warmup = generate_matrix(n)
        B_warmup = generate_matrix(n)
        multiplication(A_warmup, B_warmup, n)

        times = []
        memories = []

        for rep in range(1, repetitions + 1):
            A = generate_matrix(n)
            B = generate_matrix(n)

            # Start memory tracking
            tracemalloc.start()

            # Execute multiplication kernel
            elapsed_time, C = multiplication(A, B, n)

            # Query peak allocated memory
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak_bytes / (1024 * 1024)

            times.append(elapsed_time)
            memories.append(peak_mb)

            print(f"  Repetition {rep}/{repetitions}: {elapsed_time:.5f} s | Peak memory: {peak_mb:.3f} MB")

            csv_rows.append([n, rep, elapsed_time, peak_mb])

        # Statistical summary: median and interquartile range (IQR)
        times_sorted = sorted(times)
        median_time = times_sorted[2]
        iqr_time = times_sorted[3] - times_sorted[1]

        memories_sorted = sorted(memories)
        median_memory = memories_sorted[2]

        print(f"  --> SUMMARY: Median = {median_time:.5f} s | IQR = {iqr_time:.5f} s | Peak Mem = {median_memory:.3f} MB")

    # Save all raw measurements to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "resultados_python.csv")
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "repetition", "time_seconds", "peak_memory_mb"])
        writer.writerows(csv_rows)

    print("\n=========================================================")
    print(f"Benchmark finished! Raw data saved to: {output_file}")
    print("=========================================================")


if __name__ == "__main__":
    run_benchmark()
