#include "Benchmark.h"
#include "MatrixMultiplication.h"
#include <stdio.h>
#include <stdlib.h>

static int compare_doubles(const void* a, const void* b) {
    double diff = (*(double*)a - *(double*)b);
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

void run_benchmark(void) {
    srand(42);
    int sizes[] = {16, 32, 64, 128, 256, 384};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int repetitions = 5;

    const char* filename = "resultados_c.csv";
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file %s for writing.\n", filename);
        return;
    }

    fprintf(f, "n,repetition,time_seconds,peak_memory_mb\n");

    printf("=========================================================\n");
    printf("        PERFORMANCE BENCHMARK EXECUTION (C)\n");
    printf("=========================================================\n");

    for (int t = 0; t < num_sizes; t++) {
        int n = sizes[t];
        printf("\n--- Testing matrix size %d x %d ---\n", n, n);

        // -------------------------------------------------------------
        // Warm-up run (excluded from measured samples)
        // Stabilizes CPU hardware caches and OS page table allocations
        // -------------------------------------------------------------
        double** A_warm = generate_matrix(n);
        double** B_warm = generate_matrix(n);
        Result res_warm = multiplication(A_warm, B_warm, n);
        free_matrix(A_warm, n);
        free_matrix(B_warm, n);
        free_matrix(res_warm.C, n);

        double times[5];
        double memories[5];

        // -------------------------------------------------------------
        // Official measured repetitions
        // -------------------------------------------------------------
        for (int rep = 1; rep <= repetitions; rep++) {
            double** A = generate_matrix(n);
            double** B = generate_matrix(n);

            Result res = multiplication(A, B, n);

            // Theoretical physical memory of double matrix: n * n * 8 bytes
            double memory_mb = (double)(n * n * sizeof(double)) / (1024.0 * 1024.0);

            times[rep - 1] = res.elapsed_time;
            memories[rep - 1] = memory_mb;

            printf("  Repetition %d/%d: %.6f s | Peak memory: %.3f MB\n",
                   rep, repetitions, res.elapsed_time, memory_mb);

            fprintf(f, "%d,%d,%.6f,%.3f\n", n, rep, res.elapsed_time, memory_mb);

            free_matrix(A, n);
            free_matrix(B, n);
            free_matrix(res.C, n);
        }

        // Statistical summary: median and interquartile range (IQR)
        qsort(times, repetitions, sizeof(double), compare_doubles);
        double median_time = times[2];
        double iqr_time = times[3] - times[1];
        double median_memory = memories[2];

        printf("  --> SUMMARY: Median = %.6f s | IQR = %.6f s | Peak Mem = %.3f MB\n",
               median_time, iqr_time, median_memory);
    }

    fclose(f);
    printf("\n=========================================================\n");
    printf("Benchmark finished! Raw data saved to: %s\n", filename);
    printf("=========================================================\n");
}
