#include "MatrixMultiplication.h"
#include <stdlib.h>
#include <time.h>

/**
 * Dynamically allocates an n x n square matrix initialized to 0.0.
 */
double** allocate_matrix(int n) {
    double** mat = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        mat[i] = (double*)calloc(n, sizeof(double));
    }
    return mat;
}

/**
 * Deallocates matrix memory.
 */
void free_matrix(double** mat, int n) {
    if (!mat) return;
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

/**
 * Generates an n x n square matrix with random floating-point values between 0.0 and 10.0.
 */
double** generate_matrix(int n) {
    double** mat = allocate_matrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = ((double)rand() / (double)RAND_MAX) * 10.0;
        }
    }
    return mat;
}

/**
 * Straightforward naive triple-loop matrix multiplication kernel (i-j-k loop order).
 * Computes C = A * B with O(n^3) arithmetic operations.
 */
Result multiplication(double** A, double** B, int n) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double** C = allocate_matrix(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    Result res;
    res.elapsed_time = elapsed;
    res.C = C;
    return res;
}
