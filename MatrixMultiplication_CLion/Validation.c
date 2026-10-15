#include "Validation.h"
#include "MatrixMultiplication.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void print_matrix(double** matrix, int n, const char* label) {
    printf("\n%s:\n", label);
    for (int i = 0; i < n; i++) {
        printf("  [ ");
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", matrix[i][j]);
        }
        printf("]\n");
    }
}

void test_multiplication(int n) {
    srand(42);

    printf("=================================================================\n");
    printf("        VALIDATION SUITE: MATRIX MULTIPLICATION (C)\n");
    printf("=================================================================\n");

    // -------------------------------------------------------------
    // TEST 1: Identity Matrix (A * I = A)
    // -------------------------------------------------------------
    printf("\n[TEST 1] Identity Matrix Property: A * I = A\n");
    printf("Explanation: The identity matrix (I) is the neutral element in matrix\n");
    printf("multiplication. Multiplying any matrix by I must return the original matrix.\n\n");

    int size_demo = 3;
    double** A_demo = allocate_matrix(size_demo);
    double values[3][3] = {
        {3.0, 5.0, 2.0},
        {1.0, 7.0, 4.0},
        {6.0, 2.0, 9.0}
    };
    for (int i = 0; i < size_demo; i++) {
        for (int j = 0; j < size_demo; j++) {
            A_demo[i][j] = values[i][j];
        }
    }

    double** I_demo = allocate_matrix(size_demo);
    for (int i = 0; i < size_demo; i++) {
        I_demo[i][i] = 1.0;
    }

    Result res_ident = multiplication(A_demo, I_demo, size_demo);

    print_matrix(A_demo, size_demo, "Matrix A (original)");
    print_matrix(I_demo, size_demo, "Identity Matrix I");
    print_matrix(res_ident.C, size_demo, "Result A * I");

    int is_ident_ok = 1;
    for (int i = 0; i < size_demo; i++) {
        for (int j = 0; j < size_demo; j++) {
            if (fabs(res_ident.C[i][j] - A_demo[i][j]) > 1e-9) {
                is_ident_ok = 0;
            }
        }
    }

    if (is_ident_ok) {
        printf("\n-> EVALUATION: SUCCESS (A * I matches matrix A exactly).\n");
    } else {
        printf("\n-> EVALUATION: ERROR (Result differs from matrix A).\n");
    }

    // -------------------------------------------------------------
    // TEST 2: Zero Matrix (A * 0 = 0)
    // -------------------------------------------------------------
    printf("\n-----------------------------------------------------------------\n");
    printf("\n[TEST 2] Zero Matrix Property: A * 0 = 0\n");
    printf("Explanation: Multiplying any matrix by a zero matrix (all entries 0.0)\n");
    printf("must produce an identical zero matrix.\n\n");

    double** O_demo = allocate_matrix(size_demo);
    Result res_zero = multiplication(A_demo, O_demo, size_demo);

    print_matrix(O_demo, size_demo, "Zero Matrix O");
    print_matrix(res_zero.C, size_demo, "Result A * 0");

    int is_zero_ok = 1;
    for (int i = 0; i < size_demo; i++) {
        for (int j = 0; j < size_demo; j++) {
            if (fabs(res_zero.C[i][j]) > 1e-9) {
                is_zero_ok = 0;
            }
        }
    }

    if (is_zero_ok) {
        printf("\n-> EVALUATION: SUCCESS (All entries in A * O are 0.0).\n");
    } else {
        printf("\n-> EVALUATION: ERROR (Found non-zero entries in result).\n");
    }

    // -------------------------------------------------------------
    // TEST 3: Random Matrix Multiplication
    // -------------------------------------------------------------
    printf("\n-----------------------------------------------------------------\n");
    printf("\n[TEST 3] Random Matrix Multiplication\n");
    if (n <= 0) {
        n = (rand() % 256) + 1;
        printf("No valid 'n' provided. Randomly chosen: n = %d (range [1, 256])\n", n);
    } else {
        printf("Matrix dimension received as argument: n = %d\n", n);
    }

    double** A_rand = generate_matrix(n);
    double** B_rand = generate_matrix(n);

    printf("Executing multiplication kernel for square matrices %dx%d...\n", n, n);
    Result res_rand = multiplication(A_rand, B_rand, n);
    printf("-> Multiplication finished successfully.\n");
    printf("-> Kernel execution time: %.6f seconds.\n", res_rand.elapsed_time);
    printf("=================================================================\n\n");

    // Clean up allocated memory
    free_matrix(A_demo, size_demo);
    free_matrix(I_demo, size_demo);
    free_matrix(res_ident.C, size_demo);
    free_matrix(O_demo, size_demo);
    free_matrix(res_zero.C, size_demo);
    free_matrix(A_rand, n);
    free_matrix(B_rand, n);
    free_matrix(res_rand.C, n);
}
