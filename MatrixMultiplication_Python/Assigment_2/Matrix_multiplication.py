import time

from Assigment_1.Matrix_multiplication import multiplication as multiply_baseline_ijk

# DENSE MATRIX MULTIPLICATION APPROACHES (BASELINE + 2 OPTIMISATIONS)

#se importa la matriz desde el assigment 1 para reutilizar la funcion :D

def multiply_loop_reordering_ikj(A, B, n):
    """
    [OPTIMISATION 1 - Loop Reordering (i-k-j)]
    """
    start = time.perf_counter()
    C = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_a = A[i]
        row_c = C[i]
        for k in range(n):
            a_ik = row_a[k]
            row_b = B[k]
            for j in range(n):
                row_c[j] += a_ik * row_b[j]

    end = time.perf_counter()
    return end - start, C


def multiply_transpose_b(A, B, n):
    """
    [OPTIMISATION 2 - Transpose Matrix B]
    """
    start = time.perf_counter()
    C = [[0.0 for _ in range(n)] for _ in range(n)]

    # Pre-transpose B: B_T[j][k] = B[k][j] (O(n^2) preprocessing step)
    B_T = [[B[k][j] for k in range(n)] for j in range(n)]

    for i in range(n):
        row_a = A[i]
        row_c = C[i]
        for j in range(n):
            row_bt = B_T[j]
            total = 0.0
            for k in range(n):
                total += row_a[k] * row_bt[k]
            row_c[j] = total

    end = time.perf_counter()
    return end - start, C
