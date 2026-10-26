import time

from Assigment_1.Matrix_multiplication import multiplication as multiply_baseline_ijk

# DENSE MATRIX MULTIPLICATION APPROACHES (BASELINE + 2 OPTIMISATIONS)

#se importa la matriz desde el assigment 1 para reutilizar la funcion :D

def multiply_loop_reordering_ikj(A, B, n):
    """
    [OPTIMISATION 1 - Loop Reordering (i-k-j)]
    Swaps the two inner loops from (i-j-k) to (i-k-j) and hoists invariant
    row references and scalar A[i][k] outside the innermost loop.

    Expected effect:
        1. Spatial locality (Row-Major access): Both C[i][j] and B[k][j] vary
           across index 'j' in the innermost loop, reading contiguous memory sequentially.
        2. Loop-invariant code motion: A[i][k] is fetched once per k-iteration
           (n^2 times total) instead of n^3 times inside the j-loop.

    Returns:
        elapsed_time (float): Multiplication kernel execution time in seconds.
        C (list of lists): The resulting n x n product matrix.
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
    Pre-transposes matrix B into B_T in O(n^2) time and space so that columns
    of B become contiguous rows in B_T (B_T[j][k] == B[k][j]). Then computes
    the dot product of row A[i] and row B_T[j] using a local accumulator.

    Expected effect:
        1. Converts column-wise traversal of B[k][j] into contiguous row-wise
           traversal of B_T[j][k], eliminating strided access.
        2. Accumulates the inner product in a fast local scalar variable 'total'
           and writes to C[i][j] only once per element (n^2 writes instead of n^3).
        3. The O(n^2) transposition cost is included inside the timed kernel boundary
           and is easily amortised by the O(n^3) multiplication savings.

    Returns:
        elapsed_time (float): Total execution time (transposition + kernel) in seconds.
        C (list of lists): The resulting n x n product matrix.
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
