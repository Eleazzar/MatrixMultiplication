import time


def multiplication(A, B, n):
    """
    Multiplies two n x n square matrices A and B using the standard
    triple-loop naive algorithm (i-j-k loop order) with O(n^3) arithmetic complexity.
    
    Returns:
        elapsed_time (float): Multiplication kernel execution time in seconds.
        C (list of lists): The resulting n x n product matrix.
    """
    start = time.perf_counter()
    C = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    end = time.perf_counter()

    return end - start, C
