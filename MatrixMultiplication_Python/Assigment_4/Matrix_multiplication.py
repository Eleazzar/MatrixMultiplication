import time
from Assigment_1.Matrix_multiplication import multiplication as _seq_multiplication


def multiply_sequential(A, B, n):

    return _seq_multiplication(A, B, n)


def partition_matrix_rows(A, n, num_workers):

    base_rows = n // num_workers
    remainder = n % num_workers

    blocks = []
    current_row = 0

    for w in range(num_workers):
        rows_for_worker = base_rows + (1 if w < remainder else 0)
        if rows_for_worker > 0:
            start_row = current_row
            end_row = current_row + rows_for_worker
            blocks.append(A[start_row:end_row])
            current_row = end_row

    return blocks


def _worker_multiply_block(A_block, B, n):

    num_rows = len(A_block)
    C_block = [[0.0 for _ in range(n)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(n):
            for k in range(n):
                C_block[i][j] += A_block[i][k] * B[k][j]

    return C_block


def multiply_distributed(A, B, n, client, num_workers):

    if num_workers <= 0:
        raise ValueError("Number of workers must be at least 1.")

    start = time.perf_counter()

    # 1. Partition matrix A into row blocks and distribute data to workers
    A_blocks = partition_matrix_rows(A, n, num_workers)
    [B_future] = client.scatter([B], broadcast=True, hash=False)
    A_futures = client.scatter(A_blocks, hash=False)

    # 2. Execute multiplication on workers
    result_futures = client.map(
        _worker_multiply_block,
        A_futures,
        B=B_future,
        n=n,
        pure=False,
    )

    # 3. Collect results and reconstruct matrix C
    C_blocks = client.gather(result_futures)
    C = [row for block in C_blocks for row in block]

    end = time.perf_counter()
    elapsed_time = end - start

    return elapsed_time, C
