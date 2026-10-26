import time

# PART B: SPARSE MATRIX REPRESENTATION (CSR) AND MULTIPLICATION


class CSRMatrix:
    """
    Custom Compressed Sparse Row (CSR) matrix representation for an n x n matrix.
    Stores only nonzero entries using three flat 1D arrays:
        - values:      List of nonzero floating-point values (length = nnz).
        - col_indices: Column index for each nonzero value (length = nnz).
        - row_ptr:     Start/end pointers in 'values' for each row (length = n + 1).
    """

    def __init__(self, values, col_indices, row_ptr, n):
        self.values = values
        self.col_indices = col_indices
        self.row_ptr = row_ptr
        self.n = n
        self.nnz = len(values)

    def to_dense(self):
        """Reconstructs the n x n dense 2D matrix from CSR format (for validation)."""
        dense = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            start_idx = self.row_ptr[i]
            end_idx = self.row_ptr[i + 1]
            for idx in range(start_idx, end_idx):
                j = self.col_indices[idx]
                dense[i][j] = self.values[idx]
        return dense


def dense_to_csr(M, n):
    """
    Converts an n x n dense 2D matrix M into Compressed Sparse Row (CSR) format.
    Measured separately so conversion cost can be analysed independently from
    the sparse multiplication kernel.

    Returns:
        conversion_time (float): Time spent converting dense -> CSR in seconds.
        csr_matrix (CSRMatrix): The resulting CSR representation.
    """
    start = time.perf_counter()

    values = []
    col_indices = []
    row_ptr = [0]

    for i in range(n):
        row = M[i]
        for j in range(n):
            val = row[j]
            if val != 0.0:
                values.append(val)
                col_indices.append(j)
        row_ptr.append(len(values))

    csr_matrix = CSRMatrix(values, col_indices, row_ptr, n)
    end = time.perf_counter()

    return end - start, csr_matrix


def multiply_sparse_csr(A_csr, B_csr, n):
    """
    [PART B - Sparse Matrix Multiplication using CSR Format]
    Multiplies two n x n sparse matrices A_csr and B_csr represented in CSR format,
    returning the product matrix C_csr also in CSR format.

    Algorithm (Gustavson's Row-wise Sparse Multiplication):
        For each row 'i' of A_csr, iterates ONLY over its nonzero entries A[i, k].
        For each nonzero A[i, k], iterates ONLY over the nonzero entries B[k, j]
        in row 'k' of B_csr, accumulating products into a temporary row buffer
        and tracking touched column indices to build row 'i' of C_csr.

    Returns:
        elapsed_time (float): Sparse multiplication kernel execution time in seconds.
        C_csr (CSRMatrix): The resulting product matrix in CSR format (with C_csr.nnz).
    """
    start = time.perf_counter()

    a_vals = A_csr.values
    a_cols = A_csr.col_indices
    a_row_ptr = A_csr.row_ptr

    b_vals = B_csr.values
    b_cols = B_csr.col_indices
    b_row_ptr = B_csr.row_ptr

    c_vals = []
    c_cols = []
    c_row_ptr = [0]

    # Reusable dense accumulator for a single row of C (O(n) auxiliary space)
    row_acc = [0.0] * n
    visited = [False] * n

    for i in range(n):
        touched_cols = []
        a_start = a_row_ptr[i]
        a_end = a_row_ptr[i + 1]

        for idx_a in range(a_start, a_end):
            k = a_cols[idx_a]
            a_val = a_vals[idx_a]

            b_start = b_row_ptr[k]
            b_end = b_row_ptr[k + 1]

            for idx_b in range(b_start, b_end):
                j = b_cols[idx_b]
                if not visited[j]:
                    visited[j] = True
                    touched_cols.append(j)
                row_acc[j] += a_val * b_vals[idx_b]

        # Sort column indices to maintain canonical CSR column ordering
        touched_cols.sort()
        for j in touched_cols:
            val = row_acc[j]
            if val != 0.0:
                c_vals.append(val)
                c_cols.append(j)
            # Reset accumulator for next row
            row_acc[j] = 0.0
            visited[j] = False

        c_row_ptr.append(len(c_vals))

    C_csr = CSRMatrix(c_vals, c_cols, c_row_ptr, n)
    end = time.perf_counter()

    return end - start, C_csr
