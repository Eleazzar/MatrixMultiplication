import random
import time

from Assigment_1.Matrix_multiplication import multiplication


def format_matrix(matrix):
    """Formats a matrix into a clean, human-readable string grid."""
    return '\n'.join('  [' + ' '.join(f'{val:6.2f}' for val in row) + ' ]' for row in matrix)


def test_multiplication(n=None):
    """
    Validation test suite for matrix multiplication:
    - Fixed seed at 42 for reproducibility.
    - TEST 1: Identity Matrix (A * I = A).
    - TEST 2: Zero Matrix (A * 0 = 0).
    - TEST 3: Random Matrix Multiplication (size n x n).
    """
    SEED = 42
    random.seed(SEED)

    print("=" * 65)
    print("        VALIDATION SUITE: MATRIX MULTIPLICATION")
    print("=" * 65)

    # -------------------------------------------------------------
    # TEST 1: Identity Matrix (A * I = A)
    # -------------------------------------------------------------
    print("\n[TEST 1] Identity Matrix Property: A * I = A")
    print("Explanation: The identity matrix (I) is the neutral element in matrix")
    print("multiplication. Multiplying any matrix by I must return the original matrix.\n")

    size_demo = 3
    A_demo = [
        [3.0, 5.0, 2.0],
        [1.0, 7.0, 4.0],
        [6.0, 2.0, 9.0]
    ]
    # Generate identity matrix (1 on main diagonal, 0 elsewhere)
    I_demo = [
        [1.0 if i == j else 0.0 for j in range(size_demo)]
        for i in range(size_demo)
    ]

    _, C_identity = multiplication(A_demo, I_demo, size_demo)

    print("Matrix A (original):")
    print(format_matrix(A_demo))
    print("\nIdentity Matrix I:")
    print(format_matrix(I_demo))
    print("\nResult A * I:")
    print(format_matrix(C_identity))

    if C_identity == A_demo:
        print("\n-> EVALUATION: SUCCESS (A * I matches matrix A exactly).")
    else:
        print("\n-> EVALUATION: ERROR (Result differs from matrix A).")

    # -------------------------------------------------------------
    # TEST 2: Zero Matrix (A * 0 = 0)
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print("\n[TEST 2] Zero Matrix Property: A * 0 = 0")
    print("Explanation: Multiplying any matrix by a zero matrix (all entries 0.0)")
    print("must produce an identical zero matrix.\n")

    O_demo = [[0.0 for _ in range(size_demo)] for _ in range(size_demo)]

    _, C_zero = multiplication(A_demo, O_demo, size_demo)

    print("Zero Matrix O:")
    print(format_matrix(O_demo))
    print("\nResult A * 0:")
    print(format_matrix(C_zero))

    if C_zero == O_demo:
        print("\n-> EVALUATION: SUCCESS (All entries in A * O are 0.0).")
    else:
        print("\n-> EVALUATION: ERROR (Found non-zero entries in result).")

    # -------------------------------------------------------------
    # TEST 3: Random Matrix Multiplication
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print("\n[TEST 3] Random Matrix Multiplication")

    if n is None or n <= 0:
        n = random.randint(1, 256)
        print(f"No valid 'n' provided. Randomly chosen: n = {n} (range [1, 256])")
    else:
        print(f"Matrix dimension received as argument: n = {n}")

    print(f"Fixed random seed: {SEED}")

    A_rand = [[random.uniform(0.0, 10.0) for _ in range(n)] for _ in range(n)]
    B_rand = [[random.uniform(0.0, 10.0) for _ in range(n)] for _ in range(n)]

    print(f"Executing multiplication kernel for square matrices {n}x{n}...")
    elapsed_time, C_rand = multiplication(A_rand, B_rand, n)

    print("-> Multiplication finished successfully.")
    print(f"-> Result matrix dimensions: {len(C_rand)} rows x {len(C_rand[0])} columns.")
    print(f"-> Kernel execution time: {elapsed_time:.6f} seconds.")
    print("=" * 65)
