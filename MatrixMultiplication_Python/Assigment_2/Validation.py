import random

# Reused validation module and helpers directly from Assignment 1 (without modifying Assignment 1)
import Assigment_1.Validation as ValidationMatrix_1

from Assigment_2.Matrix_multiplication import (
    multiply_baseline_ijk,
    multiply_loop_reordering_ikj,
    multiply_transpose_b,
)
from Assigment_2.CSR import (
    CSRMatrix,
    dense_to_csr,
    multiply_sparse_csr,
)


def matrices_are_close(M1, M2, n, abs_tol=1e-9, rel_tol=1e-9):
    """
    Compares two n x n floating-point matrices using both absolute and relative
    tolerances: |a - b| <= max(abs_tol, rel_tol * max(|a|, |b|)).
    Returns (is_equal, max_abs_diff).
    """
    max_diff = 0.0
    for i in range(n):
        for j in range(n):
            a = M1[i][j]
            b = M2[i][j]
            diff = abs(a - b)
            if diff > max_diff:
                max_diff = diff
            threshold = max(abs_tol, rel_tol * max(abs(a), abs(b)))
            if diff > threshold:
                return False, max_diff
    return True, max_diff


def generate_test_matrix(n, sparsity=0.0):
    """Generates an n x n matrix with a given sparsity (fraction of zeros in [0.0, 1.0])."""
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            if random.random() < sparsity:
                row.append(0.0)
            else:
                row.append(random.uniform(0.1, 10.0))
        matrix.append(row)
    return matrix


def multiply_csr_wrapper_for_a1(A, B, n):
    """
    Adapter so that Sparse CSR multiplication can also be tested directly
    by Assignment 1's test_multiplication(n) suite (which expects dense 2D inputs/outputs).
    """
    _, A_csr = dense_to_csr(A, n)
    _, B_csr = dense_to_csr(B, n)
    elapsed, C_csr = multiply_sparse_csr(A_csr, B_csr, n)
    return elapsed, C_csr.to_dense()


def test_multiplication(n=64):
    """
    Validation test suite for Assignment 2:
    1. Reuses Assignment 1's test_multiplication() suite (TEST 1: Identity A*I=A,
       TEST 2: Zero A*0=0, TEST 3: Random matrix) by injecting each new Assignment 2
       method into Assigment_1.Validation without modifying Assignment 1's files.
    2. Extends validation with floating-point tolerance checks (abs_tol=1e-9, rel_tol=1e-9)
       and sparse fill-in verification against the Assignment 1 Baseline.
    """
    SEED = 42
    ABS_TOL = 1e-9
    REL_TOL = 1e-9

    print("=" * 72)
    print("   STAGE 1: REUSING ASSIGNMENT 1 VALIDATION SUITE FOR NEW METHODS")
    print("=" * 72)

    original_a1_mult = ValidationMatrix_1.multiplication
    try:
        for method_label, func in [
            ("Optimisation 1: Loop Reordering (i-k-j)", multiply_loop_reordering_ikj),
            ("Optimisation 2: Transpose B (B^T)", multiply_transpose_b),
            ("Part B: Sparse CSR Multiplication", multiply_csr_wrapper_for_a1),
        ]:
            print(f"\n>>> Running Assignment 1 test_multiplication() for: [{method_label}] <<<")
            ValidationMatrix_1.multiplication = func
            ValidationMatrix_1.test_multiplication(n)
    finally:
        # Always restore Assignment 1's original function reference
        ValidationMatrix_1.multiplication = original_a1_mult

    # STAGE 2: Assignment 2 Specific Cross-Validation (Tolerance & Sparsity)

    random.seed(SEED)
    print("\n" + "=" * 72)
    print("   STAGE 2: CROSS-METHOD TOLERANCE & SPARSE CSR VALIDATION")
    print("=" * 72)
    print(f"Numerical tolerance settings: abs_tol = {ABS_TOL:.1e}, rel_tol = {REL_TOL:.1e}")
    print(f"Fixed random seed: {SEED}")

    print("\n" + "-" * 72)
    print(f"[CROSS-TEST A] Dense Matrix Reference Comparison ({n}x{n}, 0% sparsity)")
    A_dense = generate_test_matrix(n, sparsity=0.0)
    B_dense = generate_test_matrix(n, sparsity=0.0)

    t_base, C_ref = multiply_baseline_ijk(A_dense, B_dense, n)
    t_ikj, C_opt1 = multiply_loop_reordering_ikj(A_dense, B_dense, n)
    t_tr, C_opt2 = multiply_transpose_b(A_dense, B_dense, n)

    _, A_d_csr = dense_to_csr(A_dense, n)
    _, B_d_csr = dense_to_csr(B_dense, n)
    t_sp, C_sp_obj = multiply_sparse_csr(A_d_csr, B_d_csr, n)
    C_sp_dense = C_sp_obj.to_dense()

    ok_ikj, diff_ikj = matrices_are_close(C_ref, C_opt1, n, ABS_TOL, REL_TOL)
    ok_tr, diff_tr = matrices_are_close(C_ref, C_opt2, n, ABS_TOL, REL_TOL)
    ok_sp, diff_sp = matrices_are_close(C_ref, C_sp_dense, n, ABS_TOL, REL_TOL)

    print(f"  * Baseline (A1 i-j-k)    : {t_base:.6f} s [Trusted Reference]")
    print(f"  * Loop Reordering (i-k-j): {t_ikj:.6f} s | Max abs diff = {diff_ikj:.2e} | {'PASS' if ok_ikj else 'FAIL'}")
    print(f"  * Transpose B (B^T)      : {t_tr:.6f} s | Max abs diff = {diff_tr:.2e} | {'PASS' if ok_tr else 'FAIL'}")
    print(f"  * Sparse CSR             : {t_sp:.6f} s | Max abs diff = {diff_sp:.2e} | {'PASS' if ok_sp else 'FAIL'}")

    print("\n" + "-" * 72)
    print(f"[CROSS-TEST B] Sparse Matrix Reference Comparison ({n}x{n}, 80% sparsity / 20% density)")
    A_sparse = generate_test_matrix(n, sparsity=0.80)
    B_sparse = generate_test_matrix(n, sparsity=0.80)

    _, C_sp_ref = multiply_baseline_ijk(A_sparse, B_sparse, n)
    _, A_s_csr = dense_to_csr(A_sparse, n)
    _, B_s_csr = dense_to_csr(B_sparse, n)
    _, C_s_csr_obj = multiply_sparse_csr(A_s_csr, B_s_csr, n)
    C_s_reconstructed = C_s_csr_obj.to_dense()

    ok_sp80, diff_sp80 = matrices_are_close(C_sp_ref, C_s_reconstructed, n, ABS_TOL, REL_TOL)
    total_elements = n * n
    print(f"  * Input A nonzeros (nnz_A) : {A_s_csr.nnz} / {total_elements} ({100.0 * A_s_csr.nnz / total_elements:.1f}% density)")
    print(f"  * Input B nonzeros (nnz_B) : {B_s_csr.nnz} / {total_elements} ({100.0 * B_s_csr.nnz / total_elements:.1f}% density)")
    print(f"  * Output C nonzeros (nnz_C): {C_s_csr_obj.nnz} / {total_elements} ({100.0 * C_s_csr_obj.nnz / total_elements:.1f}% density -> fill-in effect)")
    print(f"  * Max abs diff vs Baseline : {diff_sp80:.2e} | {'PASS' if ok_sp80 else 'FAIL'}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    test_multiplication()
