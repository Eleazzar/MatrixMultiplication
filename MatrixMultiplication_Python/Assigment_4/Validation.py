import logging
import random
from dask.distributed import Client, LocalCluster

# Reused validation module from Assignment 1 (without modifying Assignment 1)
import Assigment_1.Validation as ValidationMatrix_1

from Assigment_4.Matrix_multiplication import (
    multiply_sequential,
    multiply_distributed,
)


def matrices_are_close(M1, M2, n, abs_tol=1e-9, rel_tol=1e-9):

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


def generate_test_matrix(n):

    return [[random.uniform(0.0, 10.0) for _ in range(n)] for _ in range(n)]


def test_multiplication(n=64):
    SEED = 42
    ABS_TOL = 1e-9
    REL_TOL = 1e-9
    default_workers = 2

    print("=" * 72)
    print("   STAGE 1: REUSING ASSIGNMENT 1 VALIDATION SUITE (DASK DISTRIBUTED)")
    print("=" * 72)

    original_a1_mult = ValidationMatrix_1.multiplication

    with LocalCluster(
        n_workers=default_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.ERROR,
    ) as cluster, Client(cluster) as client:

        def distributed_wrapper_for_a1(A, B, size):
            return multiply_distributed(A, B, size, client, default_workers)

        try:
            print(f"\n>>> Running Assignment 1 test_multiplication() on Dask ({default_workers} workers) <<<")
            ValidationMatrix_1.multiplication = distributed_wrapper_for_a1
            ValidationMatrix_1.test_multiplication(n)
        finally:
            ValidationMatrix_1.multiplication = original_a1_mult

    # STAGE 2: Multi-Worker Cross-Validation & Partition-Consistency Suite
    random.seed(SEED)
    if n is None or n <= 0:
        n = 64

    print("\n" + "=" * 72)
    print("   STAGE 2: MULTI-WORKER CROSS-VALIDATION & PARTITION CONSISTENCY SUITE")
    print("=" * 72)
    print(f"Numerical tolerance settings: abs_tol = {ABS_TOL:.1e}, rel_tol = {REL_TOL:.1e}")
    print(f"Fixed random seed: {SEED}")
    print(f"Matrix dimension : {n} x {n}")

    A = generate_test_matrix(n)
    B = generate_test_matrix(n)

    # Trusted reference from Sequential Baseline (Assignment 1)
    t_seq, C_ref = multiply_sequential(A, B, n)

    print("\n" + "-" * 72)
    print(f"  * Sequential Baseline (T_seq) : {t_seq:.6f} s [Trusted Reference]")
    print("-" * 72)

    worker_counts = [1, 2, 4, 8]
    all_passed = True

    for w in worker_counts:
        with LocalCluster(
            n_workers=w,
            threads_per_worker=1,
            processes=True,
            silence_logs=logging.ERROR,
        ) as cluster, Client(cluster) as client:

            t_dist, C_dist = multiply_distributed(A, B, n, client, w)
            is_close, max_diff = matrices_are_close(C_ref, C_dist, n, ABS_TOL, REL_TOL)

            if not is_close:
                all_passed = False

            status = "PASS" if is_close else "FAIL"
            print(
                f"  * Distributed (w = {w:2d} workers): {t_dist:.6f} s "
                f"| Max abs diff = {max_diff:.2e} | {status}"
            )

    print("-" * 72)
    if all_passed:
        print("-> OVERALL VALIDATION: ALL WORKER CONFIGURATIONS PASSED EXACTLY.")
    else:
        print("-> OVERALL VALIDATION: FAILED IN ONE OR MORE CONFIGURATIONS.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    test_multiplication()
