package Assigment_3;

import java.util.Arrays;
import java.util.Random;

public class Validation {

    public static Object[] matricesAreClose(double[][] M1, double[][] M2, int n, double absTol, double relTol) {
        double maxDiff = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double a = M1[i][j];
                double b = M2[i][j];
                double diff = Math.abs(a - b);
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
                double threshold = Math.max(absTol, relTol * Math.max(Math.abs(a), Math.abs(b)));
                if (diff > threshold) {
                    return new Object[]{false, maxDiff};
                }
            }
        }
        return new Object[]{true, maxDiff};
    }

    public static void testMultiplication() {
        testMultiplication(64);
    }

    public static void testMultiplication(int n) {
        long SEED = 42L;
        double ABS_TOL = 1e-9;
        double REL_TOL = 1e-9;
        int defaultThreads = Runtime.getRuntime().availableProcessors();

        System.out.println("========================================================================");
        System.out.println("   STAGE 1: REUSING ASSIGNMENT 1 VALIDATION & ALGEBRAIC CHECKS (JAVA)");
        System.out.println("========================================================================");

        // 1.1 Reuse Assignment 1 validation suite for the Sequential Baseline
        System.out.println("\n>>> Running Assignment 1 testMultiplication() for Sequential Baseline <<<");
        Assigment_1.Validation.testMultiplication(n);

        // 1.2 Algebraic checks (Identity and Zero matrices) for Parallel Implementation
        System.out.println("\n------------------------------------------------------------------------");
        System.out.println("[PARALLEL ALGEBRAIC CHECK] Testing multiplyParallel with p = " + defaultThreads + " threads");

        int sizeDemo = 3;
        double[][] ADemo = {
                {3.0, 5.0, 2.0},
                {1.0, 7.0, 4.0},
                {6.0, 2.0, 9.0}
        };
        double[][] IDemo = new double[sizeDemo][sizeDemo];
        for (int i = 0; i < sizeDemo; i++) {
            IDemo[i][i] = 1.0;
        }
        double[][] ODemo = new double[sizeDemo][sizeDemo];

        Object[] resIdentPar = MatrixMultiplication_comparation.multiplyParallel(ADemo, IDemo, sizeDemo, defaultThreads);
        double[][] CIdentityPar = (double[][]) resIdentPar[1];

        Object[] resZeroPar = MatrixMultiplication_comparation.multiplyParallel(ADemo, ODemo, sizeDemo, defaultThreads);
        double[][] CZeroPar = (double[][]) resZeroPar[1];

        Assigment_1.Validation.printMatrix(CIdentityPar, "Parallel Result A * I (p = " + defaultThreads + ")");
        if (Arrays.deepEquals(CIdentityPar, ADemo)) {
            System.out.println("-> IDENTITY TEST (A * I = A): SUCCESS");
        } else {
            System.out.println("-> IDENTITY TEST (A * I = A): ERROR");
        }

        Assigment_1.Validation.printMatrix(CZeroPar, "Parallel Result A * 0 (p = " + defaultThreads + ")");
        if (Arrays.deepEquals(CZeroPar, ODemo)) {
            System.out.println("-> ZERO TEST (A * 0 = 0): SUCCESS");
        } else {
            System.out.println("-> ZERO TEST (A * 0 = 0): ERROR");
        }

        // ---------------------------------------------------------------------
        // STAGE 2: Cross-Method Tolerance & Thread-Safety Validation
        // ---------------------------------------------------------------------
        Random rng = new Random(SEED);
        if (n <= 0) {
            n = 64;
        }

        System.out.println("\n========================================================================");
        System.out.println("   STAGE 2: MULTI-THREAD CROSS-VALIDATION & THREAD-SAFETY SUITE");
        System.out.println("========================================================================");
        System.out.printf("Numerical tolerance settings: abs_tol = %.1e, rel_tol = %.1e\n", ABS_TOL, REL_TOL);
        System.out.println("Fixed random seed: " + SEED);
        System.out.println("Matrix dimension : " + n + " x " + n);

        double[][] A = new double[n][n];
        double[][] B = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = rng.nextDouble() * 10.0;
                B[i][j] = rng.nextDouble() * 10.0;
            }
        }

        // Reference result from Sequential Baseline
        Object[] seqRes = MatrixMultiplication_comparation.multiplySequential(A, B, n);
        double tSeq = (double) seqRes[0];
        double[][] CRef = (double[][]) seqRes[1];

        System.out.println("\n------------------------------------------------------------------------");
        System.out.printf("  * Sequential Baseline (T_seq) : %.6f s [Trusted Reference]\n", tSeq);
        System.out.println("------------------------------------------------------------------------");

        int[] threadCounts = {1, 2, 4, 8, 12, 16,24};
        boolean allPassed = true;

        for (int p : threadCounts) {
            Object[] parRes = MatrixMultiplication_comparation.multiplyParallel(A, B, n, p);
            double tPar = (double) parRes[0];
            double[][] CPar = (double[][]) parRes[1];

            Object[] cmp = matricesAreClose(CRef, CPar, n, ABS_TOL, REL_TOL);
            boolean isClose = (boolean) cmp[0];
            double maxDiff = (double) cmp[1];

            if (!isClose) {
                allPassed = false;
            }

            System.out.printf("  * Parallel (p = %2d threads)   : %.6f s | Max abs diff = %.2e | %s\n",
                    p, tPar, maxDiff, isClose ? "PASS" : "FAIL");
        }

        System.out.println("------------------------------------------------------------------------");
        if (allPassed) {
            System.out.println("-> OVERALL VALIDATION: ALL THREAD CONFIGURATIONS PASSED (RACE-FREE).");
        } else {
            System.out.println("-> OVERALL VALIDATION: FAILED IN ONE OR MORE CONFIGURATIONS.");
        }
        System.out.println("========================================================================\n");
    }
}
