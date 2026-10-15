package Assigment_1;

import java.util.Arrays;
import java.util.Random;

public class Validation {

    /**
     * Prints a matrix to the console in a clear grid format.
     */
    public static void printMatrix(double[][] matrix, String label) {
        System.out.println("\n" + label + ":");
        for (double[] row : matrix) {
            System.out.print("  [ ");
            for (double val : row) {
                System.out.printf("%6.2f ", val);
            }
            System.out.println("]");
        }
    }

    public static void testMultiplication() {
        testMultiplication(-1);
    }

    /**
     * Correctness validation test suite:
     * - TEST 1: Identity Matrix (A * I = A)
     * - TEST 2: Zero Matrix (A * 0 = 0)
     * - TEST 3: Random Matrix Multiplication (size n x n)
     */
    public static void testMultiplication(int n) {
        long SEED = 42L;
        Random rng = new Random(SEED);

        System.out.println("=================================================================");
        System.out.println("        VALIDATION SUITE: MATRIX MULTIPLICATION (JAVA)");
        System.out.println("=================================================================");

        // -------------------------------------------------------------
        // TEST 1: Identity Matrix (A * I = A)
        // -------------------------------------------------------------
        System.out.println("\n[TEST 1] Identity Matrix Property: A * I = A");
        System.out.println("Explanation: The identity matrix (I) is the neutral element in matrix");
        System.out.println("multiplication. Multiplying any matrix by I must return the original matrix.\n");

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

        Object[] resIdent = MatrixMultiplication.multiplication(ADemo, IDemo, sizeDemo);
        double[][] CIdentity = (double[][]) resIdent[1];

        printMatrix(ADemo, "Matrix A (original)");
        printMatrix(IDemo, "Identity Matrix I");
        printMatrix(CIdentity, "Result A * I");

        if (Arrays.deepEquals(CIdentity, ADemo)) {
            System.out.println("\n-> EVALUATION: SUCCESS (A * I matches matrix A exactly).");
        } else {
            System.out.println("\n-> EVALUATION: ERROR (Result differs from matrix A).");
        }

        // -------------------------------------------------------------
        // TEST 2: Zero Matrix (A * 0 = 0)
        // -------------------------------------------------------------
        System.out.println("\n-----------------------------------------------------------------");
        System.out.println("\n[TEST 2] Zero Matrix Property: A * 0 = 0");
        System.out.println("Explanation: Multiplying any matrix by a zero matrix (all entries 0.0)");
        System.out.println("must produce an identical zero matrix.\n");

        double[][] ODemo = new double[sizeDemo][sizeDemo];

        Object[] resZero = MatrixMultiplication.multiplication(ADemo, ODemo, sizeDemo);
        double[][] CZero = (double[][]) resZero[1];

        printMatrix(ODemo, "Zero Matrix O");
        printMatrix(CZero, "Result A * O");

        if (Arrays.deepEquals(CZero, ODemo)) {
            System.out.println("\n-> EVALUATION: SUCCESS (All entries in A * O are 0.0).");
        } else {
            System.out.println("\n-> EVALUATION: ERROR (Found non-zero entries in result).");
        }

        // -------------------------------------------------------------
        // TEST 3: Random Matrix Multiplication
        // -------------------------------------------------------------
        System.out.println("\n-----------------------------------------------------------------");
        System.out.println("\n[TEST 3] Random Matrix Multiplication");

        if (n <= 0) {
            n = rng.nextInt(256) + 1;
            System.out.println("No valid 'n' provided. Randomly chosen: n = " + n + " (range [1, 256])");
        } else {
            System.out.println("Matrix dimension received as argument: n = " + n);
        }

        System.out.println("Fixed random seed: " + SEED);

        double[][] ARand = new double[n][n];
        double[][] BRand = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ARand[i][j] = rng.nextDouble() * 10.0;
                BRand[i][j] = rng.nextDouble() * 10.0;
            }
        }

        System.out.println("Executing multiplication kernel for square matrices " + n + "x" + n + "...");
        Object[] resRand = MatrixMultiplication.multiplication(ARand, BRand, n);
        double elapsedTime = (double) resRand[0];
        double[][] CRand = (double[][]) resRand[1];

        System.out.println("-> Multiplication finished successfully.");
        System.out.println("-> Result matrix dimensions: " + CRand.length + " rows x " + CRand[0].length + " columns.");
        System.out.printf("-> Kernel execution time: %.6f seconds.\n", elapsedTime);
        System.out.println("=================================================================");
    }
}
