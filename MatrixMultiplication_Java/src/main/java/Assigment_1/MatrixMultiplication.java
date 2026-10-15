package Assigment_1;

public class MatrixMultiplication {

    /**
     * Multiplies two n x n dense square matrices A and B using the standard
     * naive triple-loop algorithm (i-j-k loop order) with O(n^3) arithmetic complexity.
     *
     * @param A Input matrix A of dimensions n x n.
     * @param B Input matrix B of dimensions n x n.
     * @param n Matrix dimension.
     * @return An Object array simulating a tuple: {Double elapsedTime, double[][] C}.
     */
    public static Object[] multiplication(double[][] A, double[][] B, int n) {
        long start = System.nanoTime();

        double[][] C = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        long end = System.nanoTime();

        double elapsedTime = (end - start) / 1_000_000_000.0;

        return new Object[]{elapsedTime, C};
    }
}
