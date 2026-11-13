package Assigment_3;

public class MatrixMultiplication_comparation {

    public static Object[] multiplySequential(double[][] A, double[][] B, int n) {
        return Assigment_1.MatrixMultiplication.multiplication(A, B, n);
    }

    public static Object[] multiplyParallel(double[][] A, double[][] B, int n, int numThreads) {
        if (numThreads <= 0) {
            throw new IllegalArgumentException("Number of threads must be at least 1.");
        }

        long start = System.nanoTime();

        double[][] C = new double[n][n];
        Thread[] threads = new Thread[numThreads];

        // Row-block partitioning: distribute n rows as evenly as possible among p threads
        int baseRows = n / numThreads;
        int remainder = n % numThreads;
        int currentRow = 0;

        for (int t = 0; t < numThreads; t++) {
            int rowsForThread = baseRows + (t < remainder ? 1 : 0);
            final int startRow = currentRow;
            final int endRow = currentRow + rowsForThread;
            currentRow = endRow;

            threads[t] = new Thread(() -> {
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < n; j++) {
                        for (int k = 0; k < n; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            });

            threads[t].start();
        }

        // Wait for all worker threads to finish their assigned rows
        for (int t = 0; t < numThreads; t++) {
            try {
                threads[t].join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Thread execution interrupted during matrix multiplication", e);
            }
        }

        long end = System.nanoTime();
        double elapsedTime = (end - start) / 1_000_000_000.0;

        return new Object[]{elapsedTime, C};
    }
}
