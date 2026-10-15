package Assigment_1;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Benchmark {

    /**
     * Generates an n x n square matrix filled with random numbers using a provided RNG.
     */
    public static double[][] generateMatrix(int n, Random rng) {
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = rng.nextDouble() * 10.0;
            }
        }
        return matrix;
    }

    /**
     * Runs the empirical performance benchmark:
     * - Warm-up iterations for JVM HotSpot JIT compilation.
     * - 5 measured repetitions per matrix size.
     * - Computes median execution time and interquartile range (IQR).
     * - Exports measurements to resultados_java.csv.
     */
    public static void runBenchmark() {
        long seed = 42L;
        Random rng = new Random(seed);
        int[] sizes = {16, 32, 64, 128, 256, 384};
        int repetitions = 5;

        List<String> csvRows = new ArrayList<>();

        System.out.println("=========================================================");
        System.out.println("        PERFORMANCE BENCHMARK EXECUTION (JAVA)");
        System.out.println("=========================================================");

        Runtime runtime = Runtime.getRuntime();

        for (int n : sizes) {
            System.out.println("\n--- Testing matrix size " + n + " x " + n + " ---");

            // -------------------------------------------------------------
            // Warm-up run (excluded from measured samples)
            // Allows JVM JIT compiler to optimize bytecode into machine code
            // -------------------------------------------------------------
            double[][] AWarmup = generateMatrix(n, rng);
            double[][] BWarmup = generateMatrix(n, rng);
            MatrixMultiplication.multiplication(AWarmup, BWarmup, n);

            double[] times = new double[repetitions];
            double[] memories = new double[repetitions];

            for (int rep = 1; rep <= repetitions; rep++) {
                double[][] A = generateMatrix(n, rng);
                double[][] B = generateMatrix(n, rng);

                // Suggest garbage collection before measurement
                System.gc();
                long memBefore = runtime.totalMemory() - runtime.freeMemory();

                // Timed multiplication call
                Object[] res = MatrixMultiplication.multiplication(A, B, n);
                double elapsedTime = (double) res[0];
                double[][] C = (double[][]) res[1];

                // Memory calculation in MB
                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                double measuredMemMB = (memAfter - memBefore) / (1024.0 * 1024.0);
                double theoreticalMemMB = (n * n * 8.0) / (1024.0 * 1024.0);
                double peakMemoryMB = Math.max(measuredMemMB, theoreticalMemMB);

                times[rep - 1] = elapsedTime;
                memories[rep - 1] = peakMemoryMB;

                System.out.printf("  Repetition %d/%d: %.5f s | Peak memory: %.3f MB\n",
                        rep, repetitions, elapsedTime, peakMemoryMB);

                csvRows.add(n + "," + rep + "," +
                        String.format(java.util.Locale.US, "%.5f", elapsedTime) + "," +
                        String.format(java.util.Locale.US, "%.3f", peakMemoryMB));
            }

            // Compute Median and Interquartile Range (IQR)
            Arrays.sort(times);
            double medianTime = times[2];
            double iqrTime = times[3] - times[1];

            Arrays.sort(memories);
            double medianMemory = memories[2];

            System.out.printf("  --> SUMMARY: Median = %.5f s | IQR = %.5f s | Peak Mem = %.3f MB\n",
                    medianTime, iqrTime, medianMemory);
        }

        // Export data to CSV
        String filename = "resultados_java.csv";
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("n,repetition,time_seconds,peak_memory_mb");
            for (String row : csvRows) {
                writer.println(row);
            }
            System.out.println("\n=========================================================");
            System.out.println("Benchmark finished! Raw data saved to: " + filename);
            System.out.println("=========================================================");
        } catch (IOException e) {
            System.err.println("Error saving CSV file: " + e.getMessage());
        }
    }
}
