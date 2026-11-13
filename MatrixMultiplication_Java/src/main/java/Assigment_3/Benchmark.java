package Assigment_3;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class Benchmark {

    /**
     * Computes Mean (average), Median, and Interquartile Range (IQR) for a sample array.
     *
     * @return double[]{mean, median, iqr}
     */
    private static double[] computeStats(double[] values) {
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        double mean = sum / values.length;

        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        double median = sorted[sorted.length / 2];
        double iqr = (sorted.length >= 5) ? (sorted[3] - sorted[1]) : (sorted[sorted.length - 1] - sorted[0]);

        return new double[]{mean, median, iqr};
    }

    /**
     * Measures heap memory consumption (MB) for a given configuration,
     * using the same criteria as Assignment 1 plus active thread stack memory.
     */
    private static double calculatePeakMemoryMB(Runtime runtime, long memBefore, long memAfter, int n, int threads) {
        double measuredMemMB = (memAfter - memBefore) / (1024.0 * 1024.0);
        double theoreticalMemMB = ((n * n * 8.0) + (threads * 16.0 * 1024.0)) / (1024.0 * 1024.0);
        return Math.max(measuredMemMB, theoreticalMemMB);
    }

    /**
     * Runs the Assignment 3 benchmark using the exact same dimensions as Assignments 1 and 2:
     * - Sizes: {16, 32, 64, 128, 256, 384}
     * - Repetitions: 5 per configuration (Sequential and Parallel for each thread count p)
     * - Reports Mean Time, Median Time, IQR, Peak Memory (MB), Speedup S(p), and Efficiency E(p).
     */
    public static void runBenchmark() {
        long seed = 42L;
        Random rng = new Random(seed);

        // Same matrix dimensions used in Assignment 1 and Assignment 2
        int[] sizes = {16, 32, 64, 128, 256, 384};

        int[] threadCounts = {1, 2, 4, 8, 12, 16, 24};
        int repetitions = 5;

        Runtime runtime = Runtime.getRuntime();
        int logicalCores = runtime.availableProcessors();

        List<String> csvRows = new ArrayList<>();

        System.out.println("==========================================================================================");
        System.out.println("          PERFORMANCE BENCHMARK EXECUTION (JAVA - SEQUENTIAL VS PARALLEL)");
        System.out.println("==========================================================================================");
        System.out.println("Detected logical processors: " + logicalCores);
        System.out.println("Matrix sizes: " + Arrays.toString(sizes));
        System.out.println("Thread counts (p): " + Arrays.toString(threadCounts));
        System.out.println("Repetitions per configuration: " + repetitions);
        System.out.println("==========================================================================================");

        for (int n : sizes) {
            System.out.println("\n==========================================================================================");
            System.out.println("                         TESTING MATRIX SIZE " + n + " x " + n);
            System.out.println("==========================================================================================");

            // Warm-up run (excluded from measured samples)
            double[][] AWarmup = Assigment_1.Benchmark.generateMatrix(n, rng);
            double[][] BWarmup = Assigment_1.Benchmark.generateMatrix(n, rng);
            MatrixMultiplication_comparation.multiplySequential(AWarmup, BWarmup, n);
            MatrixMultiplication_comparation.multiplyParallel(AWarmup, BWarmup, n, 4);

            // Pre-generate the 5 random matrix pairs for size n so Sequential and all Parallel
            // thread configurations multiply the exact same matrices in each repetition
            double[][][] inputsA = new double[repetitions][][];
            double[][][] inputsB = new double[repetitions][][];
            for (int r = 0; r < repetitions; r++) {
                inputsA[r] = Assigment_1.Benchmark.generateMatrix(n, rng);
                inputsB[r] = Assigment_1.Benchmark.generateMatrix(n, rng);
            }

            // -----------------------------------------------------------------
            // 1. SEQUENTIAL BASELINE (Same as Assignment 1)
            // -----------------------------------------------------------------
            System.out.println("\n--- [Sequential Baseline] Matrix size " + n + " x " + n + " ---");
            double[] seqTimes = new double[repetitions];
            double[] seqMemories = new double[repetitions];

            for (int rep = 1; rep <= repetitions; rep++) {
                System.gc();
                long memBefore = runtime.totalMemory() - runtime.freeMemory();

                Object[] res = MatrixMultiplication_comparation.multiplySequential(inputsA[rep - 1], inputsB[rep - 1], n);
                double elapsedTime = (double) res[0];

                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                double peakMemoryMB = calculatePeakMemoryMB(runtime, memBefore, memAfter, n, 1);

                seqTimes[rep - 1] = elapsedTime;
                seqMemories[rep - 1] = peakMemoryMB;

                System.out.printf("  Repetition %d/%d: %.5f s | Peak memory: %.3f MB\n",
                        rep, repetitions, elapsedTime, peakMemoryMB);

                csvRows.add(String.format(Locale.US,
                        "%d,Sequential,float64,1,%d,%.6f,%.3f",
                        n, rep, elapsedTime, peakMemoryMB));
            }

            double[] seqTimeStats = computeStats(seqTimes);
            double[] seqMemStats = computeStats(seqMemories);
            double meanSeqTime = seqTimeStats[0];
            double medianSeqTime = seqTimeStats[1];
            double iqrSeqTime = seqTimeStats[2];
            double meanSeqMem = seqMemStats[0];
            double medianSeqMem = seqMemStats[1];

            System.out.printf("  --> SUMMARY (Sequential): Mean = %.5f s | Median = %.5f s | IQR = %.5f s | Mean Mem = %.3f MB | Peak Mem (Median) = %.3f MB\n",
                    meanSeqTime, medianSeqTime, iqrSeqTime, meanSeqMem, medianSeqMem);

            // -----------------------------------------------------------------
            // 2. PARALLEL MULTI-THREAD CONFIGURATIONS (For the same size n)
            // -----------------------------------------------------------------
            for (int p : threadCounts) {
                System.out.println("\n--- [Parallel p = " + p + " threads] Matrix size " + n + " x " + n + " ---");
                double[] parTimes = new double[repetitions];
                double[] parMemories = new double[repetitions];

                for (int rep = 1; rep <= repetitions; rep++) {
                    System.gc();
                    long memBefore = runtime.totalMemory() - runtime.freeMemory();

                    Object[] res = MatrixMultiplication_comparation.multiplyParallel(inputsA[rep - 1], inputsB[rep - 1], n, p);
                    double elapsedTime = (double) res[0];

                    long memAfter = runtime.totalMemory() - runtime.freeMemory();
                    double peakMemoryMB = calculatePeakMemoryMB(runtime, memBefore, memAfter, n, p);

                    parTimes[rep - 1] = elapsedTime;
                    parMemories[rep - 1] = peakMemoryMB;

                    System.out.printf("  Repetition %d/%d: %.5f s | Peak memory: %.3f MB\n",
                            rep, repetitions, elapsedTime, peakMemoryMB);

                    csvRows.add(String.format(Locale.US,
                            "%d,Parallel,float64,%d,%d,%.6f,%.3f",
                            n, p, rep, elapsedTime, peakMemoryMB));
                }

                double[] parTimeStats = computeStats(parTimes);
                double[] parMemStats = computeStats(parMemories);
                double meanParTime = parTimeStats[0];
                double medianParTime = parTimeStats[1];
                double iqrParTime = parTimeStats[2];
                double meanParMem = parMemStats[0];
                double medianParMem = parMemStats[1];

                // Speedup and Efficiency computed both by Mean and by Median
                double speedupMean = meanSeqTime / meanParTime;
                double efficiencyMean = speedupMean / p;
                double speedupMedian = medianSeqTime / medianParTime;
                double efficiencyMedian = speedupMedian / p;

                System.out.printf("  --> SUMMARY (p = %2d)    : Mean = %.5f s | Median = %.5f s | IQR = %.5f s | Mean Mem = %.3f MB | Peak Mem (Median) = %.3f MB\n",
                        p, meanParTime, medianParTime, iqrParTime, meanParMem, medianParMem);
                System.out.printf("      METRICS (p = %2d)    : Speedup (Mean) = %.2fx (Eff = %.1f%%) | Speedup (Median) = %.2fx (Eff = %.1f%%)\n",
                        p, speedupMean, efficiencyMean * 100.0, speedupMedian, efficiencyMedian * 100.0);
            }
        }

        // Export raw data to CSV
        String filename = "results_assignment_3.csv";
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("n,method,numeric_type,threads,repetition,time_seconds,peak_memory_mb");
            for (String row : csvRows) {
                writer.println(row);
            }
            System.out.println("\n==========================================================================================");
            System.out.println("Benchmark finished! Raw data saved to: " + filename);
            System.out.println("==========================================================================================");
        } catch (IOException e) {
            System.err.println("Error saving CSV file: " + e.getMessage());
        }
    }
}
