from Assigment_2.Validation import test_multiplication
from Assigment_2.Benchmark import run_benchmark
from Assigment_2.graph import generate_plots


if __name__ == "__main__":
    # 1. Validate numerical correctness across all Dense and Sparse CSR implementations
    test_multiplication()

    # 2. Run controlled experimental benchmarks (Part A: Dense Optimisations, Part B: Sparse CSR)
    run_benchmark()

    # 3. Generate summary tables and publication-ready plots
    generate_plots()
