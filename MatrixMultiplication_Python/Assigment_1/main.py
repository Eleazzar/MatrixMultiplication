
from Assigment_1.Benchmark import run_benchmark
from Assigment_1.Validation import test_multiplication
from Assigment_1.graph import generate_plots


if __name__ == "__main__":
    # 1. Run correctness validation tests
    test_multiplication()

    # 2. Run experimental benchmark measurements
    run_benchmark()

    # 3. Generate comparative performance plots (Python vs Java vs C)
    generate_plots()