#include "Validation.h"
#include "Benchmark.h"

int main(void) {
    // 1. Run correctness validation tests
    test_multiplication(-1);

    // 2. Run experimental benchmark measurements
    run_benchmark();

    return 0;
}