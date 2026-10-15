cmake_minimum_required(VERSION 4.3)
project(MatrixMultiplication C)

set(CMAKE_C_STANDARD 11)

add_executable(MatrixMultiplication
        main.c
        MatrixMultiplication.c
        MatrixMultiplication.h
        Validation.c
        Validation.h
        Benchmark.c
        Benchmark.h
)
