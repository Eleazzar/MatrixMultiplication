# Assignment 2: Optimised Matrix Multiplication Approaches and Sparse Matrices

**Course:** Big Data (Academic Year 2026-2027)  
**Degree:** Data Science and Engineering (*Grado en Ciencia e Ingeniería de Datos*)  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)  

## 1.Project Overview

This study investigates two key questions in matrix multiplication using Python: how loop and memory-access optimizations affect dense matrix multiplication and under which sparsity conditions a compressed sparse representation offers advantages in execution time and memory consumption.
Starting from the validated basic implementation from Assignment 1, two sequential dense optimizations were implemented: Loop Reordering and Matrix Transposition.

For sparse matrices, a custom Compressed Sparse Row (CSR) representation and multiplication algorithm were developed and evaluated across different percentages of zero entries.

## 2.Experimental Environment Depedencies

The experiments were conducted on the same machine as Assignment 1 with the following configuration:
| Component | Specification / Version | Configuration Details |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Procesador: AMD Ryzen 7 2700 Eight-core Processor, RAM: 32gb |
| **Python** | Python 3.13.9 | Standards librarys + `matplotlib 3.10.x` | Pycharm enviroment |

Remember that we don't use external libraries such as numpy and Python requires matplotlib.

## 3.Dataset Generation and Reproducibility

To ensure fair and reproducible comparisons across all methods:
The matrix from task 1 was used as the matrix base.
Square dense matrices with n = [16,32,64,128,256,384] dimensions.
Square sparse matrices of fixed dimension n = 128 tested across sparsity levels [0%,25%,50%,75%,95%] of zeros.
All use deterministic Seed, the random number generator is initialized with seed 42 just like assignment 1.

## 4.Build and Run Instructions

To run the entire experiment, run the main assigment 2. When it finish, generate a file csv, graphs and results in the terminal.
Note that assignment 2 uses methods from assignment 1, so do not download just the assignment 2 package, you must download the complete environment.

## 5.Example results
<img width="1214" height="476" alt="image" src="https://github.com/user-attachments/assets/dae86bac-f221-4c6d-b5d3-8d2b02515650" />
<img width="1220" height="474" alt="image" src="https://github.com/user-attachments/assets/d0cac3f9-b07c-40d7-8926-dc185c0c7e09" />
<img width="1220" height="475" alt="image" src="https://github.com/user-attachments/assets/a3882dad-15da-45c1-b69d-15260cee0621" />
<img width="1215" height="469" alt="image" src="https://github.com/user-attachments/assets/1e87a3a5-6e0b-4d85-be16-3b6e37595251" />


## 6.Conclusions

Dense matrix optimizations clearly outperform the baseline case without changing the complexity by taking advantage of row-wise memory access and cache locality (achieving 1.72 to 2.55 speedups).
However, to refine the optimization, it is important to take into account the pros and cons, since if, for example, we require the highest possible speed regardless of memory, the transposed matrix optimization is unbeatable. Otherwise, if we want to save memory, the best choice is loop reordering.

The Sparse matrix optimization is faster when the percentage of zeros exceeds 25-30%. However, it doesn't outperform the base matrix in memory usage until the percentage of zeros exceeds 95%.



