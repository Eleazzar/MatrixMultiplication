# Assignment 1: Basic Matrix Multiplication in Python, Java, and C
**Course:** Big Data (Academic Year 2026-2027)  
**Degree:** Data Science and Engineering (*Grado en Ciencia e Ingeniería de Datos*)  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)  

## 1. Project Overview
This study investigates the performance and scalability of the basic dense matrix multiplication algorithm across three programming languages: **Python**, **Java**, and **C**. 
All implementations use the straightforward triple-loop algorithm ($i-j-k$ loop order) with theoretical arithmetic complexity of $\mathcal{O}(n^3)$. The goal is to evaluate how matrix dimension ($n$) impacts execution runtime and peak memory consumption under controlled, reproducible conditions.

## 2. Experimental Environment and Dependencies
The experiments were conducted on a single machine with the following software configuration:
| Component | Specification / Version | Configuration Details |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Procesador: AMD Ryzen 7 2700 Eight-core Processor, RAM: 32gb |
| **Python** | Python 3.13.9 | Standards librarys + `matplotlib 3.10.x` | Pycharm enviroment |
| **Java** | OpenJDK 25 | HotSpot JVM, 64-bit Server VM | IntelliJ enviroment |
| **C Compiler** | GCC 15.2.0 | Standard `C11`, Release mode | CLion enviroment |
### Python Dependencies
Python only requires `matplotlib` to produce the comparative charts:

## 3. Dataset Generation and Reproducibility

### Parameters
To ensure fair comparisons between programming languages, all key inputs are:
Square matrices with n [16,32,64,128,256,384] dimensions and a Deterministic Seed, the random number generator is initialized with seed 42 in all three implementations. This random values are Uniformly distributed pseudo-random numbers between 0.0 and 10.0

## 4.Build and Run instructions
All languages only need run the main corresponding environment and they generate a file "results_{languages}.csv". However, since Python handles the creation of graphs, it is essential to first run the Java and C mains.
Remember that must C are in the Release mode

## 5.Example results

<img width="873" height="268" alt="image" src="https://github.com/user-attachments/assets/75a536fc-af76-41cb-b52f-d08069e455ba" />

<img width="1387" height="538" alt="image" src="https://github.com/user-attachments/assets/a32e16f4-d03f-4c13-b36f-bec3ad65221b" />

<img width="1384" height="539" alt="image" src="https://github.com/user-attachments/assets/5c4a5ff9-e40d-426c-9b93-a66b3e29d2cb" />


