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
| **Operating System** | Windows 11 (64-bit) | Default power plan |
| **Python** | Python 3.13.9 | Standards librarys + `matplotlib 3.10.x` | Pycharm enviroment |
| **Java** | OpenJDK 25 | HotSpot JVM, 64-bit Server VM | IntelliJ enviroment |
| **C Compiler** | GCC 15.2.0 | Standard `C11`, Release mode with `-O3` | Clion enviroment |
### Python Dependencies
Python only requires `matplotlib` to produce the comparative charts:
```bash
pip install matplotlib
Note: Python uses the standard library modules time, tracemalloc, and csv for benchmarking; no external numerical libraries (such as NumPy)
```
## 3. Dataset Generation and Reproducibility
