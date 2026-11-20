# Assignment 4: Distributed Matrix Multiplication
**Course:** Big Data (Academic Year 2026-2027)  
**Degree:** Data Science and Engineering (*Grado en Ciencia e Ingeniería de Datos*)  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)

## 1. Project Overview
This study investigates how the number of worker processes affects execution time and memory consumption in a distributed matrix multiplication implementation using **Dask Distributed** in **Python**, compared against its equivalent sequential baseline and the shared-memory parallel implementation in **Java** from Assignment 3.
All implementations use the same basic triple-loop algorithm ($i-j-k$ loop order) with theoretical arithmetic complexity of $\mathcal{O}(n^3)$, reusing the validated sequential baseline from Assignment 1 and applying a 1D row-block partitioning strategy across workers.

## 2. Experimental Environment and Dependencies
The experiments were conducted on the same machine as Assignments 1, 2, and 3 with the following configuration:
| Component | Specification / Version | Configuration Details |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Procesador: AMD Ryzen 7 2700 Eight-core Processor (8 physical cores, 16 logical processors), RAM: 32gb |
| **Python** | Python 3.13.9 | Standard libraries + `dask[distributed] 2026.8.0` + `matplotlib 3.11.x` | Pycharm environment |
| **Java (Comparison)** | OpenJDK 21 | HotSpot JVM, 64-bit Server VM | IntelliJ environment |

### Python Dependencies
Remember that we do not use external linear algebra libraries (such as NumPy) for the multiplication kernel. Python requires **dask[distributed]** to manage the local multi-process cluster and **matplotlib** to process the CSV data and produce the comparative charts.

## 3. Dataset Generation and Reproducibility
To ensure fair and reproducible comparisons between the sequential baseline and all distributed worker configurations:
- The matrix multiplication and validation methods from Assignment 1 in Python were used as the base.
- Square dense matrices with $n = [16, 32, 64, 128, 256, 384]$ dimensions.
- Worker configurations evaluated for every matrix dimension under strong scaling: $w = [1, 2, 4, 8]$ independent worker processes.
- All configurations use a deterministic seed: the random number generator is initialized with seed 42 just like Assignments 1, 2, and 3.

## 4. Build and Run Instructions
To run the entire experiment:
1. In **Python (PyCharm)**, install the distributed dependency if not already installed: `pip install "dask[distributed]" matplotlib`.
2. Run main.py inside the Assigment_4 package. When it finishes, it will execute the two-stage validation suite, run the benchmark generating **results_assignment_4.csv**, print the summary table in the terminal, and generate the comparative graphs.
Remember that Assigment_4 reuses methods directly from Assigment_1 in Python and reads results_assignment_3.csv, so do not download just the Assigment_4 package; you must download the complete environment.

## 5. Example results

<img width="1461" height="573" alt="image" src="https://github.com/user-attachments/assets/af06bee1-9b88-49f5-9663-ec745d2b1fa2" />
<img width="1458" height="575" alt="image" src="https://github.com/user-attachments/assets/527f29b3-dee4-4f3b-b49e-4b6fae0b6bf1" />
<img width="1465" height="571" alt="image" src="https://github.com/user-attachments/assets/6a22d0a0-47d6-41a4-81f9-126b2c5c439e" />

## 6. Conclusions
The results lead us to conclude that distributed execution is not always faster than sequential execution; for small workloads, the overhead associated with partitioning, serialization, and data transfer between worker processes outweighs any computational gains, making the sequential approach the better option. When working with medium- or large-sized matrices, distributed execution allows for a significant reduction in execution time (achieving a speedup of up to 4.97x for n=384 with 8 worker processes).
However, although threads may appear to yield better results, the different programming paradigms employed mean that we cannot definitively claim, in general terms, that threads are superior to a distributed system.
