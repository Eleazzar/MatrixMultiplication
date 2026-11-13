# Assignment 3: Parallel Matrix Multiplication
**Course:** Big Data (Academic Year 2026-2027)  
**Degree:** Data Science and Engineering (*Grado en Ciencia e Ingeniería de Datos*)  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)

## 1. Project Overview
This study investigates how the number of threads affects execution time and memory consumption in a shared-memory parallel matrix multiplication implementation using **Java**, compared against its equivalent sequential baseline.
Both implementations use the same basic triple-loop algorithm ($i-j-k$ loop order) with theoretical arithmetic complexity of $\mathcal{O}(n^3)$, reusing the validated sequential baseline from Assignment 1.

## 2. Experimental Environment and Dependencies
The experiments were conducted on the same machine as Assignments 1 and 2 with the following configuration:
| Component | Specification / Version | Configuration Details |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Procesador: AMD Ryzen 7 2700 Eight-core Processor (8 physical cores, 16 logical processors), RAM: 32gb |
| **Java** | OpenJDK 21 / 25 | HotSpot JVM, 64-bit Server VM | IntelliJ environment |
| **Python** | Python 3.13.9 | Standard libraries + `matplotlib 3.10.x` | Pycharm environment |
### Python Dependencies
Remember that we do not use external linear algebra libraries (such as NumPy) and Python only requires `matplotlib` to process the CSV data and produce the comparative charts.

## 3. Dataset Generation and Reproducibility
To ensure fair and reproducible comparisons between the sequential baseline and all parallel thread configurations:
- The matrix multiplication and generator methods from Assignment 1 in Java were used as the base.
- Square dense matrices with $n = [16, 32, 64, 128, 256, 384]$ dimensions.
- Thread configurations evaluated for every matrix dimension: $p = [1, 2, 4, 8, 12, 16, 24, 32]$ threads (covering 1-thread overhead, 8 physical cores, 16 logical processors, and oversubscription).
- All configurations use a deterministic seed: the random number generator is initialized with seed `42` just like Assignments 1 and 2.

## 4. Build and Run Instructions
To run the entire experiment:
1. First, in **Java (IntelliJ)**, run the `Main` class inside the `Assigment_3` package. It will execute the validation tests and the benchmark, generating the file `results_assignment_3.csv`.
2. Next, in **Python (PyCharm)**, run `graph.py` inside the `Assigment_3` package. Just like in Assignment 1, Python will automatically search for `results_assignment_3.csv` in the default Java project location (as well as within its own folder), print the summary table in the terminal, and generate the comparative graphs.
Note that `Assigment_3` reuses methods directly from `Assigment_1` in Java, so do not download just the `Assigment_3` package; you must download the complete environment.

## 5. Example results

<img width="1220" height="481" alt="image" src="https://github.com/user-attachments/assets/3c726396-68a7-4760-b863-b2a83d1227f0" />
<img width="1221" height="476" alt="image" src="https://github.com/user-attachments/assets/fd5752bb-cb88-4d29-8850-c747fddba5c2" />
<img width="733" height="731" alt="image" src="https://github.com/user-attachments/assets/b0695cca-c487-49c5-ad6e-08aae622c034" />

## 6. Conclusions
This study has successfully evaluated the impact of shared-memory multithreading on matrix multiplication in Java. The results lead to the following conclusions:
Parallel execution is not always faster than sequential execution; for small workloads, the overhead involved in creating and synchronizing threads outweighs any potential benefits, making the sequential approach the better choice.
When working with medium- or large-sized matrices (n≥128), multithreading allows for a significant reduction in execution time. This demonstrates that distributing matrix rows across different threads is an effective way to accelerate computation-intensive tasks without requiring complex synchronization mechanisms. 



