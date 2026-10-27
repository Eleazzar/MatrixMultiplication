import csv
import os
import statistics
import matplotlib.pyplot as plt


def find_file(name, candidate_paths):
    """Finds the CSV file in local directory or default IDE paths."""
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_medians(csv_path):
    """Reads a CSV file and computes median time and memory for each matrix dimension n."""
    data = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Accepts both English and Spanish column names
            n = int(row.get("n") or row.get("tamanio_n") or row.get("size"))
            t = float(row.get("time_seconds") or row.get("tiempo_segundos"))
            m = float(row.get("peak_memory_mb") or row.get("memoria_pico_mb"))

            if n not in data:
                data[n] = {"times": [], "memories": []}
            data[n]["times"].append(t)
            data[n]["memories"].append(m)

    sizes = sorted(data.keys())
    median_times = [statistics.median(data[n]["times"]) for n in sizes]
    median_memories = [statistics.median(data[n]["memories"]) for n in sizes]

    return sizes, median_times, median_memories


def generate_plots():
    """Generates comparative figures across Python, Java, and C."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    python_paths = [
        os.path.join(current_dir, "resultados_python.csv")
    ]
    java_paths = [
        os.path.join(current_dir, "resultados_java.csv"),
        os.path.expanduser("~/IdeaProjects/MatrixMultiplication/resultados_java.csv")
    ]
    c_paths = [
        os.path.join(current_dir, "resultados_c.csv"),
        os.path.expanduser("~/CLionProjects/MatrixMultiplication/cmake-build-release/resultados_c.csv"),
        os.path.expanduser("~/CLionProjects/MatrixMultiplication/cmake-build-debug/resultados_c.csv")
    ]

    csv_py = find_file("Python", python_paths)
    csv_java = find_file("Java", java_paths)
    csv_c = find_file("C", c_paths)

    if not (csv_py and csv_java and csv_c):
        print("Error: Missing CSV files:")
        print(f"  - Python: {'Found' if csv_py else 'NOT found'}")
        print(f"  - Java:   {'Found' if csv_java else 'NOT found'}")
        print(f"  - C:      {'Found' if csv_c else 'NOT found'}")
        return

    print("Loaded data files:")
    print(f"  * Python: {csv_py}")
    print(f"  * Java:   {csv_java}")
    print(f"  * C:      {csv_c}")

    n_py, t_py, m_py = load_medians(csv_py)
    _, t_java, m_java = load_medians(csv_java)
    _, t_c, m_c = load_medians(csv_c)

    # SUMMARY CONSOLE TABLE

    print("\n" + "=" * 105)
    print("                              PERFORMANCE COMPARISON SUMMARY (MEDIANS)")
    print("=" * 105)
    print(f"{'n':<6} | {'Python Time (s)':<16} | {'Java Time (s)':<15} | {'C Time (s)':<13} | {'Python Mem (MB)':<16} | {'Java Mem (MB)':<14} | {'C Mem (MB)':<11}")
    print("-" * 105)
    for i, n in enumerate(n_py):
        print(f"{n:<6} | {t_py[i]:<16.5f} | {t_java[i]:<15.5f} | {t_c[i]:<13.5f} | {m_py[i]:<16.3f} | {m_java[i]:<14.3f} | {m_c[i]:<11.3f}")
    print("=" * 105 + "\n")

    # 1. EXECUTION TIME PLOT (Linear vs Log-Log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1.1 Linear Scale
    ax1.plot(n_py, t_py, marker="o", color="#1f77b4", linewidth=2, label="Python (CPython)")
    ax1.plot(n_py, t_java, marker="s", color="#ff7f0e", linewidth=2, label="Java (HotSpot JIT)")
    ax1.plot(n_py, t_c, marker="^", color="#2ca02c", linewidth=2, label="C (GCC -O3)")
    ax1.set_title("Execution Time (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # 1.2 Log-Log Scale
    ax2.plot(n_py, t_py, marker="o", color="#1f77b4", linewidth=2, label="Python (CPython)")
    ax2.plot(n_py, t_java, marker="s", color="#ff7f0e", linewidth=2, label="Java (HotSpot JIT)")
    ax2.plot(n_py, t_c, marker="^", color="#2ca02c", linewidth=2, label="C (GCC -O3)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Execution Time (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n) [base 2]", fontsize=11)
    ax2.set_ylabel("Execution Time (seconds) [base 10]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    time_img_path = os.path.join(current_dir, "comparative_time.png")
    plt.tight_layout()
    plt.savefig(time_img_path, dpi=300)
    print(f"[OK] Time comparison plot saved: {time_img_path}")
    plt.show()

    # 2. PEAK MEMORY PLOT (Linear vs Log-Log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 2.1 Linear Scale
    ax1.plot(n_py, m_py, marker="o", color="#1f77b4", linewidth=2, label="Python (tracemalloc)")
    ax1.plot(n_py, m_java, marker="s", color="#ff7f0e", linewidth=2, label="Java (JVM Heap)")
    ax1.plot(n_py, m_c, marker="^", color="#2ca02c", linewidth=2, label="C (Physical memory)")
    ax1.set_title("Peak Memory Consumption (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Memory (MB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # 2.2 Log-Log Scale
    ax2.plot(n_py, m_py, marker="o", color="#1f77b4", linewidth=2, label="Python (tracemalloc)")
    ax2.plot(n_py, m_java, marker="s", color="#ff7f0e", linewidth=2, label="Java (JVM Heap)")
    ax2.plot(n_py, m_c, marker="^", color="#2ca02c", linewidth=2, label="C (Physical memory)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Peak Memory Consumption (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n) [base 2]", fontsize=11)
    ax2.set_ylabel("Memory (MB) [base 10]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    mem_img_path = os.path.join(current_dir, "comparative_memory.png")
    plt.tight_layout()
    plt.savefig(mem_img_path, dpi=300)
    print(f"[OK] Memory comparison plot saved: {mem_img_path}")
    plt.show()
