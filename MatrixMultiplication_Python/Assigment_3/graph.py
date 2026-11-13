import csv
import os
import shutil
import statistics
import matplotlib.pyplot as plt


def find_file(candidate_paths):
    """Finds the CSV file in the local directory or default IntelliJ IDEA project path."""
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_summary(csv_path):
    """
    Reads results_assignment_3.csv and computes medians and IQRs per configuration:
    summary[n][key] = {
        'method', 'threads', 'median_time', 'iqr_time', 'median_mem', 'speedup', 'efficiency'
    }
    where key is 'Sequential' or f'Parallel_p{threads}'.
    """
    raw = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            method = row["method"]
            threads = int(row["threads"])
            t = float(row["time_seconds"])
            m = float(row["peak_memory_mb"])

            key = "Sequential" if method == "Sequential" else f"Parallel_p{threads}"
            entry = raw.setdefault(n, {}).setdefault(
                key,
                {
                    "method": method,
                    "threads": threads,
                    "times": [],
                    "mems": [],
                },
            )
            entry["times"].append(t)
            entry["mems"].append(m)

    summary = {}
    for n, configs in sorted(raw.items()):
        summary[n] = {}
        seq_times = sorted(configs["Sequential"]["times"])
        t_seq_med = statistics.median(seq_times)

        for key, vals in configs.items():
            ts = sorted(vals["times"])
            ms = sorted(vals["mems"])
            t_med = statistics.median(ts)
            iqr_t = ts[3] - ts[1] if len(ts) >= 5 else (ts[-1] - ts[0])
            m_med = statistics.median(ms)
            p = vals["threads"]

            speedup = t_seq_med / t_med if t_med > 0 else 0.0
            efficiency = speedup / p if p > 0 else 0.0

            summary[n][key] = {
                "method": vals["method"],
                "threads": p,
                "median_time": t_med,
                "iqr_time": iqr_t,
                "median_mem": m_med,
                "speedup": speedup,
                "efficiency": efficiency,
            }

    return summary


def generate_plots():
    """Generates console summary tables and high-resolution figures for Assignment 3."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_csv = os.path.join(current_dir, "results_assignment_3.csv")

    candidate_paths = [
        local_csv,
        os.path.expanduser("~/IdeaProjects/MatrixMultiplication/results_assignment_3.csv"),
    ]

    csv_path = find_file(candidate_paths)
    if not csv_path:
        print("Error: Could not find 'results_assignment_3.csv'.")
        print("Please run Assigment_3.Main in IntelliJ IDEA first.")
        return

    # Copy CSV into Python's Assigment_3 folder if loaded from IdeaProjects so both stay in sync
    if os.path.abspath(csv_path) != os.path.abspath(local_csv):
        shutil.copy2(csv_path, local_csv)
        print(f"[OK] Synced CSV from Java project to: {local_csv}")

    print(f"Loaded benchmark data from: {csv_path}")
    summary = load_summary(csv_path)

    sizes = sorted(summary.keys())
    thread_counts = sorted(
        val["threads"]
        for key, val in summary[sizes[0]].items()
        if val["method"] == "Parallel"
    )

    # -------------------------------------------------------------------------
    # CONSOLE SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 118)
    print("             ASSIGNMENT 3 SUMMARY: SEQUENTIAL VS PARALLEL SCALING (MEDIANS OVER 5 REPETITIONS)")
    print("=" * 118)
    print(
        f"{'n':<6} | {'Config':<16} | {'Threads (p)':<11} | {'Median Time (s)':<16} | "
        f"{'IQR (s)':<12} | {'Peak Mem (MB)':<14} | {'Speedup S(p)':<13} | {'Efficiency E(p)':<15}"
    )
    print("-" * 118)

    for n in sizes:
        seq = summary[n]["Sequential"]
        print(
            f"{n:<6} | {'Sequential':<16} | {1:<11} | {seq['median_time']:<16.6f} | "
            f"{seq['iqr_time']:<12.6f} | {seq['median_mem']:<14.3f} | {'1.00x':<13} | {'100.0%':<15}"
        )
        for p in thread_counts:
            par = summary[n][f"Parallel_p{p}"]
            sp_str = f"{par['speedup']:.2f}x"
            ef_str = f"{par['efficiency'] * 100.0:.1f}%"
            print(
                f"{n:<6} | {'Parallel':<16} | {p:<11} | {par['median_time']:<16.6f} | "
                f"{par['iqr_time']:<12.6f} | {par['median_mem']:<14.3f} | {sp_str:<13} | {ef_str:<15}"
            )
        print("-" * 118)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    # -------------------------------------------------------------------------
    # PLOT 1: Direct Comparison Sequential vs Multi-Thread across Matrix Sizes (n)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    t_seq_by_n = [summary[n]["Sequential"]["median_time"] for n in sizes]
    ax1.plot(sizes, t_seq_by_n, marker="o", color="#d62728", linewidth=2.5, label="Sequential")
    ax2.plot(sizes, t_seq_by_n, marker="o", color="#d62728", linewidth=2.5, label="Sequential")

    highlight_threads = [
        (1, "Parallel-P1", "#ff7f0e", "s", "--"),
        (2, "Parallel-P2", "#2ca02c", "^", "-"),
        (4, "Parallel-P4", "#1f77b4", "D", "-"),
        (8, "Parallel-P8", "#9467bd", "v", "-"),
        (16, "Parallel-P16", "#8c564b", "*", "-"),
        (32, "Parallel-P32", "#7f7f7f", "P", ":"),
    ]

    for p, label, color, marker, ls in highlight_threads:
        if f"Parallel_p{p}" in summary[sizes[0]]:
            y_par = [summary[n][f"Parallel_p{p}"]["median_time"] for n in sizes]
            ax1.plot(sizes, y_par, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)
            ax2.plot(sizes, y_par, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)

    ax1.set_title("Execution Time vs Matrix Size: Sequential vs Parallel (Linear)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Execution Time (seconds)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Execution Time vs Matrix Size: Sequential vs Parallel (Log-Log)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n) [log2]", fontsize=11)
    ax2.set_ylabel("Median Execution Time (seconds) [log10]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    path_time = os.path.join(current_dir, "assignment3_time_and_overhead.png")
    plt.tight_layout()
    plt.savefig(path_time, dpi=300)
    print(f"[OK] Saved Sequential vs Parallel Time Comparison Plot: {path_time}")
    plt.show()

    # -------------------------------------------------------------------------
    # PLOT 2: Peak Memory Consumption vs Matrix Size and Thread Count
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    mem_seq = [summary[n]["Sequential"]["median_mem"] for n in sizes]
    mem_p1 = [summary[n]["Parallel_p1"]["median_mem"] for n in sizes]
    mem_p8 = [summary[n]["Parallel_p8"]["median_mem"] for n in sizes]
    mem_p16 = [summary[n]["Parallel_p16"]["median_mem"] for n in sizes]
    mem_p32 = [summary[n]["Parallel_p32"]["median_mem"] for n in sizes]

    ax1.plot(sizes, mem_seq, marker="o", linewidth=2, label="Sequential")
    ax1.plot(sizes, mem_p1, marker="s", linewidth=2, label="Parallel-P1")
    ax1.plot(sizes, mem_p8, marker="^", linewidth=2, label="Parallel-P8")
    ax1.plot(sizes, mem_p16, marker="D", linewidth=2, label="Parallel-P16")
    ax1.plot(sizes, mem_p32, marker="*", linewidth=2, label="Parallel-P32")
    ax1.set_title("Peak Memory Consumption vs Matrix Size (n)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    for idx, n in enumerate(sizes):
        y_mem = [summary[n][f"Parallel_p{p}"]["median_mem"] for p in thread_counts]
        ax2.plot(
            thread_counts,
            y_mem,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            label=f"n = {n}",
        )

    ax2.set_xticks(thread_counts)
    ax2.set_title("Peak Memory Consumption vs Thread Count (p)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Threads (p)", fontsize=11)
    ax2.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9, ncol=2)

    path_mem = os.path.join(current_dir, "assignment3_memory.png")
    plt.tight_layout()
    plt.savefig(path_mem, dpi=300)
    print(f"[OK] Saved Memory Plot: {path_mem}")
    plt.show()


if __name__ == "__main__":
    generate_plots()
