import csv
import os
import statistics
import matplotlib.pyplot as plt


def find_file(candidate_paths):
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_summary(csv_path, worker_col="workers", method_prefix="Distributed"):
    raw = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            method = row["method"]
            w = int(row.get(worker_col) or row.get("threads") or 1)
            t = float(row["time_seconds"])
            m = float(row["peak_memory_mb"])

            key = "Sequential" if method == "Sequential" else f"{method_prefix}_w{w}"
            entry = raw.setdefault(n, {}).setdefault(
                key,
                {
                    "method": method,
                    "workers": w,
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
            w = vals["workers"]

            speedup = t_seq_med / t_med if t_med > 0 else 0.0
            efficiency = speedup / w if w > 0 else 0.0

            summary[n][key] = {
                "method": vals["method"],
                "workers": w,
                "median_time": t_med,
                "iqr_time": iqr_t,
                "median_mem": m_med,
                "speedup": speedup,
                "efficiency": efficiency,
            }

    return summary


def generate_plots():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "results_assignment_4.csv")

    if not os.path.exists(csv_path):
        print("Error: Could not find 'results_assignment_4.csv'.")
        print("Please run Assigment_4/Benchmark.py first.")
        return

    print(f"Loaded benchmark data from: {csv_path}")
    summary = load_summary(csv_path, worker_col="workers", method_prefix="Distributed")

    sizes = sorted(summary.keys())
    worker_counts = sorted(
        val["workers"]
        for key, val in summary[sizes[0]].items()
        if val["method"] != "Sequential"
    )

    # CONSOLE SUMMARY TABLE
    print("\n" + "=" * 118)
    print("           ASSIGNMENT 4 SUMMARY: SEQUENTIAL VS DISTRIBUTED SCALING (MEDIANS OVER 5 REPETITIONS)")
    print("=" * 118)
    print(
        f"{'n':<6} | {'Config':<16} | {'Workers (w)':<11} | {'Median Time (s)':<16} | "
        f"{'IQR (s)':<12} | {'Peak Mem (MB)':<14} | {'Speedup S(w)':<13} | {'Efficiency E(w)':<15}"
    )
    print("-" * 118)

    for n in sizes:
        seq = summary[n]["Sequential"]
        print(
            f"{n:<6} | {'Sequential':<16} | {1:<11} | {seq['median_time']:<16.6f} | "
            f"{seq['iqr_time']:<12.6f} | {seq['median_mem']:<14.3f} | {'1.00x':<13} | {'100.0%':<15}"
        )
        for w in worker_counts:
            dist = summary[n][f"Distributed_w{w}"]
            sp_str = f"{dist['speedup']:.2f}x"
            ef_str = f"{dist['efficiency'] * 100.0:.1f}%"
            print(
                f"{n:<6} | {'Distributed':<16} | {w:<11} | {dist['median_time']:<16.6f} | "
                f"{dist['iqr_time']:<12.6f} | {dist['median_mem']:<14.3f} | {sp_str:<13} | {ef_str:<15}"
            )
        print("-" * 118)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    # PLOT 1: Execution Time vs Matrix Size: Sequential vs Distributed (Linear & Log-Log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    t_seq_by_n = [summary[n]["Sequential"]["median_time"] for n in sizes]
    ax1.plot(sizes, t_seq_by_n, marker="o", color="#d62728", linewidth=2.5, label="Sequential")
    ax2.plot(sizes, t_seq_by_n, marker="o", color="#d62728", linewidth=2.5, label="Sequential")

    highlight_workers = [
        (1, "Distributed-W1", "#ff7f0e", "s", "--"),
        (2, "Distributed-W2", "#2ca02c", "^", "-"),
        (4, "Distributed-W4", "#1f77b4", "D", "-"),
        (8, "Distributed-W8", "#9467bd", "v", "-"),
    ]

    for w, label, color, marker, ls in highlight_workers:
        if f"Distributed_w{w}" in summary[sizes[0]]:
            y_dist = [summary[n][f"Distributed_w{w}"]["median_time"] for n in sizes]
            ax1.plot(sizes, y_dist, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)
            ax2.plot(sizes, y_dist, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)

    ax1.set_title("Execution Time vs Matrix Size: Sequential vs Distributed (Linear)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Execution Time (seconds)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Execution Time vs Matrix Size: Sequential vs Distributed (Log-Log)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n) [log2]", fontsize=11)
    ax2.set_ylabel("Median Execution Time (seconds) [log10]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    path_time = os.path.join(current_dir, "assignment4_time_and_overhead.png")
    plt.tight_layout()
    plt.savefig(path_time, dpi=300)
    print(f"[OK] Saved Sequential vs Distributed Time Comparison Plot: {path_time}")
    plt.show()

    # PLOT 2: Peak Memory Consumption vs Matrix Size and Worker Count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    mem_seq = [summary[n]["Sequential"]["median_mem"] for n in sizes]
    ax1.plot(sizes, mem_seq, marker="o", linewidth=2, label="Sequential")

    for w, label, _, marker, _ in highlight_workers:
        if f"Distributed_w{w}" in summary[sizes[0]]:
            mem_w = [summary[n][f"Distributed_w{w}"]["median_mem"] for n in sizes]
            ax1.plot(sizes, mem_w, marker=marker, linewidth=2, label=label)

    ax1.set_title("Peak Memory Consumption vs Matrix Size (n)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    for idx, n in enumerate(sizes):
        y_mem = [summary[n][f"Distributed_w{w}"]["median_mem"] for w in worker_counts]
        ax2.plot(
            worker_counts,
            y_mem,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            label=f"n = {n}",
        )

    ax2.set_xticks(worker_counts)
    ax2.set_title("Peak Memory Consumption vs Worker Count (w)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Workers (w)", fontsize=11)
    ax2.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9, ncol=2)

    path_mem = os.path.join(current_dir, "assignment4_memory.png")
    plt.tight_layout()
    plt.savefig(path_mem, dpi=300)
    print(f"[OK] Saved Memory Plot: {path_mem}")
    plt.show()

    # PLOT 3: Cross-Paradigm Comparison (Sequential vs Parallel Java vs Distributed Dask)

    # PLOT 3: Cross-Paradigm Comparison (Sequential vs Parallel Java vs Distributed Dask)

    a3_candidates = [
        os.path.join(current_dir, "..", "Assigment_3", "results_assignment_3.csv"),
        os.path.expanduser("~/IdeaProjects/MatrixMultiplication/results_assignment_3.csv"),
    ]
    a3_csv = find_file(a3_candidates)

    if a3_csv:
        summary_a3 = load_summary(a3_csv, worker_col="threads", method_prefix="Parallel")
        common_sizes = [n for n in sizes if n in summary_a3]

        if common_sizes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

            y_py_seq = [summary[n]["Sequential"]["median_time"] for n in common_sizes]
            y_py_d1 = [summary[n]["Distributed_w1"]["median_time"] for n in common_sizes]
            y_py_d8 = [summary[n]["Distributed_w8"]["median_time"] for n in common_sizes]

            y_java_seq = [summary_a3[n]["Sequential"]["median_time"] for n in common_sizes]
            y_java_p1 = [summary_a3[n]["Parallel_w1"]["median_time"] for n in common_sizes]
            y_java_p8 = [summary_a3[n]["Parallel_w8"]["median_time"] for n in common_sizes]

            series = [
                (y_py_seq, "Sequential (Python)", "#d62728", "o", "-"),
                (y_py_d1, "Distributed Dask (w=1)", "#ff7f0e", "s", "-"),
                (y_py_d8, "Distributed Dask (w=8)", "#1f77b4", "D", "-"),
                (y_java_seq, "Sequential (Java)", "#2ca02c", "^", "--"),
                (y_java_p1, "Parallel Threads (Java p=1)", "#8c564b", "P", "--"),
                (y_java_p8, "Parallel Threads (Java p=8)", "#9467bd", "v", "--"),
            ]

            for y_vals, label, color, marker, ls in series:
                ax1.plot(common_sizes, y_vals, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)
                ax2.plot(common_sizes, y_vals, marker=marker, color=color, linestyle=ls, linewidth=2, label=label)

            ax1.set_title("Cross-Paradigm Comparison: Sequential vs Parallel vs Distributed (Linear)", fontsize=11.5, fontweight="bold")
            ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
            ax1.set_ylabel("Median Execution Time (seconds)", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(fontsize=9.5)

            ax2.set_xscale("log", base=2)
            ax2.set_yscale("log")
            ax2.set_title("Cross-Paradigm Comparison: Sequential vs Parallel vs Distributed (Log-Log)", fontsize=11.5, fontweight="bold")
            ax2.set_xlabel("Matrix Dimension (n) [log2]", fontsize=11)
            ax2.set_ylabel("Median Execution Time (seconds) [log10]", fontsize=11)
            ax2.grid(True, which="both", linestyle="--", alpha=0.6)
            ax2.legend(fontsize=9.5)

            path_comp = os.path.join(current_dir, "assignment4_vs_parallel.png")
            plt.tight_layout()
            plt.savefig(path_comp, dpi=300)
            print(f"[OK] Saved Cross-Paradigm Comparison Plot: {path_comp}")
            plt.show()


if __name__ == "__main__":
    generate_plots()