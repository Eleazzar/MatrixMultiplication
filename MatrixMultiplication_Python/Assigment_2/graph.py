import csv
import os
import statistics
import matplotlib.pyplot as plt


def load_part_a_summary(csv_path):
    """
    Loads Part A CSV and returns a structured dictionary:
    summary[method][n] = {'median_time': ..., 'iqr_time': ..., 'median_mem': ...}
    """
    raw = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            method = row["method"]
            t = float(row["time_seconds"])
            m = float(row["peak_memory_mb"])

            raw.setdefault(method, {}).setdefault(n, {"times": [], "mems": []})
            raw[method][n]["times"].append(t)
            raw[method][n]["mems"].append(m)

    summary = {}
    for method, by_n in raw.items():
        summary[method] = {}
        for n, vals in sorted(by_n.items()):
            ts = sorted(vals["times"])
            ms = sorted(vals["mems"])
            iqr_t = ts[3] - ts[1] if len(ts) >= 5 else (ts[-1] - ts[0])
            summary[method][n] = {
                "median_time": statistics.median(ts),
                "iqr_time": iqr_t,
                "median_mem": statistics.median(ms),
            }
    return summary


def load_part_b_summary(csv_path):
    """
    Loads Part B CSV and returns:
    summary[method][sparsity_pct] = {
        'density_pct', 'median_kernel', 'iqr_kernel', 'median_conv',
        'median_total', 'median_mem', 'nnz_A', 'nnz_B', 'nnz_C', 'n'
    }
    """
    raw = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            method = row["method"]
            sp = float(row["sparsity_percent"])
            dens = float(row["density_percent"])
            nnz_a = int(row["nnz_A"])
            nnz_b = int(row["nnz_B"])
            nnz_c = int(row["nnz_C"])
            conv_t = float(row["conversion_time_seconds"])
            kern_t = float(row["kernel_time_seconds"])
            tot_t = float(row["total_time_seconds"])
            mem = float(row["peak_memory_mb"])

            entry = raw.setdefault(method, {}).setdefault(
                sp,
                {
                    "n": n,
                    "density_pct": dens,
                    "nnz_A": [],
                    "nnz_B": [],
                    "nnz_C": [],
                    "conv_times": [],
                    "kern_times": [],
                    "tot_times": [],
                    "mems": [],
                },
            )
            entry["nnz_A"].append(nnz_a)
            entry["nnz_B"].append(nnz_b)
            entry["nnz_C"].append(nnz_c)
            entry["conv_times"].append(conv_t)
            entry["kern_times"].append(kern_t)
            entry["tot_times"].append(tot_t)
            entry["mems"].append(mem)

    summary = {}
    for method, by_sp in raw.items():
        summary[method] = {}
        for sp, vals in sorted(by_sp.items()):
            ks = sorted(vals["kern_times"])
            iqr_k = ks[3] - ks[1] if len(ks) >= 5 else (ks[-1] - ks[0])
            summary[method][sp] = {
                "n": vals["n"],
                "density_pct": vals["density_pct"],
                "nnz_A": int(statistics.median(vals["nnz_A"])),
                "nnz_B": int(statistics.median(vals["nnz_B"])),
                "nnz_C": int(statistics.median(vals["nnz_C"])),
                "median_conv": statistics.median(vals["conv_times"]),
                "median_kernel": statistics.median(ks),
                "iqr_kernel": iqr_k,
                "median_total": statistics.median(vals["tot_times"]),
                "median_mem": statistics.median(vals["mems"]),
            }
    return summary


def generate_plots(show_Interactive=False):
    """Generates console summary tables and high-resolution figures for Part A and Part B."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_a = os.path.join(current_dir, "results_part_a_dense.csv")
    csv_b = os.path.join(current_dir, "results_part_b_sparse.csv")

    if not os.path.exists(csv_a) or not os.path.exists(csv_b):
        print("Error: Missing benchmark CSV files in Assigment_2 directory.")
        return

    # PART A: CONSOLE SUMMARY TABLE & PLOTS
    sum_a = load_part_a_summary(csv_a)
    sizes = sorted(next(iter(sum_a.values())).keys())

    print("\n" + "=" * 112)
    print("                     PART A SUMMARY: DENSE OPTIMISATION APPROACHES (MEDIANS OVER 5 RUNS)")
    print("=" * 112)
    print(
        f"{'n':<6} | {'Baseline ijk (s)':<17} | {'Loop ikj (s)':<15} | {'Transpose B (s)':<16} | "
        f"{'Speedup ikj':<12} | {'Speedup B^T':<12} | {'Mem ijk/B^T (MB)':<18}"
    )
    print("-" * 112)
    for n in sizes:
        t_base = sum_a["Baseline_ijk"][n]["median_time"]
        t_ikj = sum_a["Loop_Reordering_ikj"][n]["median_time"]
        t_tr = sum_a["Transpose_B"][n]["median_time"]
        m_base = sum_a["Baseline_ijk"][n]["median_mem"]
        m_tr = sum_a["Transpose_B"][n]["median_mem"]
        sp_ikj = t_base / t_ikj if t_ikj > 0 else 0.0
        sp_tr = t_base / t_tr if t_tr > 0 else 0.0
        print(
            f"{n:<6} | {t_base:<17.5f} | {t_ikj:<15.5f} | {t_tr:<16.5f} | "
            f"{sp_ikj:<11.2f}x | {sp_tr:<11.2f}x | {m_base:.3f} / {m_tr:.3f}"
        )
    print("=" * 112)

    # Plot A1: Execution Time (Linear vs Log-Log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    styles_a = [
        ("Baseline_ijk", "Baseline (i-j-k)", "#d62728", "o"),
        ("Loop_Reordering_ikj", "Loop Reordering (i-k-j)", "#1f77b4", "s"),
        ("Transpose_B", "Transpose B (B^T)", "#2ca02c", "*"),
    ]

    for key, label, color, marker in styles_a:
        y_time = [sum_a[key][n]["median_time"] for n in sizes]
        ax1.plot(sizes, y_time, marker=marker, color=color, linewidth=2, label=label)
        ax2.plot(sizes, y_time, marker=marker, color=color, linewidth=2, label=label)

    ax1.set_title("Part A: Dense Execution Time (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Execution Time (seconds)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Part A: Dense Execution Time (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n) [log2]", fontsize=11)
    ax2.set_ylabel("Median Execution Time (seconds) [log10]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    path_a_time = os.path.join(current_dir, "part_a_dense_time.png")
    plt.tight_layout()
    plt.savefig(path_a_time, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Part A Time Plot: {path_a_time}")

    # Plot A2: Peak Memory & Relative Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, label, color, marker in styles_a:
        y_mem = [sum_a[key][n]["median_mem"] for n in sizes]
        ax1.plot(sizes, y_mem, marker=marker, color=color, linewidth=2, label=label)

    ax1.set_title("Part A: Peak Memory Consumption (tracemalloc)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax1.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    sp_ikj_list = [sum_a["Baseline_ijk"][n]["median_time"] / sum_a["Loop_Reordering_ikj"][n]["median_time"] for n in sizes]
    sp_tr_list = [sum_a["Baseline_ijk"][n]["median_time"] / sum_a["Transpose_B"][n]["median_time"] for n in sizes]
    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, label="Baseline (1.0x)")
    ax2.plot(sizes, sp_ikj_list, marker="s", color="#1f77b4", linewidth=2, label="Loop Reordering (i-k-j)")
    ax2.plot(sizes, sp_tr_list, marker="*", color="#2ca02c", linewidth=2, label="Transpose B (B^T)")
    ax2.set_title("Part A: Speedup over Baseline (i-j-k)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Dimension (n)", fontsize=11)
    ax2.set_ylabel("Speedup Factor (T_baseline / T_opt)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    path_a_mem = os.path.join(current_dir, "part_a_dense_memory_speedup.png")
    plt.tight_layout()
    plt.savefig(path_a_mem, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Part A Memory & Speedup Plot: {path_a_mem}")

    # PART B: CONSOLE SUMMARY TABLE & PLOTS

    sum_b = load_part_b_summary(csv_b)
    sparsities = sorted(sum_b["Sparse_CSR"].keys())
    n_fixed = sum_b["Sparse_CSR"][sparsities[0]]["n"]
    total_cells = n_fixed * n_fixed

    print("\n" + "=" * 124)
    print(f"               PART B SUMMARY: SPARSE CSR vs DENSE ACROSS SPARSITY LEVELS (n = {n_fixed}, MEDIANS)")
    print("=" * 124)
    print(
        f"{'Sparsity (%)':<13} | {'Density (%)':<11} | {'Dense ijk (s)':<14} | {'Dense ikj (s)':<14} | "
        f"{'CSR Kernel (s)':<14} | {'CSR Total (s)':<13} | {'CSR Mem (MB)':<12} | {'nnz(A) -> nnz(C) [Out %]':<24}"
    )
    print("-" * 124)
    for sp in sparsities:
        dens = sum_b["Sparse_CSR"][sp]["density_pct"]
        t_d_ijk = sum_b["Dense_Baseline_ijk"][sp]["median_kernel"]
        t_d_ikj = sum_b["Dense_Optimised_ikj"][sp]["median_kernel"]
        t_csr_k = sum_b["Sparse_CSR"][sp]["median_kernel"]
        t_csr_tot = sum_b["Sparse_CSR"][sp]["median_total"]
        m_csr = sum_b["Sparse_CSR"][sp]["median_mem"]
        nnz_a = sum_b["Sparse_CSR"][sp]["nnz_A"]
        nnz_c = sum_b["Sparse_CSR"][sp]["nnz_C"]
        out_pct = 100.0 * nnz_c / total_cells
        print(
            f"{sp:<13.1f} | {dens:<11.1f} | {t_d_ijk:<14.5f} | {t_d_ikj:<14.5f} | "
            f"{t_csr_k:<14.5f} | {t_csr_tot:<13.5f} | {m_csr:<12.3f} | "
            f"{nnz_a:<6} -> {nnz_c:<6} ({out_pct:5.1f}%)"
        )
    print("=" * 124 + "\n")

    # Plot B1: Execution Time vs Sparsity (Linear & Log Scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    y_d_ijk = [sum_b["Dense_Baseline_ijk"][sp]["median_kernel"] for sp in sparsities]
    y_d_ikj = [sum_b["Dense_Optimised_ikj"][sp]["median_kernel"] for sp in sparsities]
    y_csr_k = [sum_b["Sparse_CSR"][sp]["median_kernel"] for sp in sparsities]
    y_csr_t = [sum_b["Sparse_CSR"][sp]["median_total"] for sp in sparsities]

    for ax in (ax1, ax2):
        ax.plot(sparsities, y_d_ijk, marker="o", color="#d62728", linewidth=2, label="Dense Baseline (i-j-k)")
        ax.plot(sparsities, y_d_ikj, marker="s", color="#1f77b4", linewidth=2, label="Dense Optimised (i-k-j)")
        ax.plot(sparsities, y_csr_k, marker="*", color="#2ca02c", linewidth=2, label="Sparse CSR (Kernel only)")
        ax.plot(sparsities, y_csr_t, marker="v", color="#ff7f0e", linestyle="--", linewidth=2, label="Sparse CSR (Kernel + Conversion)")
        ax.set_xlabel("Matrix Sparsity (% of zero entries)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(fontsize=9.5)

    ax1.set_title(f"Part B: Execution Time vs Sparsity (n={n_fixed}, Linear)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Median Execution Time (seconds)", fontsize=11)

    ax2.set_yscale("log")
    ax2.set_title(f"Part B: Execution Time vs Sparsity (n={n_fixed}, Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Median Execution Time (seconds) [log10]", fontsize=11)

    path_b_time = os.path.join(current_dir, "part_b_sparse_time.png")
    plt.tight_layout()
    plt.savefig(path_b_time, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Part B Time Plot: {path_b_time}")

    # Plot B2: Peak Memory & Fill-in (Input vs Output Nonzero Density)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    m_d_ijk = [sum_b["Dense_Baseline_ijk"][sp]["median_mem"] for sp in sparsities]
    m_csr = [sum_b["Sparse_CSR"][sp]["median_mem"] for sp in sparsities]

    ax1.plot(sparsities, m_d_ijk, marker="o", color="#d62728", linewidth=2, label="Dense 2D Matrix (C allocation)")
    ax1.plot(sparsities, m_csr, marker="*", color="#2ca02c", linewidth=2, label="Sparse CSR Output + Row Buffer")
    ax1.set_title(f"Part B: Peak Kernel Memory vs Sparsity (n={n_fixed})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Matrix Sparsity (% of zero entries)", fontsize=11)
    ax1.set_ylabel("Median Peak Memory (MB)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    in_dens = [100.0 * sum_b["Sparse_CSR"][sp]["nnz_A"] / total_cells for sp in sparsities]
    out_dens = [100.0 * sum_b["Sparse_CSR"][sp]["nnz_C"] / total_cells for sp in sparsities]

    ax2.plot(sparsities, in_dens, marker="s", color="#1f77b4", linewidth=2, label="Input Matrix A/B Density (% nonzeros)")
    ax2.plot(sparsities, out_dens, marker="D", color="#9467bd", linewidth=2, label="Output Matrix C Density (% nonzeros)")
    ax2.set_title(f"Part B: Input vs Output Density (n={n_fixed})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Input Matrix Sparsity (% of zero entries)", fontsize=11)
    ax2.set_ylabel("Nonzero Density (% of n² entries)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    path_b_mem = os.path.join(current_dir, "part_b_sparse_memory.png")
    plt.tight_layout()
    plt.savefig(path_b_mem, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Part B Memory Plot: {path_b_mem}")

