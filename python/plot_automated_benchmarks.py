import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

INPUT_FILE = "benchmarks/automated_results.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("Benchmark CSV is empty.")

if not (df["verification"] == "PASSED").all():
    raise ValueError("Some benchmark workloads failed verification.")

events = df["events"] / 1_000_000

labels = [
    f"{int(n / 1_000_000)}M" if n >= 1_000_000
    else f"{int(n / 1000)}K"
    for n in df["events"]
]

# --------------------------------------------------
# 1. Execution time comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(labels, df["cpu_mean_ms"], marker="o",
         label="CPU")

plt.plot(labels, df["kernel_mean_ms"], marker="o",
         label="GPU Kernel")

plt.plot(labels, df["gpu_e2e_mean_ms"], marker="o",
         label="GPU End-to-End")

plt.xlabel("Number of Market Events")
plt.ylabel("Mean Execution Time (ms)")
plt.title("CPU vs GPU Execution Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/automated_execution_time.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 2. Speedup comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(labels, df["kernel_speedup"], marker="o",
         label="Kernel-only Speedup")

plt.plot(labels, df["e2e_speedup"], marker="o",
         label="End-to-End Speedup")

plt.axhline(y=1.0, linestyle="--",
            label="Break-even (1x)")

plt.xlabel("Number of Market Events")
plt.ylabel("Speedup (x)")
plt.title("GPU Speedup Relative to CPU")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/automated_speedup.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 3. Throughput comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(labels, df["cpu_throughput_mops"],
         marker="o", label="CPU")

plt.plot(labels, df["gpu_kernel_throughput_mops"],
         marker="o", label="GPU Kernel")

plt.plot(labels, df["gpu_e2e_throughput_mops"],
         marker="o", label="GPU End-to-End")

plt.xlabel("Number of Market Events")
plt.ylabel("Throughput (Million Events/s)")
plt.title("CPU vs GPU Throughput")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/automated_throughput.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 4. GPU timing breakdown
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(labels, df["h2d_mean_ms"],
        label="Host-to-Device")

plt.bar(labels, df["kernel_mean_ms"],
        bottom=df["h2d_mean_ms"],
        label="GPU Kernel")

bottom_d2h = (
    df["h2d_mean_ms"] +
    df["kernel_mean_ms"]
)

plt.bar(labels, df["d2h_mean_ms"],
        bottom=bottom_d2h,
        label="Device-to-Host")

plt.xlabel("Number of Market Events")
plt.ylabel("Mean Time (ms)")
plt.title("GPU Pipeline Timing Breakdown")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/automated_gpu_breakdown.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. Transfer overhead percentage
# --------------------------------------------------

df["transfer_overhead_ms"] = (
    df["h2d_mean_ms"] +
    df["d2h_mean_ms"]
)

df["transfer_percentage"] = (
    df["transfer_overhead_ms"] /
    df["gpu_e2e_mean_ms"]
) * 100

plt.figure(figsize=(10, 6))

plt.plot(labels, df["transfer_percentage"],
         marker="o")

plt.xlabel("Number of Market Events")
plt.ylabel("Transfer Time (% of GPU E2E)")
plt.title("GPU Data Transfer Overhead")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/automated_transfer_overhead.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 6. Benchmark summary CSV
# --------------------------------------------------

summary_columns = [
    "events",
    "cpu_mean_ms",
    "cpu_median_ms",
    "kernel_mean_ms",
    "gpu_e2e_mean_ms",
    "h2d_mean_ms",
    "d2h_mean_ms",
    "kernel_speedup",
    "e2e_speedup",
    "cpu_throughput_mops",
    "gpu_kernel_throughput_mops",
    "gpu_e2e_throughput_mops",
    "transfer_percentage",
    "verification"
]

df[summary_columns].to_csv(
    f"{OUTPUT_DIR}/automated_benchmark_summary.csv",
    index=False
)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\n========================================")
print("AUTOMATED CUDA BENCHMARK ANALYSIS")
print("========================================")

print(
    df[
        [
            "events",
            "kernel_speedup",
            "e2e_speedup",
            "transfer_percentage",
            "verification"
        ]
    ].to_string(index=False)
)

print("\nGenerated graphs:")

print("1. automated_execution_time.png")
print("2. automated_speedup.png")
print("3. automated_throughput.png")
print("4. automated_gpu_breakdown.png")
print("5. automated_transfer_overhead.png")

print("\nSummary:")
print("results/automated_benchmark_summary.csv")

print("\nAll benchmark results passed verification.")
