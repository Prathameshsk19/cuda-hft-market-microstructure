import os
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
DATA_FILE = "benchmarks/initial_results.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load benchmark data
df = pd.read_csv(DATA_FILE)

required_columns = {
    "events",
    "cpu_ms",
    "gpu_kernel_ms",
    "gpu_e2e_ms"
}

if not required_columns.issubset(df.columns):
    raise ValueError("Benchmark CSV is missing required columns.")

if (df[["events", "cpu_ms", "gpu_kernel_ms", "gpu_e2e_ms"]] <= 0).any().any():
    raise ValueError("All benchmark values must be positive.")

# Calculate performance metrics
df["kernel_speedup"] = df["cpu_ms"] / df["gpu_kernel_ms"]

df["e2e_speedup"] = df["cpu_ms"] / df["gpu_e2e_ms"]

# Throughput in millions of events per second
df["cpu_throughput"] = df["events"] / (df["cpu_ms"] * 1000)

df["gpu_kernel_throughput"] = (
    df["events"] / (df["gpu_kernel_ms"] * 1000)
)

df["gpu_e2e_throughput"] = (
    df["events"] / (df["gpu_e2e_ms"] * 1000)
)

# GPU overhead beyond kernel execution
df["gpu_overhead_ms"] = df["gpu_e2e_ms"] - df["gpu_kernel_ms"]

# Save calculated metrics
df.to_csv("results/benchmark_analysis.csv", index=False)

# Convert event counts into millions for plotting
df["events_millions"] = df["events"] / 1_000_000


# --------------------------------------------------
# Graph 1: CPU vs GPU execution time
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df["events_millions"],
    df["cpu_ms"],
    marker="o",
    label="CPU"
)

plt.plot(
    df["events_millions"],
    df["gpu_kernel_ms"],
    marker="s",
    label="GPU Kernel"
)

plt.plot(
    df["events_millions"],
    df["gpu_e2e_ms"],
    marker="^",
    label="GPU End-to-End"
)

plt.xlabel("Number of Events (Millions)")
plt.ylabel("Execution Time (ms)")
plt.title("CPU vs GPU Execution Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/execution_time.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 2: GPU speedup
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df["events_millions"],
    df["kernel_speedup"],
    marker="o",
    label="Kernel-only Speedup"
)

plt.plot(
    df["events_millions"],
    df["e2e_speedup"],
    marker="s",
    label="End-to-End Speedup"
)

plt.axhline(y=1, linestyle="--", label="Break-even (1x)")

plt.xlabel("Number of Events (Millions)")
plt.ylabel("Speedup (x)")
plt.title("GPU Speedup vs Workload Size")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/speedup.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 3: Processing throughput
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df["events_millions"],
    df["cpu_throughput"],
    marker="o",
    label="CPU"
)

plt.plot(
    df["events_millions"],
    df["gpu_kernel_throughput"],
    marker="s",
    label="GPU Kernel"
)

plt.plot(
    df["events_millions"],
    df["gpu_e2e_throughput"],
    marker="^",
    label="GPU End-to-End"
)

plt.xlabel("Number of Events (Millions)")
plt.ylabel("Throughput (Million Events/Second)")
plt.title("Processing Throughput Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/throughput.png", dpi=300)
plt.close()


# --------------------------------------------------
# Graph 4: GPU overhead
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    df["events_millions"].astype(str),
    df["gpu_overhead_ms"]
)

plt.xlabel("Number of Events (Millions)")
plt.ylabel("GPU Overhead (ms)")
plt.title("GPU End-to-End Overhead Beyond Kernel Execution")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gpu_overhead.png", dpi=300)
plt.close()


# --------------------------------------------------
# Print benchmark summary
# --------------------------------------------------

print("\n========== BENCHMARK ANALYSIS ==========\n")

print(
    df[
        [
            "events",
            "kernel_speedup",
            "e2e_speedup",
            "cpu_throughput",
            "gpu_kernel_throughput",
            "gpu_e2e_throughput"
        ]
    ].round(2).to_string(index=False)
)

print("\nGenerated graphs:")

print(f"{OUTPUT_DIR}/execution_time.png")
print(f"{OUTPUT_DIR}/speedup.png")
print(f"{OUTPUT_DIR}/throughput.png")
print(f"{OUTPUT_DIR}/gpu_overhead.png")

print(f"\nCalculated data saved to results/benchmark_analysis.csv")

print("\nBenchmark analysis completed successfully!")
