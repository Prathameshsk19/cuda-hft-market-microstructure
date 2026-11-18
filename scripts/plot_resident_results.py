import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT = "benchmarks/analysis/resident_statistics.csv"
OUT = Path("benchmarks/analysis/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

workloads = ["100k", "500k", "1m", "2m", "5m"]
events = [100000, 500000, 1000000, 2000000, 5000000]

def get_metric(metric, column):
    return (
        df[df["metric"] == metric]
        .set_index("workload")
        .loc[workloads][column]
        .to_numpy(dtype=float)
    )

x = np.array(events)

# Figure 1: End-to-end latency
fig, ax = plt.subplots(figsize=(9, 6))

for metric, label in [
    ("CPU", "CPU"),
    ("Conventional GPU", "Conventional GPU"),
    ("Resident GPU E2E", "GPU-resident"),
]:
    means = get_metric(metric, "mean_ms")
    lower = get_metric(metric, "ci95_lower_ms")
    upper = get_metric(metric, "ci95_upper_ms")

    ax.errorbar(
        x, means,
        yerr=[means - lower, upper - means],
        marker="o", capsize=4, label=label
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of events")
ax.set_ylabel("Batch latency (ms)")
ax.set_title("End-to-End Batch Latency")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_latency_comparison.png", dpi=300)
plt.close(fig)

# Figure 2: Speedup
cpu = get_metric("CPU", "mean_ms")
conventional = get_metric("Conventional GPU", "mean_ms")
resident = get_metric("Resident GPU E2E", "mean_ms")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, cpu / resident, marker="o", label="Resident GPU vs CPU")
ax.plot(x, conventional / resident, marker="s",
        label="Conventional GPU vs Resident GPU")

ax.set_xscale("log")
ax.set_xlabel("Number of events")
ax.set_ylabel("Speedup (x)")
ax.set_title("GPU-Resident Performance Speedup")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_speedup.png", dpi=300)
plt.close(fig)

# Figure 3: Resident GPU timing breakdown
h2d = get_metric("Resident H2D", "mean_ms")
kernel = get_metric("Resident Kernel", "mean_ms")
d2h = get_metric("Resident D2H", "mean_ms")

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(workloads, h2d, label="Host-to-device transfer")
ax.bar(workloads, kernel, bottom=h2d, label="Kernel execution")
ax.bar(workloads, d2h, bottom=h2d + kernel,
       label="Device-to-host transfer")

ax.set_xlabel("Workload")
ax.set_ylabel("Latency (ms)")
ax.set_title("GPU-Resident Batch Latency Breakdown")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "03_latency_breakdown.png", dpi=300)
plt.close(fig)

# Figure 4: Throughput
fig, ax = plt.subplots(figsize=(9, 6))

for metric, label in [
    ("CPU", "CPU"),
    ("Conventional GPU", "Conventional GPU"),
    ("Resident GPU E2E", "GPU-resident"),
]:
    throughput = get_metric(
        metric, "throughput_million_events_s"
    )
    ax.plot(x, throughput, marker="o", label=label)

ax.set_xscale("log")
ax.set_xlabel("Number of events")
ax.set_ylabel("Throughput (million events/s)")
ax.set_title("Processing Throughput Comparison")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_throughput.png", dpi=300)
plt.close(fig)

print("Generated research figures:")
for path in sorted(OUT.glob("*.png")):
    print(path)
