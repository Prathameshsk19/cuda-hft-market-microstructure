import csv
import math
import os
import statistics

WORKLOADS = {
    "100k": 100000,
    "500k": 500000,
    "1m": 1000000,
    "2m": 2000000,
    "5m": 5000000,
}

PASSES = 20
T_CRITICAL_95_DF19 = 2.093

COLUMNS = {
    "CPU": "cpu_batch_ms",
    "Conventional GPU": "conventional_gpu_batch_ms",
    "Resident H2D": "resident_h2d_ms",
    "Resident Kernel": "resident_kernel_ms",
    "Resident D2H": "resident_d2h_ms",
    "Resident GPU E2E": "resident_e2e_ms",
}

os.makedirs("benchmarks/analysis", exist_ok=True)

results = []

for label, events in WORKLOADS.items():
    filename = f"benchmarks/gpu_resident_{label}_raw.csv"

    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 20:
        raise ValueError(
            f"{filename}: expected 20 measurements, got {len(rows)}"
        )

    iterations = [int(row["iteration"]) for row in rows]

    if len(set(iterations)) != 20:
        raise ValueError(f"{filename}: duplicate iteration numbers")

    data = {
        name: [float(row[column]) for row in rows]
        for name, column in COLUMNS.items()
    }

    cpu_mean = statistics.mean(data["CPU"])
    conventional_mean = statistics.mean(data["Conventional GPU"])
    resident_mean = statistics.mean(data["Resident GPU E2E"])

    for name, values in data.items():
        mean = statistics.mean(values)
        median = statistics.median(values)
        sd = statistics.stdev(values)
        margin = T_CRITICAL_95_DF19 * sd / math.sqrt(len(values))

        # Throughput is meaningful for complete processing batches.
        if name in ("CPU", "Conventional GPU", "Resident GPU E2E"):
            throughput = (events * PASSES) / (mean * 1000)
        else:
            throughput = ""

        results.append({
            "workload": label,
            "events": events,
            "metric": name,
            "mean_ms": mean,
            "median_ms": median,
            "std_dev_ms": sd,
            "min_ms": min(values),
            "max_ms": max(values),
            "ci95_lower_ms": mean - margin,
            "ci95_upper_ms": mean + margin,
            "throughput_million_events_s": throughput,
        })

    print(f"\n===== {label}: {events:,} events =====")
    print(f"CPU mean:                 {cpu_mean:.4f} ms")
    print(f"Conventional GPU mean:    {conventional_mean:.4f} ms")
    print(f"Resident GPU mean:        {resident_mean:.4f} ms")
    print(f"Resident vs CPU:          {cpu_mean/resident_mean:.4f}x")
    print(f"Resident vs conventional:{conventional_mean/resident_mean: .4f}x")

output = "benchmarks/analysis/resident_statistics.csv"

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========================================")
print("STATISTICAL ANALYSIS COMPLETE")
print("========================================")
print(f"Measurements analysed: {len(WORKLOADS) * 20}")
print(f"Output: {output}")
