import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "benchmarks/integrated_pipeline_sessions"
OUT = "results/integrated_pipeline_analysis"

SESSIONS = ["session_1", "session_2", "session_3"]

CONFIGS = [
    "pageable_sync_1_stream",
    "pinned_sync_1_stream",
    "pinned_async_1_stream",
    "pinned_async_2_streams",
]

CONFIG_LABELS = {
    "pageable_sync_1_stream": "Pageable sync",
    "pinned_sync_1_stream": "Pinned sync",
    "pinned_async_1_stream": "Pinned async (1 stream)",
    "pinned_async_2_streams": "Pinned async (2 streams)",
}

CHUNKS = [50000, 100000, 250000, 500000, 1000000, 2500000]
T_CRITICAL_DF2 = 4.3026527299

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------
# 1. Load and validate raw measurements
# --------------------------------------------------

all_data = []

for session in SESSIONS:
    path = os.path.join(BASE, session, "integrated_pipeline_raw.csv")
    df = pd.read_csv(path)
    df["session"] = session

    required = {
        "configuration", "chunk_size", "iteration",
        "execution_order", "elapsed_ms"
    }

    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected CSV columns in {path}")

    if len(df) != 480:
        raise ValueError(f"{path}: expected 480 rows, found {len(df)}")

    if df["elapsed_ms"].isna().any() or (df["elapsed_ms"] <= 0).any():
        raise ValueError(f"Invalid timing values in {path}")

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

expected_groups = len(SESSIONS) * len(CONFIGS) * len(CHUNKS)
group_counts = data.groupby(
    ["session", "configuration", "chunk_size"]
).size()

if len(group_counts) != expected_groups or not (group_counts == 20).all():
    raise ValueError("Expected exactly 20 observations per session/configuration/chunk.")

data.to_csv(os.path.join(OUT, "combined_raw_measurements.csv"), index=False)

# --------------------------------------------------
# 2. Calculate each session's mean latency
# --------------------------------------------------

session_means = (
    data.groupby(["session", "configuration", "chunk_size"])["elapsed_ms"]
    .mean()
    .reset_index(name="session_mean_ms")
)

session_means.to_csv(
    os.path.join(OUT, "session_level_means.csv"), index=False
)

# --------------------------------------------------
# 3. Between-session statistics
# --------------------------------------------------

rows = []

for (config, chunk), group in session_means.groupby(
    ["configuration", "chunk_size"]
):
    values = group["session_mean_ms"].to_numpy()

    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    margin = T_CRITICAL_DF2 * sd / np.sqrt(len(values))

    rows.append({
        "configuration": config,
        "chunk_size": chunk,
        "sessions": len(values),
        "mean_latency_ms": mean,
        "between_session_sd_ms": sd,
        "ci95_lower_ms": mean - margin,
        "ci95_upper_ms": mean + margin,
        "ci95_margin_ms": margin,
    })

stats = pd.DataFrame(rows)

# --------------------------------------------------
# 4. Paired speedup against pageable baseline
# --------------------------------------------------

wide = session_means.pivot(
    index=["session", "chunk_size"],
    columns="configuration",
    values="session_mean_ms"
).reset_index()

speed_rows = []

for chunk in CHUNKS:
    subset = wide[wide["chunk_size"] == chunk]

    baseline = subset["pageable_sync_1_stream"].to_numpy()

    for config in CONFIGS:
        measured = subset[config].to_numpy()

        # Each ratio compares configurations within the same session.
        ratios = baseline / measured
        reductions = (baseline - measured) / baseline * 100

        ratio_mean = np.mean(ratios)
        ratio_sd = np.std(ratios, ddof=1)
        ratio_margin = T_CRITICAL_DF2 * ratio_sd / np.sqrt(len(ratios))

        speed_rows.append({
            "configuration": config,
            "chunk_size": chunk,
            "mean_paired_speedup": ratio_mean,
            "speedup_between_session_sd": ratio_sd,
            "speedup_ci95_lower": ratio_mean - ratio_margin,
            "speedup_ci95_upper": ratio_mean + ratio_margin,
            "mean_latency_reduction_percent": np.mean(reductions),
        })

speed = pd.DataFrame(speed_rows)

# --------------------------------------------------
# 4B. Paired latency-difference analysis
# Positive difference means the two-stream configuration is faster.
# --------------------------------------------------

comparisons = [
    ("pageable_sync_1_stream", "pinned_async_2_streams"),
    ("pinned_sync_1_stream", "pinned_async_2_streams"),
    ("pinned_async_1_stream", "pinned_async_2_streams"),
]

paired_rows = []

for reference, candidate in comparisons:
    for chunk in CHUNKS:
        subset = wide[wide["chunk_size"] == chunk]

        reference_values = subset[reference].to_numpy()
        candidate_values = subset[candidate].to_numpy()

        differences = reference_values - candidate_values
        mean_difference = np.mean(differences)
        sd_difference = np.std(differences, ddof=1)
        margin = T_CRITICAL_DF2 * sd_difference / np.sqrt(len(differences))

        percentage_reductions = (
            differences / reference_values
        ) * 100

        paired_rows.append({
            "reference_configuration": reference,
            "candidate_configuration": candidate,
            "chunk_size": chunk,
            "sessions": len(differences),
            "mean_paired_latency_reduction_ms": mean_difference,
            "between_session_sd_ms": sd_difference,
            "ci95_lower_ms": mean_difference - margin,
            "ci95_upper_ms": mean_difference + margin,
            "mean_paired_latency_reduction_percent": np.mean(
                percentage_reductions
            ),
        })

paired_stats = pd.DataFrame(paired_rows)

paired_stats.to_csv(
    os.path.join(OUT, "integrated_pipeline_paired_differences.csv"),
    index=False
)

stats = stats.merge(
    speed,
    on=["configuration", "chunk_size"],
    how="left"
)

stats["configuration_label"] = stats["configuration"].map(CONFIG_LABELS)

stats.to_csv(
    os.path.join(OUT, "integrated_pipeline_combined_statistics.csv"),
    index=False
)

# Also save a readable table of all three session means.
session_table = session_means.pivot_table(
    index=["configuration", "chunk_size"],
    columns="session",
    values="session_mean_ms"
).reset_index()

session_table.to_csv(
    os.path.join(OUT, "integrated_pipeline_session_comparison.csv"),
    index=False
)

# --------------------------------------------------
# 5. Figure 1: Mean latency with between-session CI
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(CHUNKS))
width = 0.19

for i, config in enumerate(CONFIGS):
    subset = stats[stats["configuration"] == config].sort_values("chunk_size")

    positions = x + (i - 1.5) * width
    means = subset["mean_latency_ms"].to_numpy()
    lower = means - subset["ci95_lower_ms"].to_numpy()
    upper = subset["ci95_upper_ms"].to_numpy() - means

    ax.bar(
        positions, means, width,
        label=CONFIG_LABELS[config],
        yerr=np.vstack([lower, upper]),
        capsize=3
    )

ax.set_xticks(x)
ax.set_xticklabels(["50K", "100K", "250K", "500K", "1M", "2.5M"])
ax.set_xlabel("Chunk size (events)")
ax.set_ylabel("Mean execution latency (ms)")
ax.set_title("Integrated Pipeline Latency Across Three Sessions")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_mean_latency_ci.png"), dpi=300)
plt.close(fig)

# --------------------------------------------------
# 6. Figure 2: Paired speedup
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

x_positions = np.arange(len(CHUNKS))
chunk_labels = ["50K", "100K", "250K", "500K", "1M", "2.5M"]

for config in CONFIGS:
    subset = speed[speed["configuration"] == config].sort_values("chunk_size")
    ax.plot(
        x_positions,
        subset["mean_paired_speedup"],
        marker="o",
        linewidth=2,
        label=CONFIG_LABELS[config]
    )

ax.axhline(1.0, linestyle="--", linewidth=1)
ax.set_xticks(x_positions)
ax.set_xticklabels(chunk_labels)
ax.set_xlabel("Chunk size (events)")
ax.set_ylabel("Paired speedup vs pageable synchronous")
ax.set_title("Paired Performance Speedup Across Three Sessions")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_paired_speedup.png"), dpi=300)
plt.close(fig)

# --------------------------------------------------
# 7. Figure 3: Latency reduction
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

for config in CONFIGS[1:]:
    subset = speed[speed["configuration"] == config].sort_values("chunk_size")
    ax.plot(
        x_positions,
        subset["mean_latency_reduction_percent"],
        marker="o",
        linewidth=2,
        label=CONFIG_LABELS[config]
    )

ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xticks(x_positions)
ax.set_xticklabels(chunk_labels)
ax.set_xlabel("Chunk size (events)")
ax.set_ylabel("Latency reduction vs pageable baseline (%)")
ax.set_title("Latency Reduction from Memory and Stream Optimisations")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_latency_reduction.png"), dpi=300)
plt.close(fig)

# --------------------------------------------------
# 8. Figure 4: Session-to-session reproducibility
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

positions = np.arange(len(CHUNKS))
offsets = [-0.22, 0.0, 0.22]

for i, session in enumerate(SESSIONS):
    subset = session_means[
        (session_means["session"] == session) &
        (session_means["configuration"] == "pinned_async_2_streams")
    ].sort_values("chunk_size")

    ax.plot(
        positions + offsets[i],
        subset["session_mean_ms"],
        marker="o",
        linewidth=1.8,
        label=session.replace("_", " ").title()
    )

ax.set_xticks(positions)
ax.set_xticklabels(["50K", "100K", "250K", "500K", "1M", "2.5M"])
ax.set_xlabel("Chunk size (events)")
ax.set_ylabel("Mean execution latency (ms)")
ax.set_title("Session-to-Session Reproducibility: Two-Stream Configuration")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_session_reproducibility.png"), dpi=300)
plt.close(fig)

# --------------------------------------------------
# 9. Print best configuration per chunk
# --------------------------------------------------

print("\nIntegrated pipeline analysis completed.")
print(f"Raw observations: {len(data)}")
print(f"Session-level means: {len(session_means)}")
print(f"Combined configurations: {len(stats)}")
print(f"Output directory: {OUT}")

print("\nBest configuration by mean latency:")
best = stats.loc[
    stats.groupby("chunk_size")["mean_latency_ms"].idxmin(),
    ["chunk_size", "configuration", "mean_latency_ms",
     "between_session_sd_ms", "ci95_lower_ms", "ci95_upper_ms"]
].sort_values("chunk_size")

print(best.to_string(index=False))

print("\nAnalysis files and figures:")
for filename in sorted(os.listdir(OUT)):
    print(" -", filename)
