#!/usr/bin/env python3

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("profiling/nsys/integrated_pipeline_audit.sqlite")
OUTPUT_PATH = Path(
    "results/integrated_pipeline_analysis/nsys_overlap_analysis.csv"
)

def load_events(db, table, stream, columns):
    selected = ", ".join(columns)
    query = f"""
        SELECT {selected}
        FROM {table}
        WHERE streamId = ?
        ORDER BY start
    """
    return db.execute(query, (stream,)).fetchall()


def check_intervals(events):
    """Check whether intervals overlap within one stream."""
    overlaps = 0
    previous_end = None

    for event in events:
        start, end = event[0], event[1]

        if previous_end is not None and start < previous_end:
            overlaps += 1

        previous_end = max(previous_end or end, end)

    return overlaps


def calculate_overlap(kernels, copies, copy_kind):
    """Calculate intersections between two sorted, non-overlapping streams."""
    i = 0
    j = 0
    overlap_ns = 0
    overlap_pairs = 0

    while i < len(kernels) and j < len(copies):
        kernel_start, kernel_end = kernels[i]
        copy_start, copy_end, _ = copies[j]

        intersection = max(
            0,
            min(kernel_end, copy_end) - max(kernel_start, copy_start)
        )

        if intersection > 0:
            overlap_ns += intersection
            overlap_pairs += 1

        if kernel_end <= copy_end:
            i += 1
        else:
            j += 1

    kernel_time_ns = sum(end - start for start, end in kernels)
    percentage = (
        100.0 * overlap_ns / kernel_time_ns
        if kernel_time_ns else 0.0
    )

    return {
        "kernel_stream": None,
        "copy_stream": None,
        "transfer": copy_kind,
        "kernel_events": len(kernels),
        "copy_events": len(copies),
        "kernel_execution_ms": kernel_time_ns / 1e6,
        "overlapping_event_pairs": overlap_pairs,
        "overlap_ms": overlap_ns / 1e6,
        "overlap_percentage_of_kernel_time": percentage,
    }


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Nsight database not found: {DB_PATH}")

    db = sqlite3.connect(DB_PATH)

    try:
        kernel_streams = [
            row[0] for row in db.execute(
                "SELECT DISTINCT streamId "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY streamId"
            )
        ]

        copy_streams = [
            row[0] for row in db.execute(
                "SELECT DISTINCT streamId "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY streamId"
            )
        ]

        print("Nsight GPU overlap analysis")
        print(f"Database: {DB_PATH}")
        print(f"Kernel streams: {kernel_streams}")
        print(f"Copy streams: {copy_streams}")

        # Validate the non-overlap assumption within each stream.
        print("\nWithin-stream interval validation:")

        validation_passed = True

        for stream in kernel_streams:
            events = load_events(
                db, "CUPTI_ACTIVITY_KIND_KERNEL", stream, ["start", "end"]
            )
            count = check_intervals(events)
            print(f"Kernel stream {stream}: {count} overlapping intervals")
            validation_passed &= count == 0

        for stream in copy_streams:
            events = load_events(
                db, "CUPTI_ACTIVITY_KIND_MEMCPY", stream,
                ["start", "end", "copyKind"]
            )
            count = check_intervals(events)
            print(f"Copy stream {stream}: {count} overlapping intervals")
            validation_passed &= count == 0

        if not validation_passed:
            raise RuntimeError(
                "Overlapping intervals detected within a stream. "
                "The simple interval summation needs further handling."
            )

        # Compare each kernel stream against transfers in the other stream.
        comparisons = []

        for kernel_stream, copy_stream in [(13, 14), (14, 13)]:
            kernels = load_events(
                db, "CUPTI_ACTIVITY_KIND_KERNEL",
                kernel_stream, ["start", "end"]
            )

            for kind, label in [(1, "H2D"), (2, "D2H")]:
                copies = db.execute("""
                    SELECT start, end, copyKind
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE streamId = ? AND copyKind = ?
                    ORDER BY start
                """, (copy_stream, kind)).fetchall()

                result = calculate_overlap(kernels, copies, label)
                result["kernel_stream"] = kernel_stream
                result["copy_stream"] = copy_stream
                comparisons.append(result)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(comparisons[0].keys())
            )
            writer.writeheader()
            writer.writerows(comparisons)

        print("\nCross-stream overlap results:")

        for result in comparisons:
            print(
                f"Kernel stream {result['kernel_stream']} vs "
                f"copy stream {result['copy_stream']} | "
                f"{result['transfer']}: "
                f"{result['overlap_ms']:.3f} ms "
                f"({result['overlap_percentage_of_kernel_time']:.2f}%)"
            )

        print(f"\nValidation: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"CSV saved: {OUTPUT_PATH}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
