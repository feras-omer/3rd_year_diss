#!/usr/bin/env python3
"""
Summarize runtime benchmark trial CSVs and produce dissertation-ready plots.

This script expects per-step CSVs produced by `benchmark_diffusion_runtime.py`.
It aggregates them into per-trial and per-map tables for old-vs-new runtime
comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_trial_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def summarize_trial(path: Path) -> Dict[str, object] | None:
    rows = load_trial_rows(path)
    if not rows:
        return None

    last = rows[-1]
    commanded_speed = np.array([safe_float(row["commanded_speed"]) for row in rows], dtype=float)
    steer = np.array([safe_float(row["steer"]) for row in rows], dtype=float)
    min_lidar = np.array([safe_float(row["min_lidar"]) for row in rows], dtype=float)
    elapsed = np.array([safe_float(row["t"]) for row in rows], dtype=float)
    distance_travelled = np.array([safe_float(row["distance_travelled"]) for row in rows], dtype=float)

    steer_rate = np.diff(steer) if len(steer) > 1 else np.array([0.0], dtype=float)

    return {
        "csv_path": str(path),
        "mode": last["mode"],
        "map_name": last["map_name"],
        "trial_id": last["trial_id"],
        "model_path": last["model_path"],
        "success": int(last["lap_complete"]),
        "collided": int(last["collided"]),
        "termination_reason": last["termination_reason"],
        "duration_sec": float(elapsed[-1]),
        "distance_travelled_m": float(distance_travelled[-1]),
        "mean_speed": float(np.mean(commanded_speed)),
        "median_speed": float(np.median(commanded_speed)),
        "max_speed": float(np.max(commanded_speed)),
        "min_lidar": float(np.min(min_lidar)),
        "p05_lidar": float(np.percentile(min_lidar, 5)),
        "mean_lidar": float(np.mean(min_lidar)),
        "steering_variance": float(np.var(steer)),
        "steering_rate_std": float(np.std(steer_rate)),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(trial_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["mode"]), str(row["map_name"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (mode, map_name), rows in sorted(grouped.items()):
        success = np.array([float(row["success"]) for row in rows], dtype=float)
        durations = np.array([float(row["duration_sec"]) for row in rows], dtype=float)
        distances = np.array([float(row["distance_travelled_m"]) for row in rows], dtype=float)
        mean_speed = np.array([float(row["mean_speed"]) for row in rows], dtype=float)
        min_lidar = np.array([float(row["min_lidar"]) for row in rows], dtype=float)
        p05_lidar = np.array([float(row["p05_lidar"]) for row in rows], dtype=float)
        steering_variance = np.array([float(row["steering_variance"]) for row in rows], dtype=float)
        steering_rate_std = np.array([float(row["steering_rate_std"]) for row in rows], dtype=float)

        summary_rows.append(
            {
                "mode": mode,
                "map_name": map_name,
                "trial_count": len(rows),
                "success_rate": float(np.mean(success)),
                "mean_duration_sec": float(np.mean(durations)),
                "median_duration_sec": float(np.median(durations)),
                "mean_distance_travelled_m": float(np.mean(distances)),
                "mean_speed": float(np.mean(mean_speed)),
                "mean_min_lidar": float(np.mean(min_lidar)),
                "mean_p05_lidar": float(np.mean(p05_lidar)),
                "mean_steering_variance": float(np.mean(steering_variance)),
                "mean_steering_rate_std": float(np.mean(steering_rate_std)),
            }
        )

    return summary_rows


def plot_success_rate(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    maps = sorted({str(row["map_name"]) for row in summary_rows})
    modes = sorted({str(row["mode"]) for row in summary_rows})
    x = np.arange(len(maps))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"current": "#2d6a4f", "legacy": "#b85c38"}

    for index, mode in enumerate(modes):
        values = []
        for map_name in maps:
            match = next(
                row for row in summary_rows if row["mode"] == mode and row["map_name"] == map_name
            )
            values.append(float(match["success_rate"]))

        offset = (index - (len(modes) - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=mode.title(), color=colors.get(mode, None))

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Lap Completion Rate by Map")
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_metric_by_map(
    path: Path,
    summary_rows: List[Dict[str, object]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
) -> None:
    maps = sorted({str(row["map_name"]) for row in summary_rows})
    modes = sorted({str(row["mode"]) for row in summary_rows})
    x = np.arange(len(maps))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"current": "#2d6a4f", "legacy": "#b85c38"}

    for index, mode in enumerate(modes):
        values = []
        for map_name in maps:
            match = next(
                row for row in summary_rows if row["mode"] == mode and row["map_name"] == map_name
            )
            values.append(float(match[metric_key]))

        offset = (index - (len(modes) - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=mode.title(), color=colors.get(mode, None))

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_markdown(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Runtime Benchmark Summary",
        "",
        "| Mode | Map | Trials | Success Rate | Mean Duration (s) | Mean Distance (m) | Mean Speed (m/s) | Mean Min LiDAR (m) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            f"{row['mode']} | "
            f"{row['map_name']} | "
            f"{row['trial_count']} | "
            f"{float(row['success_rate']):.3f} | "
            f"{float(row['mean_duration_sec']):.2f} | "
            f"{float(row['mean_distance_travelled_m']):.2f} | "
            f"{float(row['mean_speed']):.2f} | "
            f"{float(row['mean_min_lidar']):.3f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    trial_rows: List[Dict[str, object]] = []
    for csv_path in sorted(args.input_dir.rglob("*.csv")):
        summary = summarize_trial(csv_path)
        if summary is not None:
            trial_rows.append(summary)

    if not trial_rows:
        raise RuntimeError(f"No benchmark CSV files found under {args.input_dir}")

    summary_rows = aggregate_rows(trial_rows)

    trial_fieldnames = [
        "csv_path",
        "mode",
        "map_name",
        "trial_id",
        "model_path",
        "success",
        "collided",
        "termination_reason",
        "duration_sec",
        "distance_travelled_m",
        "mean_speed",
        "median_speed",
        "max_speed",
        "min_lidar",
        "p05_lidar",
        "mean_lidar",
        "steering_variance",
        "steering_rate_std",
    ]
    summary_fieldnames = [
        "mode",
        "map_name",
        "trial_count",
        "success_rate",
        "mean_duration_sec",
        "median_duration_sec",
        "mean_distance_travelled_m",
        "mean_speed",
        "mean_min_lidar",
        "mean_p05_lidar",
        "mean_steering_variance",
        "mean_steering_rate_std",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trial_summary.csv", trial_rows, trial_fieldnames)
    write_csv(args.output_dir / "runtime_summary_by_map.csv", summary_rows, summary_fieldnames)
    write_markdown(args.output_dir / "runtime_summary.md", summary_rows)

    plot_success_rate(args.output_dir / "success_rate_by_map.png", summary_rows)
    plot_metric_by_map(
        args.output_dir / "mean_duration_by_map.png",
        summary_rows,
        metric_key="mean_duration_sec",
        ylabel="Seconds",
        title="Mean Trial Duration by Map",
    )
    plot_metric_by_map(
        args.output_dir / "mean_min_lidar_by_map.png",
        summary_rows,
        metric_key="mean_min_lidar",
        ylabel="Metres",
        title="Mean Minimum LiDAR Clearance by Map",
    )
    plot_metric_by_map(
        args.output_dir / "mean_speed_by_map.png",
        summary_rows,
        metric_key="mean_speed",
        ylabel="m/s",
        title="Mean Commanded Speed by Map",
    )

    print(f"Wrote {args.output_dir / 'trial_summary.csv'}")
    print(f"Wrote {args.output_dir / 'runtime_summary_by_map.csv'}")
    print(f"Wrote {args.output_dir / 'runtime_summary.md'}")
    print(f"Wrote {args.output_dir / 'success_rate_by_map.png'}")
    print(f"Wrote {args.output_dir / 'mean_duration_by_map.png'}")
    print(f"Wrote {args.output_dir / 'mean_min_lidar_by_map.png'}")
    print(f"Wrote {args.output_dir / 'mean_speed_by_map.png'}")


if __name__ == "__main__":
    main()
