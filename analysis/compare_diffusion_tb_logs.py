#!/usr/bin/env python3
"""
Extract and compare TensorBoard scalar logs for the old and new diffusion trainers.

This script is deliberately conservative about comparisons:
- old training exposes only `loss/train`
- new training exposes `loss/train_step`, `loss/train_epoch`, `loss/val_epoch`, `lr`

So the outputs distinguish between:
- comparable step-level training loss curves
- richer new-only epoch/validation curves
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_LOG = ROOT / "old stuff" / "events.out.tfevents.1767726472.feras-Dell-G15-5511.249812.0"
DEFAULT_NEW_LOG_DIR = ROOT / "diffusion_f110" / "tb_logs" / "diffusion"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-log", type=Path, default=DEFAULT_OLD_LOG)
    parser.add_argument(
        "--new-log",
        type=Path,
        default=None,
        help="Optional specific new TensorBoard event file. Defaults to the newest file in diffusion_f110/tb_logs/diffusion.",
    )
    parser.add_argument("--new-log-dir", type=Path, default=DEFAULT_NEW_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window for the smoothed step-loss comparison plot.",
    )
    return parser.parse_args()


def pick_newest_event_file(log_dir: Path) -> Path:
    event_files = sorted(log_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    return event_files[-1]


def load_scalars(path: Path) -> Dict[str, List[dict]]:
    accumulator = event_accumulator.EventAccumulator(str(path))
    accumulator.Reload()

    scalars: Dict[str, List[dict]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [
            {"wall_time": item.wall_time, "step": int(item.step), "value": float(item.value)}
            for item in accumulator.Scalars(tag)
        ]
    return scalars


def write_scalar_csv(path: Path, scalars: Dict[str, List[dict]]) -> None:
    rows = []
    for tag, items in scalars.items():
        for item in items:
            rows.append(
                {
                    "tag": tag,
                    "step": item["step"],
                    "wall_time": item["wall_time"],
                    "value": item["value"],
                }
            )

    rows.sort(key=lambda row: (row["tag"], row["step"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_step_loss_comparison(
    path: Path,
    old_scalars: Dict[str, List[dict]],
    new_scalars: Dict[str, List[dict]],
    smooth_window: int,
) -> None:
    old_items = old_scalars.get("loss/train", [])
    new_items = new_scalars.get("loss/train_step", [])
    if not old_items or not new_items:
        return

    old_steps = np.array([item["step"] for item in old_items], dtype=float)
    old_values = np.array([item["value"] for item in old_items], dtype=float)
    new_steps = np.array([item["step"] for item in new_items], dtype=float)
    new_values = np.array([item["value"] for item in new_items], dtype=float)

    old_smoothed = moving_average(old_values, smooth_window)
    new_smoothed = moving_average(new_values, smooth_window)

    old_smoothed_steps = old_steps[smooth_window - 1 :] if len(old_smoothed) != len(old_steps) else old_steps
    new_smoothed_steps = new_steps[smooth_window - 1 :] if len(new_smoothed) != len(new_steps) else new_steps

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(old_steps, old_values, color="#d9a679", alpha=0.25, linewidth=1.0)
    ax.plot(new_steps, new_values, color="#95d5b2", alpha=0.25, linewidth=1.0)
    ax.plot(old_smoothed_steps, old_smoothed, label="Old train loss", color="#b85c38", linewidth=2.0)
    ax.plot(new_smoothed_steps, new_smoothed, label="New train step loss", color="#2d6a4f", linewidth=2.0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Diffusion Training Loss Comparison")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_new_epoch_curves(path: Path, new_scalars: Dict[str, List[dict]]) -> None:
    train_epoch = new_scalars.get("loss/train_epoch", [])
    val_epoch = new_scalars.get("loss/val_epoch", [])
    lr = new_scalars.get("lr", [])

    if not train_epoch and not val_epoch and not lr:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if train_epoch:
        axes[0].plot(
            [item["step"] for item in train_epoch],
            [item["value"] for item in train_epoch],
            label="Train epoch loss",
            color="#2d6a4f",
            linewidth=2.0,
        )
    if val_epoch:
        axes[0].plot(
            [item["step"] for item in val_epoch],
            [item["value"] for item in val_epoch],
            label="Validation epoch loss",
            color="#1d3557",
            linewidth=2.0,
        )
    axes[0].set_ylabel("Loss")
    axes[0].set_title("New Diffusion Trainer Epoch Metrics")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    if lr:
        axes[1].plot(
            [item["step"] for item in lr],
            [item["value"] for item in lr],
            label="Learning rate",
            color="#6d597a",
            linewidth=2.0,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def summarize_scalar(items: List[dict]) -> Dict[str, float]:
    values = np.array([item["value"] for item in items], dtype=float)
    steps = np.array([item["step"] for item in items], dtype=float)
    return {
        "points": float(len(items)),
        "first_step": float(steps[0]) if len(steps) else float("nan"),
        "last_step": float(steps[-1]) if len(steps) else float("nan"),
        "first_value": float(values[0]) if len(values) else float("nan"),
        "last_value": float(values[-1]) if len(values) else float("nan"),
        "min_value": float(np.min(values)) if len(values) else float("nan"),
    }


def write_summary_markdown(
    path: Path,
    *,
    old_log: Path,
    new_log: Path,
    old_scalars: Dict[str, List[dict]],
    new_scalars: Dict[str, List[dict]],
) -> None:
    old_train = summarize_scalar(old_scalars.get("loss/train", []))
    new_train = summarize_scalar(new_scalars.get("loss/train_step", []))
    new_train_epoch = summarize_scalar(new_scalars.get("loss/train_epoch", []))
    new_val_epoch = summarize_scalar(new_scalars.get("loss/val_epoch", []))

    lines = [
        "# Diffusion TensorBoard Comparison",
        "",
        f"- Old log: `{old_log}`",
        f"- New log: `{new_log}`",
        "",
        "## Available Scalars",
        "",
        f"- Old: `{', '.join(old_scalars.keys())}`",
        f"- New: `{', '.join(new_scalars.keys())}`",
        "",
        "## Step-Level Training Loss",
        "",
        "| Run | Points | First Step | Last Step | First Loss | Last Loss | Minimum Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Old `loss/train` | {int(old_train['points'])} | {int(old_train['first_step'])} | {int(old_train['last_step'])} | {old_train['first_value']:.6f} | {old_train['last_value']:.6f} | {old_train['min_value']:.6f} |",
        f"| New `loss/train_step` | {int(new_train['points'])} | {int(new_train['first_step'])} | {int(new_train['last_step'])} | {new_train['first_value']:.6f} | {new_train['last_value']:.6f} | {new_train['min_value']:.6f} |",
        "",
        "## New Trainer Epoch Metrics",
        "",
        "| Scalar | Points | First Value | Last Value | Minimum Value |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `loss/train_epoch` | {int(new_train_epoch['points'])} | {new_train_epoch['first_value']:.6f} | {new_train_epoch['last_value']:.6f} | {new_train_epoch['min_value']:.6f} |",
        f"| `loss/val_epoch` | {int(new_val_epoch['points'])} | {new_val_epoch['first_value']:.6f} | {new_val_epoch['last_value']:.6f} | {new_val_epoch['min_value']:.6f} |",
        "",
        "## Method Note",
        "",
        "- The old and new trainers do not log identical scalars, so TensorBoard comparison is used as supporting evidence about optimisation behaviour, not as the sole measure of model quality.",
        "- The more defensible primary comparisons remain the offline checkpoint evaluation and the runtime benchmark.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    new_log = args.new_log if args.new_log is not None else pick_newest_event_file(args.new_log_dir)
    old_scalars = load_scalars(args.old_log)
    new_scalars = load_scalars(new_log)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_scalar_csv(args.output_dir / "old_tb_scalars.csv", old_scalars)
    write_scalar_csv(args.output_dir / "new_tb_scalars.csv", new_scalars)
    plot_step_loss_comparison(
        args.output_dir / "loss_comparison.png",
        old_scalars,
        new_scalars,
        args.smooth_window,
    )
    plot_new_epoch_curves(args.output_dir / "new_epoch_curves.png", new_scalars)
    write_summary_markdown(
        args.output_dir / "tb_summary.md",
        old_log=args.old_log,
        new_log=new_log,
        old_scalars=old_scalars,
        new_scalars=new_scalars,
    )

    print(f"Using new log {new_log}")
    print(f"Wrote {args.output_dir / 'old_tb_scalars.csv'}")
    print(f"Wrote {args.output_dir / 'new_tb_scalars.csv'}")
    print(f"Wrote {args.output_dir / 'loss_comparison.png'}")
    print(f"Wrote {args.output_dir / 'new_epoch_curves.png'}")
    print(f"Wrote {args.output_dir / 'tb_summary.md'}")


if __name__ == "__main__":
    main()
