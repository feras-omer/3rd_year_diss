#!/usr/bin/env python3
"""
Offline comparison for the legacy and current diffusion checkpoints.

This script compares the old mid-year checkpoint against the current checkpoint
on the same dataset and exports:

- per-model summary metrics
- side-by-side comparison tables
- subset metrics for sharp-turn and turn-direction slices
- a short markdown summary suitable for dissertation notes

The comparison is intentionally offline. It measures how each checkpoint,
decoded with its native inference routine, reconstructs expert actions from the
same recorded observations. Runtime comparisons should be handled separately.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "diffusion_f110" / "data" / "expert_merged.npz"
DEFAULT_OLD_CHECKPOINT = ROOT / "old stuff" / "diffusion_policy.pt"
DEFAULT_NEW_CHECKPOINT = ROOT / "diffusion_f110" / "models" / "diffusion_policy.pt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "offline"

DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 7
DEFAULT_EVAL_FRACTION = 0.1

LIDAR_FEATURE_COUNT = 108
MAX_STEERING_ANGLE = 0.4

LEGACY_DENOISE_STEPS = 6
CURRENT_DENOISE_STEPS = 8


class DiffusionPolicyCompat(nn.Module):
    """Minimal policy class that matches the stored checkpoint structure."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        observation: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        timestep = timestep.unsqueeze(1)
        model_input = torch.cat([observation, noisy_action, timestep], dim=1)
        return self.net(model_input)


@dataclass
class CheckpointSpec:
    label: str
    path: Path
    state_dict: Dict[str, torch.Tensor]
    observation_dim: int
    action_dim: int
    hidden_dim: int
    outputs_are_normalised: bool
    action_mean: torch.Tensor
    action_std: torch.Tensor
    denoise_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--old-checkpoint", type=Path, default=DEFAULT_OLD_CHECKPOINT)
    parser.add_argument("--new-checkpoint", type=Path, default=DEFAULT_NEW_CHECKPOINT)
    parser.add_argument(
        "--split",
        choices=["full", "holdout"],
        default="full",
        help="Use the full dataset or a deterministic holdout slice.",
    )
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def infer_dims_from_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> tuple[int, int, int]:
    first_weight = state_dict["net.0.weight"]
    final_weight = state_dict["net.4.weight"]

    hidden_dim = int(first_weight.shape[0])
    action_dim = int(final_weight.shape[0])
    observation_dim = int(first_weight.shape[1] - action_dim - 1)
    return observation_dim, action_dim, hidden_dim


def load_checkpoint_spec(
    label: str,
    path: Path,
    denoise_steps: int,
) -> CheckpointSpec:
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        observation_dim = int(checkpoint.get("observation_dim", 112))
        action_dim = int(checkpoint.get("action_dim", 2))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        outputs_are_normalised = True
        action_mean = checkpoint["action_mean"].float()
        action_std = checkpoint["action_std"].float()
    else:
        state_dict = checkpoint
        observation_dim, action_dim, hidden_dim = infer_dims_from_state_dict(state_dict)
        outputs_are_normalised = False
        action_mean = torch.zeros(action_dim, dtype=torch.float32)
        action_std = torch.ones(action_dim, dtype=torch.float32)

    return CheckpointSpec(
        label=label,
        path=path,
        state_dict=state_dict,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        outputs_are_normalised=outputs_are_normalised,
        action_mean=action_mean,
        action_std=action_std,
        denoise_steps=denoise_steps,
    )


def load_dataset(
    dataset_path: Path,
    split: str,
    eval_fraction: float,
    seed: int,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(dataset_path)
    observations = data["obs"].astype(np.float32)
    actions = data["act"].astype(np.float32)

    if split == "holdout":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(observations))
        eval_count = max(1, int(len(observations) * eval_fraction))
        selected = permutation[-eval_count:]
        observations = observations[selected]
        actions = actions[selected]

    if max_samples is not None:
        observations = observations[:max_samples]
        actions = actions[:max_samples]

    return observations, actions


def build_subsets(
    observations: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, np.ndarray]:
    lidar = observations[:, :LIDAR_FEATURE_COUNT]
    yaw_rate = np.abs(observations[:, LIDAR_FEATURE_COUNT + 1])
    steering = actions[:, 0]
    steering_abs = np.abs(steering)

    center = np.mean(lidar[:, 50:58], axis=1)
    left_front = np.mean(lidar[:, 62:74], axis=1)
    right_front = np.mean(lidar[:, 34:46], axis=1)
    asymmetry = np.abs(left_front - right_front)

    hairpin = (
        (steering_abs >= 0.16)
        & (center <= 0.55)
        & ((asymmetry >= 0.10) | (yaw_rate >= 0.18))
    )

    return {
        "all": np.ones(len(actions), dtype=bool),
        "hairpin": hairpin,
        "non_hairpin": ~hairpin,
        "left_turn": steering >= 0.10,
        "right_turn": steering <= -0.10,
        "straight": steering_abs < 0.05,
        "high_steer": steering_abs >= 0.25,
    }


def batched_indices(total_count: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, total_count, batch_size):
        yield slice(start, min(start + batch_size, total_count))


def infer_actions(
    spec: CheckpointSpec,
    observations: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionPolicyCompat(
        observation_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        hidden_dim=spec.hidden_dim,
    ).to(device)
    model.load_state_dict(spec.state_dict)
    model.eval()

    action_mean = spec.action_mean.to(device)
    action_std = spec.action_std.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    predictions: List[np.ndarray] = []

    with torch.no_grad():
        for batch_slice in batched_indices(len(observations), batch_size):
            batch = torch.from_numpy(observations[batch_slice]).to(device)

            if spec.outputs_are_normalised:
                action = torch.randn(
                    (len(batch), spec.action_dim),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                for timestep_value in np.linspace(1.0, 0.0, spec.denoise_steps):
                    timestep = torch.full(
                        (len(batch),),
                        float(timestep_value),
                        device=device,
                        dtype=torch.float32,
                    )
                    noise_estimate = model(batch, action, timestep)
                    action = action - noise_estimate / spec.denoise_steps

                action = action * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
            else:
                action = torch.zeros(
                    (len(batch), spec.action_dim),
                    device=device,
                    dtype=torch.float32,
                )
                timestep = torch.ones(len(batch), device=device, dtype=torch.float32)
                for _ in range(spec.denoise_steps):
                    noise_estimate = model(batch, action, timestep)
                    action = action - 0.5 * noise_estimate

            predictions.append(action.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def compute_metrics(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    error = predictions - ground_truth
    steering_error = error[:, 0]
    speed_error = error[:, 1]

    return {
        "sample_count": float(len(ground_truth)),
        "steering_mae": float(np.mean(np.abs(steering_error))),
        "steering_rmse": float(np.sqrt(np.mean(steering_error**2))),
        "speed_mae": float(np.mean(np.abs(speed_error))),
        "speed_rmse": float(np.sqrt(np.mean(speed_error**2))),
        "combined_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "mean_pred_steering": float(np.mean(predictions[:, 0])),
        "mean_pred_speed": float(np.mean(predictions[:, 1])),
        "mean_true_steering": float(np.mean(ground_truth[:, 0])),
        "mean_true_speed": float(np.mean(ground_truth[:, 1])),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_markdown_summary(
    path: Path,
    *,
    dataset_path: Path,
    split: str,
    eval_fraction: float,
    total_samples: int,
    overall_rows: List[Dict[str, object]],
    subset_rows: List[Dict[str, object]],
) -> None:
    by_model = {row["model"]: row for row in overall_rows}
    old_row = by_model["old"]
    new_row = by_model["new"]

    steering_delta = float(old_row["steering_mae"]) - float(new_row["steering_mae"])
    speed_delta = float(old_row["speed_mae"]) - float(new_row["speed_mae"])
    combined_delta = float(old_row["combined_rmse"]) - float(new_row["combined_rmse"])

    lines = [
        "# Diffusion Checkpoint Comparison",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Split mode: `{split}`",
        f"- Evaluation fraction: `{eval_fraction}`",
        f"- Samples evaluated: `{total_samples}`",
        "",
        "## Overall",
        "",
        "| Model | Steering MAE | Steering RMSE | Speed MAE | Speed RMSE | Combined RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in overall_rows:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{format_float(float(row['steering_mae']))} | "
            f"{format_float(float(row['steering_rmse']))} | "
            f"{format_float(float(row['speed_mae']))} | "
            f"{format_float(float(row['speed_rmse']))} | "
            f"{format_float(float(row['combined_rmse']))} |"
        )

    lines.extend(
        [
            "",
            "## Improvement",
            "",
            f"- Steering MAE improvement from old to new: `{steering_delta:.6f}`",
            f"- Speed MAE improvement from old to new: `{speed_delta:.6f}`",
            f"- Combined RMSE improvement from old to new: `{combined_delta:.6f}`",
            "",
            "Positive values above mean the new checkpoint is better on that metric.",
            "",
            "## Subsets",
            "",
            "| Model | Subset | Samples | Steering MAE | Speed MAE | Combined RMSE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in subset_rows:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{row['subset']} | "
            f"{int(float(row['sample_count']))} | "
            f"{format_float(float(row['steering_mae']))} | "
            f"{format_float(float(row['speed_mae']))} | "
            f"{format_float(float(row['combined_rmse']))} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot_overall_metrics(path: Path, overall_rows: List[Dict[str, object]]) -> None:
    metric_names = [
        "steering_mae",
        "steering_rmse",
        "speed_mae",
        "speed_rmse",
        "combined_rmse",
    ]
    metric_labels = [
        "Steering MAE",
        "Steering RMSE",
        "Speed MAE",
        "Speed RMSE",
        "Combined RMSE",
    ]

    by_model = {row["model"]: row for row in overall_rows}
    old_values = [float(by_model["old"][metric]) for metric in metric_names]
    new_values = [float(by_model["new"][metric]) for metric in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, old_values, width, label="Old", color="#b85c38")
    ax.bar(x + width / 2, new_values, width, label="New", color="#2d6a4f")

    ax.set_ylabel("Error")
    ax.set_title("Offline Diffusion Checkpoint Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_subset_metric(
    path: Path,
    subset_rows: List[Dict[str, object]],
    *,
    metric_name: str,
    metric_label: str,
    subset_order: List[str],
    title: str,
) -> None:
    filtered = [row for row in subset_rows if row["subset"] in subset_order]
    by_key = {(row["model"], row["subset"]): row for row in filtered}

    old_values = [float(by_key[("old", subset)][metric_name]) for subset in subset_order]
    new_values = [float(by_key[("new", subset)][metric_name]) for subset in subset_order]

    x = np.arange(len(subset_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, old_values, width, label="Old", color="#b85c38")
    ax.bar(x + width / 2, new_values, width, label="New", color="#2d6a4f")

    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([subset.replace("_", " ").title() for subset in subset_order])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_subset_sample_counts(
    path: Path,
    subset_rows: List[Dict[str, object]],
    subset_order: List[str],
) -> None:
    sample_counts = {}
    for row in subset_rows:
        subset = row["subset"]
        if subset in subset_order:
            sample_counts[subset] = int(float(row["sample_count"]))

    counts = [sample_counts[subset] for subset in subset_order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        np.arange(len(subset_order)),
        counts,
        color="#3a86ff",
        width=0.6,
    )
    ax.set_ylabel("Samples")
    ax.set_title("Offline Evaluation Subset Sizes")
    ax.set_xticks(np.arange(len(subset_order)))
    ax.set_xticklabels([subset.replace("_", " ").title() for subset in subset_order])
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    observations, actions = load_dataset(
        dataset_path=args.dataset,
        split=args.split,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    subsets = build_subsets(observations, actions)

    specs = [
        load_checkpoint_spec("old", args.old_checkpoint, LEGACY_DENOISE_STEPS),
        load_checkpoint_spec("new", args.new_checkpoint, CURRENT_DENOISE_STEPS),
    ]

    overall_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []

    for index, spec in enumerate(specs):
        predictions = infer_actions(
            spec=spec,
            observations=observations,
            batch_size=args.batch_size,
            seed=args.seed + index,
        )

        overall_metrics = compute_metrics(actions, predictions)
        overall_metrics.update(
            {
                "model": spec.label,
                "checkpoint_path": str(spec.path),
                "outputs_are_normalised": spec.outputs_are_normalised,
                "denoise_steps": spec.denoise_steps,
            }
        )
        overall_rows.append(overall_metrics)

        for subset_name, subset_mask in subsets.items():
            if not np.any(subset_mask):
                continue

            subset_metrics = compute_metrics(actions[subset_mask], predictions[subset_mask])
            subset_metrics.update(
                {
                    "model": spec.label,
                    "subset": subset_name,
                    "checkpoint_path": str(spec.path),
                }
            )
            subset_rows.append(subset_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_fieldnames = [
        "model",
        "checkpoint_path",
        "outputs_are_normalised",
        "denoise_steps",
        "sample_count",
        "steering_mae",
        "steering_rmse",
        "speed_mae",
        "speed_rmse",
        "combined_rmse",
        "mean_pred_steering",
        "mean_pred_speed",
        "mean_true_steering",
        "mean_true_speed",
    ]
    subset_fieldnames = [
        "model",
        "subset",
        "checkpoint_path",
        "sample_count",
        "steering_mae",
        "steering_rmse",
        "speed_mae",
        "speed_rmse",
        "combined_rmse",
        "mean_pred_steering",
        "mean_pred_speed",
        "mean_true_steering",
        "mean_true_speed",
    ]

    write_csv(args.output_dir / "checkpoint_metrics.csv", overall_rows, overall_fieldnames)
    write_csv(
        args.output_dir / "checkpoint_metrics_by_subset.csv",
        subset_rows,
        subset_fieldnames,
    )
    write_markdown_summary(
        args.output_dir / "checkpoint_comparison.md",
        dataset_path=args.dataset,
        split=args.split,
        eval_fraction=args.eval_fraction,
        total_samples=len(actions),
        overall_rows=overall_rows,
        subset_rows=subset_rows,
    )

    subset_order = ["hairpin", "high_steer", "left_turn", "right_turn", "straight"]
    plot_overall_metrics(args.output_dir / "overall_metrics.png", overall_rows)
    plot_subset_metric(
        args.output_dir / "subset_steering_mae.png",
        subset_rows,
        metric_name="steering_mae",
        metric_label="Steering MAE",
        subset_order=subset_order,
        title="Steering Error by Subset",
    )
    plot_subset_metric(
        args.output_dir / "subset_speed_mae.png",
        subset_rows,
        metric_name="speed_mae",
        metric_label="Speed MAE",
        subset_order=subset_order,
        title="Speed Error by Subset",
    )
    plot_subset_metric(
        args.output_dir / "subset_combined_rmse.png",
        subset_rows,
        metric_name="combined_rmse",
        metric_label="Combined RMSE",
        subset_order=subset_order,
        title="Combined Error by Subset",
    )
    plot_subset_sample_counts(
        args.output_dir / "subset_sample_counts.png",
        subset_rows,
        subset_order=subset_order,
    )

    print(f"Wrote {args.output_dir / 'checkpoint_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'checkpoint_metrics_by_subset.csv'}")
    print(f"Wrote {args.output_dir / 'checkpoint_comparison.md'}")
    print(f"Wrote {args.output_dir / 'overall_metrics.png'}")
    print(f"Wrote {args.output_dir / 'subset_steering_mae.png'}")
    print(f"Wrote {args.output_dir / 'subset_speed_mae.png'}")
    print(f"Wrote {args.output_dir / 'subset_combined_rmse.png'}")
    print(f"Wrote {args.output_dir / 'subset_sample_counts.png'}")


if __name__ == "__main__":
    main()
