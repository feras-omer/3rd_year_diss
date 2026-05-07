import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import F110Dataset
from diffusion_model import DiffusionPolicy

"""
Trains or fine-tunes the diffusion policy on expert F1TENTH trajectories.

The sampler emphasizes likely hairpin segments using both steering magnitude and
LiDAR geometry, while still keeping broad coverage of the rest of the dataset.
"""


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "expert_merged.npz"
MODEL_PATH = ROOT / "models" / "diffusion_policy.pt"
LOG_DIR = ROOT / "tb_logs" / "diffusion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.9
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 120
HIDDEN_DIM = 256
NUMBER_OF_WORKERS = 0
MAX_STEERING_ANGLE = 0.4
LIDAR_FEATURE_COUNT = 108


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def compute_hairpin_scores(dataset, indices):
    observations = dataset.observations[indices].numpy()
    actions = dataset.actions[indices].numpy()

    lidar = observations[:, :LIDAR_FEATURE_COUNT]
    yaw_rate = np.abs(observations[:, LIDAR_FEATURE_COUNT + 1])
    steering = np.abs(actions[:, 0])

    center = np.mean(lidar[:, 50:58], axis=1)
    left_front = np.mean(lidar[:, 62:74], axis=1)
    right_front = np.mean(lidar[:, 34:46], axis=1)
    asymmetry = np.abs(left_front - right_front)

    steer_ratio = np.clip(steering / MAX_STEERING_ANGLE, 0.0, 1.0)
    center_closeness = np.clip(1.0 - center, 0.0, 1.0)

    hairpin_score = (
        1.6 * steer_ratio
        + 1.1 * center_closeness
        + 0.9 * asymmetry
        + 0.5 * yaw_rate
    )

    hairpin_mask = (
        (steering >= 0.16)
        & (center <= 0.55)
        & ((asymmetry >= 0.10) | (yaw_rate >= 0.18))
    )

    return hairpin_score, hairpin_mask


def build_train_sampler(dataset, train_indices):
    steering = dataset.actions[train_indices, 0].abs().numpy()
    steer_ratio = np.clip(steering / MAX_STEERING_ANGLE, 0.0, 1.0)
    hairpin_score, hairpin_mask = compute_hairpin_scores(dataset, train_indices)

    weights = 1.0 + 2.5 * steer_ratio**2
    weights += 2.0 * (steering >= 0.15)
    weights += 3.5 * (steering >= 0.25)
    weights += 5.0 * (steering >= 0.32)
    weights += 2.0 * hairpin_score
    weights += 6.0 * hairpin_mask.astype(np.float32)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )


def sample_noisy_actions(actions_normalised, timestep):
    noise = torch.randn_like(actions_normalised)
    noisy_actions = actions_normalised + noise * timestep.unsqueeze(1)
    return noisy_actions, noise


def compute_loss(predicted_noise, target_noise, observations, actions_normalised):
    per_dim_loss = F.smooth_l1_loss(
        predicted_noise,
        target_noise,
        reduction="none",
    )

    lidar = observations[:, :LIDAR_FEATURE_COUNT]
    center = torch.mean(lidar[:, 50:58], dim=1)
    left_front = torch.mean(lidar[:, 62:74], dim=1)
    right_front = torch.mean(lidar[:, 34:46], dim=1)
    asymmetry = torch.abs(left_front - right_front)

    steering_emphasis = 1.0 + 2.5 * torch.abs(actions_normalised[:, 0])
    hairpin_emphasis = 1.0 + 1.5 * torch.clamp(1.0 - center, min=0.0, max=1.0)
    hairpin_emphasis += 1.0 * torch.clamp(asymmetry, min=0.0, max=1.0)

    weighted_loss = steering_emphasis * hairpin_emphasis * (
        2.8 * per_dim_loss[:, 0] + 1.0 * per_dim_loss[:, 1]
    )
    return weighted_loss.mean()


def evaluate(model, data_loader, action_mean, action_std):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for observations, actions in data_loader:
            observations = observations.to(DEVICE)
            actions = actions.to(DEVICE)
            actions_normalised = (actions - action_mean) / action_std

            timestep = torch.rand(len(observations), device=DEVICE)
            noisy_actions, noise = sample_noisy_actions(actions_normalised, timestep)
            predicted_noise = model(observations, noisy_actions, timestep)

            batch_loss = compute_loss(
                predicted_noise,
                noise,
                observations,
                actions_normalised,
            )
            batch_size = observations.shape[0]
            total_loss += batch_loss.item() * batch_size
            total_count += batch_size

    model.train()
    return total_loss / max(total_count, 1)


def load_or_initialize_model(model_path, observation_dim, action_dim, from_scratch):
    model = DiffusionPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    if from_scratch or not model_path.exists():
        return model, None

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    model.load_state_dict(checkpoint)
    return model, None


def main():
    args = parse_args()

    torch.manual_seed(7)
    np.random.seed(7)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(LOG_DIR))

    full_dataset = F110Dataset(args.dataset, augment_mirror=False)
    dataset_size = len(full_dataset)
    permutation = np.random.permutation(dataset_size)

    train_size = int(TRAIN_SPLIT * dataset_size)
    train_indices = permutation[:train_size].tolist()
    val_indices = permutation[train_size:].tolist()

    train_dataset = F110Dataset(args.dataset, augment_mirror=True)
    val_dataset = F110Dataset(args.dataset, augment_mirror=False)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    action_mean = full_dataset.actions[train_indices].mean(dim=0).to(DEVICE)
    action_std = full_dataset.actions[train_indices].std(dim=0).clamp_min(1e-3).to(DEVICE)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=build_train_sampler(full_dataset, train_indices),
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    observation_dim = full_dataset.observations.shape[1]
    action_dim = full_dataset.actions.shape[1]
    model, existing_checkpoint = load_or_initialize_model(
        args.model,
        observation_dim,
        action_dim,
        args.from_scratch,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val_loss = float("inf")
    best_checkpoint = None
    global_step = 0

    if existing_checkpoint is not None:
        print(f"Fine-tuning from checkpoint: {args.model}")
    else:
        print("Training from scratch")

    hairpin_score, hairpin_mask = compute_hairpin_scores(full_dataset, train_indices)
    print(
        "train samples="
        f"{len(train_indices)} hairpin_like={int(hairpin_mask.sum())} "
        f"hairpin_ratio={hairpin_mask.mean():.3f} "
        f"hairpin_score_mean={hairpin_score.mean():.3f}"
    )

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        sample_count = 0

        for observations, actions in progress_bar:
            observations = observations.to(DEVICE)
            actions = actions.to(DEVICE)
            actions_normalised = (actions - action_mean) / action_std

            timestep = torch.rand(len(observations), device=DEVICE)
            noisy_actions, noise = sample_noisy_actions(actions_normalised, timestep)
            predicted_noise = model(observations, noisy_actions, timestep)

            loss = compute_loss(
                predicted_noise,
                noise,
                observations,
                actions_normalised,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = observations.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, action_mean, action_std)
        scheduler.step()

        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        writer.add_scalar("loss/val_epoch", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "action_mean": action_mean.detach().cpu(),
                "action_std": action_std.detach().cpu(),
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "hidden_dim": HIDDEN_DIM,
            }
            torch.save(best_checkpoint, args.model)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} best_val={best_val_loss:.5f}"
        )

    writer.close()

    if best_checkpoint is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    print(f"Best diffusion model saved to {args.model}")


if __name__ == "__main__":
    main()
