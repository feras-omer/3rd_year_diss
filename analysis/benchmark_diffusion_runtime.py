#!/usr/bin/env python3
"""
Common runtime benchmark for legacy and current diffusion controllers.

This script runs one trial and logs a consistent CSV schema for both modes:

- legacy: mid-year diffusion runner logic
- current: current diffusion runner logic

The benchmark is intended to be launched while the simulator is already running.
It does not launch ROS nodes for the map or simulator itself.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import rclpy
import torch
import torch.nn as nn
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class DiffusionPolicyCompat(nn.Module):
    """Minimal policy class compatible with both saved checkpoint formats."""

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
class ModelSpec:
    path: Path
    state_dict: Dict[str, torch.Tensor]
    observation_dim: int
    action_dim: int
    hidden_dim: int
    outputs_are_normalised: bool
    action_mean: torch.Tensor
    action_std: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["legacy", "current"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--map-name", type=str, default="unknown_map")
    parser.add_argument("--trial-id", type=str, default="trial_01")

    parser.add_argument("--scan-topic", type=str, default="/scan")
    parser.add_argument("--odom-topic", type=str, default="/ego_racecar/odom")
    parser.add_argument("--drive-topic", type=str, default="/drive")

    parser.add_argument("--control-period", type=float, default=0.1)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--collision-threshold", type=float, default=0.25)

    parser.add_argument("--maximum-speed", type=float, default=5.0)
    parser.add_argument("--maximum-steering-angle", type=float, default=0.4)
    parser.add_argument("--maximum-lidar-range", type=float, default=10.0)
    parser.add_argument("--lidar-stride", type=int, default=10)

    parser.add_argument("--lap-radius", type=float, default=0.8)
    parser.add_argument("--min-lap-time", type=float, default=15.0)
    parser.add_argument("--min-lap-distance", type=float, default=35.0)

    parser.add_argument("--legacy-denoise-steps", type=int, default=6)
    parser.add_argument("--legacy-curvature-scaling", type=float, default=0.8)
    parser.add_argument("--legacy-wheelbase", type=float, default=0.33)
    parser.add_argument("--legacy-lookahead-distance", type=float, default=1.2)
    parser.add_argument("--legacy-speed-gain", type=float, default=3.0)

    parser.add_argument("--current-denoise-steps", type=int, default=8)
    parser.add_argument("--current-forward-speed-gain", type=float, default=3.2)
    parser.add_argument("--current-turn-speed-penalty", type=float, default=0.65)
    parser.add_argument("--current-min-command-speed", type=float, default=0.8)

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


def load_model_spec(path: Path) -> ModelSpec:
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

    return ModelSpec(
        path=path,
        state_dict=state_dict,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        outputs_are_normalised=outputs_are_normalised,
        action_mean=action_mean,
        action_std=action_std,
    )


class RuntimeBenchmark(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("diffusion_runtime_benchmark")
        self.args = args
        self.device = "cpu"

        spec = load_model_spec(args.model_path)
        self.policy = DiffusionPolicyCompat(
            observation_dim=spec.observation_dim,
            action_dim=spec.action_dim,
            hidden_dim=spec.hidden_dim,
        ).to(self.device)
        self.policy.load_state_dict(spec.state_dict)
        self.policy.eval()

        self.outputs_are_normalised = spec.outputs_are_normalised
        self.action_mean = spec.action_mean.to(self.device)
        self.action_std = spec.action_std.to(self.device)

        self.latest_scan: LaserScan | None = None
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.current_position = np.zeros(2, dtype=np.float32)
        self.current_yaw = 0.0

        self.start_position: np.ndarray | None = None
        self.last_position: np.ndarray | None = None
        self.distance_travelled = 0.0

        self.previous_action = np.array(
            [0.0, args.maximum_speed],
            dtype=np.float32,
        )

        self.trial_started_at = time.time()
        self.terminated = False
        self.termination_reason = ""

        self.output_csv = args.output_csv
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_csv.open("w", newline="")
        self.csv_writer = csv.DictWriter(
            self.log_file,
            fieldnames=[
                "mode",
                "map_name",
                "trial_id",
                "model_path",
                "t",
                "x",
                "y",
                "yaw",
                "odom_speed",
                "yaw_rate",
                "raw_action_0",
                "raw_action_1",
                "steer",
                "commanded_speed",
                "speed_limit",
                "min_lidar",
                "distance_from_start",
                "distance_travelled",
                "lap_complete",
                "collided",
                "terminated",
                "termination_reason",
            ],
        )
        self.csv_writer.writeheader()

        self.create_subscription(LaserScan, args.scan_topic, self._scan_callback, 20)
        self.create_subscription(Odometry, args.odom_topic, self._odom_callback, 50)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            args.drive_topic,
            10,
        )

        self.timer = self.create_timer(args.control_period, self._control_step)
        self.get_logger().info(
            f"Runtime benchmark started mode={args.mode} "
            f"model={args.model_path} map={args.map_name} trial={args.trial_id}"
        )

    def _scan_callback(self, message: LaserScan) -> None:
        self.latest_scan = message

    def _odom_callback(self, message: Odometry) -> None:
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)
        self.current_yaw_rate = message.twist.twist.angular.z

        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        self.current_position[:] = [x, y]

        z = float(message.pose.pose.orientation.z)
        w = float(message.pose.pose.orientation.w)
        self.current_yaw = math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

        if self.start_position is None:
            self.start_position = self.current_position.copy()
            self.last_position = self.current_position.copy()
            return

        if self.last_position is not None:
            step_distance = float(np.linalg.norm(self.current_position - self.last_position))
            self.distance_travelled += step_distance
        self.last_position = self.current_position.copy()

    def _build_observation(self) -> torch.Tensor:
        assert self.latest_scan is not None

        lidar_ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.args.maximum_lidar_range,
            posinf=self.args.maximum_lidar_range,
            neginf=0.0,
        )

        lidar_downsampled = lidar_ranges[:: self.args.lidar_stride]
        lidar_normalised = np.clip(
            lidar_downsampled / self.args.maximum_lidar_range,
            0.0,
            1.0,
        )

        speed_normalised = np.clip(
            self.current_speed / self.args.maximum_speed,
            0.0,
            1.0,
        )
        yaw_rate_clipped = np.clip(self.current_yaw_rate, -1.0, 1.0)

        if self.args.mode == "legacy":
            history = np.array([0.0, 0.0], dtype=np.float32)
        else:
            history = self.previous_action.astype(np.float32)

        observation = np.concatenate(
            [lidar_normalised, [speed_normalised, yaw_rate_clipped], history]
        ).astype(np.float32)
        return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    def _sample_legacy_action(self, observation: torch.Tensor) -> np.ndarray:
        action = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        timestep = torch.ones((1,), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for _ in range(self.args.legacy_denoise_steps):
                noise_estimate = self.policy(observation.to(self.device), action, timestep)
                action = action - 0.5 * noise_estimate

        return action.squeeze(0).cpu().numpy()

    def _sample_current_action(self, observation: torch.Tensor) -> np.ndarray:
        if self.outputs_are_normalised:
            action = torch.randn((1, 2), dtype=torch.float32, device=self.device)
        else:
            action = torch.tensor(
                self.previous_action,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

        with torch.no_grad():
            for timestep_value in np.linspace(1.0, 0.0, self.args.current_denoise_steps):
                timestep = torch.full(
                    (1,),
                    float(timestep_value),
                    dtype=torch.float32,
                    device=self.device,
                )
                noise_estimate = self.policy(observation.to(self.device), action, timestep)
                action = action - noise_estimate / self.args.current_denoise_steps

        if self.outputs_are_normalised:
            action = action * self.action_std.unsqueeze(0) + self.action_mean.unsqueeze(0)

        return action.squeeze(0).cpu().numpy()

    def _legacy_curvature_to_steering(self, curvature_signal: float) -> float:
        target_angle = curvature_signal * self.args.legacy_curvature_scaling
        target_x = self.args.legacy_lookahead_distance
        target_y = math.tan(target_angle) * target_x

        steering_angle = math.atan2(
            2.0 * self.args.legacy_wheelbase * target_y,
            target_x**2,
        )
        return float(
            np.clip(
                steering_angle,
                -self.args.maximum_steering_angle,
                self.args.maximum_steering_angle,
            )
        )

    def _current_speed_limit(self, observation: torch.Tensor, steering_angle: float) -> float:
        forward_lidar = float(np.mean(observation[0, 52:56].cpu().numpy()))
        base_speed = self.args.current_forward_speed_gain * forward_lidar
        steering_ratio = abs(steering_angle) / self.args.maximum_steering_angle
        corner_penalty = 1.0 - self.args.current_turn_speed_penalty * steering_ratio
        return float(
            np.clip(
                base_speed * max(corner_penalty, 0.25),
                self.args.current_min_command_speed,
                self.args.maximum_speed,
            )
        )

    def _stop_vehicle(self) -> None:
        command = AckermannDriveStamped()
        command.drive.steering_angle = 0.0
        command.drive.speed = 0.0
        self.drive_publisher.publish(command)

    def _check_termination(
        self,
        *,
        elapsed: float,
        min_lidar: float,
        distance_from_start: float,
    ) -> tuple[bool, bool, bool, str]:
        collided = min_lidar < self.args.collision_threshold
        lap_complete = False

        if (
            self.start_position is not None
            and elapsed >= self.args.min_lap_time
            and self.distance_travelled >= self.args.min_lap_distance
            and distance_from_start <= self.args.lap_radius
        ):
            lap_complete = True

        if collided:
            return lap_complete, collided, True, "collision"
        if lap_complete:
            return lap_complete, collided, True, "lap_complete"
        if elapsed >= self.args.timeout_sec:
            return lap_complete, collided, True, "timeout"
        return lap_complete, collided, False, ""

    def _control_step(self) -> None:
        if self.terminated or self.latest_scan is None or self.start_position is None:
            return

        observation = self._build_observation()

        if self.args.mode == "legacy":
            raw_action = self._sample_legacy_action(observation)
            steering_angle = self._legacy_curvature_to_steering(float(raw_action[0]))
            forward_lidar = float(np.mean(observation[0, 52:56].cpu().numpy()))
            speed_limit = float(
                np.clip(
                    self.args.legacy_speed_gain * forward_lidar,
                    1.0,
                    self.args.maximum_speed,
                )
            )
            commanded_speed = speed_limit
        else:
            raw_action = self._sample_current_action(observation)
            steering_angle = float(
                np.clip(
                    raw_action[0],
                    -self.args.maximum_steering_angle,
                    self.args.maximum_steering_angle,
                )
            )
            predicted_speed = float(np.clip(raw_action[1], 0.0, self.args.maximum_speed))
            speed_limit = self._current_speed_limit(observation, steering_angle)
            commanded_speed = min(
                max(predicted_speed, self.args.current_min_command_speed),
                speed_limit,
            )
            self.previous_action[:] = [steering_angle, commanded_speed]

        min_lidar = float(np.nanmin(np.asarray(self.latest_scan.ranges, dtype=np.float32)))
        elapsed = time.time() - self.trial_started_at
        distance_from_start = float(np.linalg.norm(self.current_position - self.start_position))

        lap_complete, collided, terminated, termination_reason = self._check_termination(
            elapsed=elapsed,
            min_lidar=min_lidar,
            distance_from_start=distance_from_start,
        )

        command = AckermannDriveStamped()
        command.drive.steering_angle = steering_angle
        command.drive.speed = commanded_speed
        self.drive_publisher.publish(command)

        self.csv_writer.writerow(
            {
                "mode": self.args.mode,
                "map_name": self.args.map_name,
                "trial_id": self.args.trial_id,
                "model_path": str(self.args.model_path),
                "t": elapsed,
                "x": float(self.current_position[0]),
                "y": float(self.current_position[1]),
                "yaw": self.current_yaw,
                "odom_speed": self.current_speed,
                "yaw_rate": self.current_yaw_rate,
                "raw_action_0": float(raw_action[0]),
                "raw_action_1": float(raw_action[1]),
                "steer": steering_angle,
                "commanded_speed": commanded_speed,
                "speed_limit": speed_limit,
                "min_lidar": min_lidar,
                "distance_from_start": distance_from_start,
                "distance_travelled": self.distance_travelled,
                "lap_complete": int(lap_complete),
                "collided": int(collided),
                "terminated": int(terminated),
                "termination_reason": termination_reason,
            }
        )
        self.log_file.flush()

        if terminated:
            self.terminated = True
            self.termination_reason = termination_reason
            self._stop_vehicle()
            self.get_logger().info(
                f"Trial finished reason={termination_reason} "
                f"time={elapsed:.2f}s distance={self.distance_travelled:.2f}m"
            )


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RuntimeBenchmark(args)

    try:
        while rclpy.ok() and not node.terminated:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._stop_vehicle()
    finally:
        node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
