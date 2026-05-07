import csv
import math
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from ros_f110_ppo.diffusion_model import DiffusionPolicy

"""
Runs a diffusion-based policy in the F1TENTH simulator.

The runtime keeps the observation format aligned with the training dataset by
feeding back the previously published action. That alignment is important for
tight corners, where the policy needs temporal continuity to hold steering.
"""


MODEL_PATH = "/home/feras/sim_ws/src/ros_f110_ppo/models/diffusion_policy.pt"
LOG_FILE_PATH = "/home/feras/sim_ws/src/ros_f110_ppo/logs/diffusion_runtime.csv"


class DiffusionRunner(Node):
    def __init__(self):
        super().__init__("diffusion_runner")

        self.maximum_speed = 5.0
        self.maximum_steering_angle = 0.4
        self.maximum_lidar_range = 10.0
        self.lidar_stride = 10
        self.control_period = 0.1
        self.number_of_denoise_steps = 8
        self.forward_speed_gain = 3.2
        self.turn_speed_penalty = 0.65

        self.latest_scan = None
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.previous_action = np.array([0.0, self.maximum_speed], dtype=np.float32)

        self.create_subscription(LaserScan, "/scan", self._scan_callback, 20)
        self.create_subscription(
            Odometry, "/ego_racecar/odom", self._odom_callback, 50
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, "/drive", 10
        )

        self.policy, self.action_mean, self.action_std, self.outputs_are_normalised = (
            self._load_policy_checkpoint()
        )

        self.log_file = open(LOG_FILE_PATH, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(
            ["t", "steer", "speed", "pred_speed", "safety_speed", "min_lidar"]
        )
        self.start_time = time.time()

        self.timer = self.create_timer(self.control_period, self._control_step)
        self.get_logger().info("Diffusion runner started")

    def _load_policy_checkpoint(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            observation_dim = int(checkpoint.get("observation_dim", 112))
            action_dim = int(checkpoint.get("action_dim", 2))
            hidden_dim = int(checkpoint.get("hidden_dim", 256))
            policy = DiffusionPolicy(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            policy.load_state_dict(checkpoint["model_state_dict"])
            action_mean = checkpoint["action_mean"].float()
            action_std = checkpoint["action_std"].float()
            outputs_are_normalised = True
        else:
            policy = DiffusionPolicy(observation_dim=112, action_dim=2)
            policy.load_state_dict(checkpoint)
            action_mean = torch.zeros(2, dtype=torch.float32)
            action_std = torch.ones(2, dtype=torch.float32)
            outputs_are_normalised = False

        policy.eval()
        return policy, action_mean, action_std, outputs_are_normalised

    def _scan_callback(self, message: LaserScan):
        self.latest_scan = message

    def _odom_callback(self, message: Odometry):
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)
        self.current_yaw_rate = message.twist.twist.angular.z

    def _build_observation(self):
        lidar_ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.maximum_lidar_range,
            posinf=self.maximum_lidar_range,
            neginf=0.0,
        )

        lidar_downsampled = lidar_ranges[:: self.lidar_stride]
        lidar_normalised = np.clip(
            lidar_downsampled / self.maximum_lidar_range, 0.0, 1.0
        )

        speed_normalised = np.clip(
            self.current_speed / self.maximum_speed, 0.0, 1.0
        )
        yaw_rate_clipped = np.clip(self.current_yaw_rate, -1.0, 1.0)

        observation = np.concatenate(
            [
                lidar_normalised,
                [speed_normalised, yaw_rate_clipped],
                self.previous_action,
            ]
        ).astype(np.float32)

        return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    def _sample_diffusion_policy(self, observation):
        if self.outputs_are_normalised:
            action = torch.randn((1, 2), dtype=torch.float32)
        else:
            action = torch.tensor(
                self.previous_action, dtype=torch.float32
            ).unsqueeze(0)

        with torch.no_grad():
            for timestep_value in np.linspace(1.0, 0.0, self.number_of_denoise_steps):
                timestep = torch.full((1,), timestep_value, dtype=torch.float32)
                noise_estimate = self.policy(observation, action, timestep)
                action = action - noise_estimate / self.number_of_denoise_steps

        if self.outputs_are_normalised:
            action = action * self.action_std.unsqueeze(0) + self.action_mean.unsqueeze(0)

        action = action.squeeze(0).numpy()
        action[0] = np.clip(
            action[0], -self.maximum_steering_angle, self.maximum_steering_angle
        )
        action[1] = np.clip(action[1], 0.0, self.maximum_speed)
        return action

    def _compute_safety_speed(self, observation, steering_angle):
        forward_lidar = float(np.mean(observation[0, 52:56].numpy()))
        base_speed = self.forward_speed_gain * forward_lidar
        steering_ratio = abs(steering_angle) / self.maximum_steering_angle
        corner_penalty = 1.0 - self.turn_speed_penalty * steering_ratio
        return float(
            np.clip(base_speed * max(corner_penalty, 0.25), 0.8, self.maximum_speed)
        )

    def _control_step(self):
        if self.latest_scan is None:
            return

        observation = self._build_observation()
        predicted_action = self._sample_diffusion_policy(observation)

        steering_angle = float(predicted_action[0])
        predicted_speed = float(predicted_action[1])
        safety_speed = self._compute_safety_speed(observation, steering_angle)
        commanded_speed = min(max(predicted_speed, 0.8), safety_speed)

        minimum_lidar_distance = float(np.nanmin(self.latest_scan.ranges))
        elapsed_time = time.time() - self.start_time

        self.csv_writer.writerow(
            [
                elapsed_time,
                steering_angle,
                commanded_speed,
                predicted_speed,
                safety_speed,
                minimum_lidar_distance,
            ]
        )

        drive_command = AckermannDriveStamped()
        drive_command.drive.steering_angle = steering_angle
        drive_command.drive.speed = commanded_speed
        self.drive_publisher.publish(drive_command)

        self.previous_action[:] = [steering_angle, commanded_speed]


def main():
    rclpy.init()
    node = DiffusionRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
