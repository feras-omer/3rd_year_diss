import rclpy
from rclpy.node import Node
import numpy as np
import os

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

"""
Logs expert trajectories from the simulator for offline diffusion training.

The logger records observations, actions and timestamps at a fixed rate and
stores them in a compressed NumPy file.
"""


class F110DataLogger(Node):
    def __init__(self):
        super().__init__("f110_data_logger")

        self.declare_parameter(
            "output_directory", "/home/feras/sim_ws/src/ros_f110_ppo/logs"
        )
        self.declare_parameter("filename", "expert_run.npz")
        self.declare_parameter("logging_rate_hz", 10.0)
        self.declare_parameter("maximum_lidar_range", 10.0)
        self.declare_parameter("lidar_stride", 10)
        self.declare_parameter("maximum_speed", 5.0)

        self.output_directory = self.get_parameter("output_directory").value
        self.filename = self.get_parameter("filename").value
        self.logging_rate = float(self.get_parameter("logging_rate_hz").value)
        self.maximum_lidar_range = float(
            self.get_parameter("maximum_lidar_range").value
        )
        self.lidar_stride = int(self.get_parameter("lidar_stride").value)
        self.maximum_speed = float(self.get_parameter("maximum_speed").value)

        os.makedirs(self.output_directory, exist_ok=True)
        self.output_path = os.path.join(self.output_directory, self.filename)

        self.latest_scan = None
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.current_action = np.zeros(2, dtype=np.float32)

        self.observation_buffer = []
        self.action_buffer = []
        self.timestamp_buffer = []

        self.create_subscription(LaserScan, "/scan", self._scan_callback, 20)
        self.create_subscription(Odometry, "/odom", self._odom_callback, 50)
        self.create_subscription(
            AckermannDriveStamped, "/drive", self._drive_callback, 20
        )

        self.timer = self.create_timer(1.0 / self.logging_rate, self._log_step)

        self.get_logger().info(f"Logging expert data to {self.output_path}")
        self.get_logger().info("Press Ctrl+C to stop and save the dataset")

    def _scan_callback(self, message: LaserScan):
        self.latest_scan = message

    def _odom_callback(self, message: Odometry):
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self.current_speed = float(np.hypot(vx, vy))
        self.current_yaw_rate = float(message.twist.twist.angular.z)

    def _drive_callback(self, message: AckermannDriveStamped):
        self.previous_action = self.current_action.copy()
        self.current_action = np.array(
            [message.drive.steering_angle, message.drive.speed], dtype=np.float32
        )

    def _build_observation(self):
        lidar_ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.maximum_lidar_range,
            posinf=self.maximum_lidar_range,
            neginf=0.0,
        )

        lidar_downsampled = lidar_ranges[::self.lidar_stride]
        lidar_normalised = np.clip(
            lidar_downsampled / self.maximum_lidar_range, 0.0, 1.0
        )

        speed_normalised = np.clip(
            self.current_speed / self.maximum_speed, 0.0, 1.0
        )
        yaw_rate_clipped = np.clip(self.current_yaw_rate, -1.0, 1.0)

        observation = np.concatenate(
            [lidar_normalised, [speed_normalised, yaw_rate_clipped], self.previous_action]
        ).astype(np.float32)

        return observation

    def _log_step(self):
        if self.latest_scan is None:
            return

        observation = self._build_observation()
        timestamp = self.get_clock().now().nanoseconds

        self.observation_buffer.append(observation)
        self.action_buffer.append(self.current_action.copy())
        self.timestamp_buffer.append(timestamp)

        if len(self.observation_buffer) % 500 == 0:
            self.get_logger().info(f"Collected {len(self.observation_buffer)} samples")

    def save(self):
        observations = (
            np.stack(self.observation_buffer)
            if self.observation_buffer
            else np.zeros((0,), dtype=np.float32)
        )
        actions = (
            np.stack(self.action_buffer)
            if self.action_buffer
            else np.zeros((0,), dtype=np.float32)
        )
        timestamps = np.array(self.timestamp_buffer, dtype=np.int64)

        np.savez_compressed(
            self.output_path, obs=observations, act=actions, t=timestamps
        )

        self.get_logger().info(
            f"Saved {len(observations)} samples to {self.output_path}"
        )


def main():
    rclpy.init()
    node = F110DataLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
