import gym
import numpy as np
import time
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class RosF110Env(gym.Env):
    """
    Gym-style environment that interfaces a PPO agent with the F1TENTH ROS2 simulator.

    The environment receives LiDAR and odometry data from ROS topics and publishes
    Ackermann steering commands. It is designed for online reinforcement learning
    using Stable Baselines3.
    """

    def __init__(self):
        super().__init__()

        rclpy.init(args=None)
        self.node = rclpy.create_node("ros_f110_env")

        # Environment parameters
        self.maximum_speed = 5.0
        self.maximum_lidar_range = 10.0
        self.lidar_downsample_stride = 10
        self.control_timestep = 0.05

        # Action space: steering (normalised) and speed
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, self.maximum_speed], dtype=np.float32),
        )

        # Observation space: downsampled LiDAR + speed + yaw rate
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(110,),
            dtype=np.float32,
        )

        # Internal state
        self.latest_scan = None
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self.has_collided = False

        # ROS interfaces
        self.node.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.node.create_subscription(
            Odometry, "/ego_racecar/odom", self._odom_callback, 10
        )
        self.drive_publisher = self.node.create_publisher(
            AckermannDriveStamped, "/drive", 10
        )

    def _scan_callback(self, message: LaserScan):
        self.latest_scan = message
        if np.min(message.ranges) < 0.25:
            self.has_collided = True

    def _odom_callback(self, message: Odometry):
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self.current_speed = np.hypot(vx, vy)
        self.current_yaw_rate = message.twist.twist.angular.z

    def reset(self):
        self.has_collided = False
        self.latest_scan = None

        # Wait until a LiDAR scan is received
        while self.latest_scan is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        return self._build_observation()

    def step(self, action):
        steering_command, _ = action

        drive_command = AckermannDriveStamped()
        drive_command.drive.steering_angle = float(steering_command * 0.4)
        drive_command.drive.speed = self.maximum_speed
        self.drive_publisher.publish(drive_command)

        start_time = time.time()
        while time.time() - start_time < self.control_timestep:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        observation = self._build_observation()
        reward = self._compute_reward(observation, action)
        episode_done = self.has_collided

        if episode_done:
            stop_command = AckermannDriveStamped()
            stop_command.drive.steering_angle = 0.0
            stop_command.drive.speed = 0.0
            self.drive_publisher.publish(stop_command)

        return observation, reward, episode_done, {}

    def _build_observation(self):
        lidar_ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.maximum_lidar_range,
            posinf=self.maximum_lidar_range,
            neginf=0.0,
        )

        lidar_downsampled = lidar_ranges[::self.lidar_downsample_stride]
        lidar_normalised = lidar_downsampled / self.maximum_lidar_range

        speed_normalised = np.clip(
            self.current_speed / self.maximum_speed, 0.0, 1.0
        )
        yaw_rate_clipped = np.clip(self.current_yaw_rate, -1.0, 1.0)

        observation = np.concatenate(
            [lidar_normalised, [speed_normalised, yaw_rate_clipped]]
        ).astype(np.float32)

        return observation

    def _compute_reward(self, observation, action):
        lidar_readings = observation[:-2]
        speed = observation[-2]
        steering = action[0]

        front_lidar = np.mean(
            lidar_readings[len(lidar_readings) // 2 - 2 : len(lidar_readings) // 2 + 2]
        )

        reward = (
            speed * front_lidar
            - 0.05 * abs(steering)
            - 0.05 * abs(self.current_yaw_rate)
        )

        turning_penalty = abs(self.current_yaw_rate) * speed
        reward -= 0.2 * turning_penalty

        if self.has_collided:
            reward -= 30.0

        return reward
