from stable_baselines3 import PPO
from ros_f110_ppo.ros_f110_env import RosF110Env

import numpy as np
import csv
import time

"""
Runs a trained PPO policy in the F1TENTH simulator and logs runtime behaviour.

The script records steering commands, speed outputs and minimum LiDAR distance
over time for quantitative evaluation on unseen maps.
"""

LOG_FILE_PATH = "/home/feras/sim_ws/src/ros_f110_ppo/logs/ppo_runtime.csv"
PPO_MODEL_PATH = "/home/feras/sim_ws/src/ros_f110_ppo/ppo_ros_f110.zip"


def main():
    environment = RosF110Env()

    policy = PPO.load(PPO_MODEL_PATH)
    print("PPO model loaded")

    observation = environment.reset()
    experiment_start_time = time.time()

    with open(LOG_FILE_PATH, "w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["t", "steer", "speed", "min_lidar"])

        while True:
            action, _ = policy.predict(observation, deterministic=True)

            steering_command = float(action[0])
            speed_command = float(action[1])

            minimum_lidar_distance = (
                np.min(environment.latest_scan.ranges)
                if environment.latest_scan is not None
                else np.nan
            )

            elapsed_time = time.time() - experiment_start_time
            csv_writer.writerow(
                [elapsed_time, steering_command, speed_command, minimum_lidar_distance]
            )

            observation, _, episode_done, _ = environment.step(action)

            if episode_done:
                observation = environment.reset()


if __name__ == "__main__":
    main()
