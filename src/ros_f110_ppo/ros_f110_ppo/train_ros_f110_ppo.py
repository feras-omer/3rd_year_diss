from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from ros_f110_ppo.ros_f110_env import RosF110Env

"""
This script continues training a PPO policy in the F1TENTH ROS2 simulator.

A previously trained PPO model is loaded and further trained on additional maps
to improve generalisation. Training statistics are logged using TensorBoard.
"""

# Path to the existing PPO checkpoint
PPO_MODEL_PATH = "/home/feras/sim_ws/src/ros_f110_ppo/ppo_ros_f110.zip"

# Directory used for TensorBoard logging
TENSORBOARD_LOG_DIR = "/home/feras/sim_ws/src/ros_f110_ppo/tb_logs/ppo"


def main():
    # Create the ROS-Gym environment
    environment = RosF110Env()

    # Configure TensorBoard and console logging
    logger = configure(TENSORBOARD_LOG_DIR, ["stdout", "tensorboard"])

    # Load the existing PPO model and attach the environment
    model = PPO.load(
        PPO_MODEL_PATH,
        env=environment,
        verbose=1,
    )

    model.set_logger(logger)

    # Continue training without resetting the timestep counter
    model.learn(
        total_timesteps=70_000,
        reset_num_timesteps=False,
    )

    # Overwrite the existing checkpoint with the updated policy
    model.save(PPO_MODEL_PATH)


if __name__ == "__main__":
    main()
