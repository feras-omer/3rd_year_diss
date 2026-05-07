# Diffusion Policy for F1TENTH

This repository packages a dissertation project around diffusion-policy control for F1TENTH, alongside PPO baselines, simulator integration, and evaluation scripts.

The codebase is intentionally left close to the working project layout used during development. The source files are not reorganized here; instead, this repository documents how to use the existing components for:

- F1TENTH physical-car deployment
- F1TENTH ROS simulator runs
- Offline diffusion training
- Old-vs-new diffusion evaluation

## Repository Overview

| Path | Purpose |
| --- | --- |
| `src/f1tenth_gym_ros/` | ROS 2 bridge for `f1tenth_gym`, maps, launch files, Docker support |
| `src/ros_f110_ppo/` | ROS 2 package containing PPO runtime/training and the main diffusion runtime used for hardware |
| `diffusion_f110/` | Standalone PyTorch diffusion training code and datasets |
| `diffusion_runner_for_simulation/` | Alternate diffusion runner for simulator topic naming |
| `analysis/` | Benchmarking and comparison scripts for checkpoints, TensorBoard logs, and runtime trials |
| `old model/` | Legacy diffusion checkpoint used for comparison |
| `results/` | Generated figures, CSV summaries, and markdown outputs |

## Supported Workflows

### 1. Physical Car Deployment

For physical-car use, the intended runtime is the `ros_f110_ppo` package version of the diffusion controller:

- Runner: `src/ros_f110_ppo/ros_f110_ppo/diffusion_runner.py`
- Model: `src/ros_f110_ppo/models/diffusion_policy.pt`

This runner is configured for:

- `LaserScan` on `/scan`
- `Odometry` on `/odom`
- drive commands on `/drive`

If you deploy this onto a hardware workspace, keep the package structure intact so the model path and ROS package layout remain valid.

### 2. Simulator Deployment

For simulation, use the same `ros_f110_ppo` package but swap the diffusion runner file with the simulator-specific version:

- Source package runner to replace:
  `src/ros_f110_ppo/ros_f110_ppo/diffusion_runner.py`
- Simulator variant:
  `diffusion_runner_for_simulation/diffusion_runner.py`

The simulator runner differs mainly in topic expectations, especially odometry:

- simulator odometry topic: `/ego_racecar/odom`
- simulator scan topic: `/scan`
- simulator drive topic: `/drive`

The simulator itself is provided through `src/f1tenth_gym_ros/`.

### 3. Offline Diffusion Training

The standalone diffusion trainer lives in `diffusion_f110/` and is separate from the ROS 2 package:

- dataset loader: `diffusion_f110/dataset.py`
- model definition: `diffusion_f110/diffusion_model.py`
- trainer: `diffusion_f110/train_diffusion.py`
- training data: `diffusion_f110/data/*.npz`

### 4. Evaluation and Comparison

The `analysis/` directory contains scripts for:

- offline checkpoint comparison
- TensorBoard log comparison
- runtime benchmarking
- runtime summary plots and tables

This is the part of the repository intended for comparing the legacy diffusion checkpoint against the model currently shipped under `src/ros_f110_ppo/models/`.

## Software Requirements

This project was developed around the following environment:

- Ubuntu 20.04
- ROS 2 Foxy
- Python 3.8

### Core ROS 2 Dependencies

The project uses these ROS 2 packages and message types across the simulator and controller packages:

- `rclpy`
- `sensor_msgs`
- `nav_msgs`
- `ackermann_msgs`
- `geometry_msgs`
- `tf2_ros`
- `launch`
- `launch_ros`
- `joint_state_publisher`
- `robot_state_publisher`
- `xacro`
- `nav2_map_server`
- `nav2_lifecycle_manager`
- `teleop_twist_keyboard`

### Python Dependencies

Across training, runtime, and analysis, the code imports:

- `numpy`
- `gym`
- `torch`
- `stable_baselines3`
- `matplotlib`
- `tensorboard`
- `tqdm`
- `transforms3d`

### Simulator-Specific Dependencies

To use the simulator stack you also need:

- `f1tenth_gym`
- optionally Docker and Docker Compose
- optionally `rocker` for GUI forwarding
- optionally NVIDIA container support for GPU-backed container workflows

## Recommended Workspace Layout

Because several scripts use hard-coded paths from the original development setup, the safest ROS workspace layout is:

```text
~/sim_ws/
  src/
    f1tenth_gym_ros/
    ros_f110_ppo/
```

In other words, copy the package directories from this repository into `~/sim_ws/src/` rather than nesting this whole repository one level deeper inside another package folder.

## Simulator Setup

1. Install ROS 2 Foxy on Ubuntu 20.04.
2. Install `f1tenth_gym`.
3. Copy these directories into `~/sim_ws/src/`:
   - `src/f1tenth_gym_ros/`
   - `src/ros_f110_ppo/`
4. Replace `~/sim_ws/src/ros_f110_ppo/ros_f110_ppo/diffusion_runner.py` with `diffusion_runner_for_simulation/diffusion_runner.py`.
5. Update the map path in `src/f1tenth_gym_ros/config/sim.yaml` if needed.
6. Install ROS dependencies with `rosdep`.
7. Build the workspace with `colcon build`.
8. Source the workspace and launch the simulator:

```bash
source /opt/ros/foxy/setup.bash
source ~/sim_ws/install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

9. In another terminal, run the diffusion controller:

```bash
source /opt/ros/foxy/setup.bash
source ~/sim_ws/install/local_setup.bash
ros2 run ros_f110_ppo run_diffusion_f110
```

## Physical Car Setup

1. Install ROS 2 and the required message packages on the car or onboard computer.
2. Copy `src/ros_f110_ppo/` into `~/sim_ws/src/ros_f110_ppo`.
3. Keep the package version of `ros_f110_ppo/ros_f110_ppo/diffusion_runner.py` unchanged for hardware use.
4. Ensure the model file exists at `src/ros_f110_ppo/models/diffusion_policy.pt` inside that package directory.
5. Build the workspace with `colcon build`.
6. Bring up the physical-car driver stack first. In the official RoboRacer `f1tenth_system` repository, the onboard bringup launch lives at `f1tenth_stack/launch/bringup_launch.py`, so the corresponding launch command is:

```bash
source /opt/ros/foxy/setup.bash
source ~/sim_ws/install/local_setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

This bringup stack is the part that starts the low-level car interfaces such as joystick teleop, VESC, odometry, LiDAR, and the Ackermann mux before your controller node is launched.

7. After the car bringup is running, source the workspace and run the diffusion controller:

```bash
source /opt/ros/foxy/setup.bash
source ~/sim_ws/install/local_setup.bash
ros2 run ros_f110_ppo run_diffusion_f110
```

## PPO Baseline

The PPO components are in `src/ros_f110_ppo/ros_f110_ppo/`:

- `train_ros_f110_ppo.py` continues PPO training
- `run_ros_f110_ppo.py` runs the trained PPO policy in the simulator
- `ros_f110_env.py` exposes ROS observations/actions as a Gym environment

## Diffusion Training

Train or fine-tune the diffusion model from the repository root with:

```bash
python3 diffusion_f110/train_diffusion.py
```

The main artifacts are:

- checkpoint output: `diffusion_f110/models/diffusion_policy.pt`
- TensorBoard logs: `diffusion_f110/tb_logs/diffusion/`

## Analysis and Model Comparison

### Offline Checkpoint Comparison

Use explicit paths when comparing the legacy and current checkpoints:

```bash
python3 analysis/compare_diffusion_checkpoints.py \
  --dataset diffusion_f110/data/expert_merged.npz \
  --old-checkpoint "old model/diffusion_policy.pt" \
  --new-checkpoint src/ros_f110_ppo/models/diffusion_policy.pt \
  --output-dir results/offline
```

### TensorBoard Comparison

```bash
python3 analysis/compare_diffusion_tb_logs.py \
  --new-log-dir diffusion_f110/tb_logs/diffusion \
  --output-dir results/training
```

### Runtime Benchmarking

`analysis/benchmark_diffusion_runtime.py` assumes the simulator is already running and can benchmark both legacy and current diffusion controllers against a common CSV schema.

## Notes

- Several scripts preserve original absolute-path assumptions from the development workspace.
- The simulator and hardware diffusion runners are both kept because they target different ROS topic layouts.
- The analysis code is intended for comparing the legacy checkpoint in `old model/` with the current checkpoint under `src/ros_f110_ppo/models/`.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
