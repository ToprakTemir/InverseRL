from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

import minari
from minari import DataCollector

from gymnasium.envs.registration import register

from environments.XarmTableEnvironment import XarmTableEnv
from PushDemonstratorEnv import PushDemonstratorEnv


register(
    id="XarmPushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=100_000,
)


def collect_demo_by_PPO(dataset_id, num_demos):
    forward_model_path = "models/adjusted_gripper/best_model.zip"
    forward_model = PPO.load(forward_model_path)

    env = gym.make("XarmPushTrainer-v0")
    env = DataCollector(env, record_infos=False)

    successful_demo_count = 0
    while successful_demo_count < num_demos:
        print(f"collecting {successful_demo_count}th demo")

        observation, _ = env.reset()
        print(f"initial distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")

        episode_over = False
        successful = False
        step_count = 0
        while not episode_over:
            action, _ = forward_model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_over = terminated or truncated

            if np.linalg.norm(observation[0:2] - [0, -1]) > 0.9:
                episode_over = True
                successful = True

        if successful:
            successful_demo_count += 1
            print(
                f"demo successful. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")
        else:
            print(
                f"demo failed. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")

        print()

    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="PPO_xarm_pusher",
        author="Bora Toprak Temir"
    )


def generate_push_trajectory(env, speed=0.002, pre_push_offset=0.1, push_distance=1, table_z=-0.03):
    """
    Generates a trajectory (list of action points) for pushing an object away,
    moving at a fixed speed (distance per step).

    The trajectory has two phases:
      1. Approach Phase: Moves from the current end-effector position to a pre-push position
         located behind the object (relative to the push direction).
      2. Push Phase: Moves along the push direction for a specified push distance.

    Each action is an 8D vector:
       [target_x, target_y, target_z, target_quat_x, target_quat_y, target_quat_z, target_quat_w, gripper_closeness]

    Note: We use a fixed orientation (identity)

    Parameters:
      env: The environment instance (assumed to have attributes 'data' and method '_get_obs()').
      speed: The distance to move per step (default 0.01 m per step).
      pre_push_offset: The distance behind the object for the pre-push position.
      push_distance: The distance to push the object.
      table_z: The table's z-coordinate (default 0.0); the end-effector is kept just above it.

    Returns:
      traj_actions: A list of action points (numpy arrays of shape (8,)).
    """
    # Ensure the end-effector stays just above the table.
    desired_z = table_z + 0.000

    # Get the current end-effector position from the mocap (flatten if necessary)
    current_pos = env.data.mocap_pos.copy()
    if current_pos.ndim > 1:
        current_pos = current_pos[0]
    current_pos[2] = desired_z

    # Get the object's position from the observation.
    obs = env._get_obs()
    object_pos = obs[:3].copy()
    object_pos[2] = desired_z

    # Define the robot base position (assumed fixed; adjust if needed)
    robot_base = np.array([0, -1, desired_z])

    # Compute the push direction (from robot base to object, projected on the table)
    push_direction = object_pos - robot_base
    push_direction[2] = 0  # ignore vertical component
    norm = np.linalg.norm(push_direction)
    push_direction = push_direction / norm

    # Compute the pre-push target (a point behind the object along the push direction)
    pre_push_target = object_pos - push_direction * pre_push_offset
    pre_push_target[2] = desired_z

    d0 = np.linalg.norm(object_pos - robot_base)
    robot_reach = 1.17
    push_displacement = max(0, robot_reach - d0)
    final_target = pre_push_target + push_direction * push_displacement
    final_target[2] = desired_z

    # Generate the approach trajectory at fixed speed.
    approach_vec = pre_push_target - current_pos
    approach_distance = np.linalg.norm(approach_vec)
    steps_approach = int(np.ceil(approach_distance / speed))
    traj_positions_approach = [
        current_pos + approach_vec * (i / steps_approach)
        for i in range(1, steps_approach + 1)
    ]

    # Generate the push trajectory at fixed speed.
    push_vec = final_target - pre_push_target
    push_distance_actual = np.linalg.norm(push_vec)
    steps_push = int(np.ceil(push_distance_actual / speed))
    traj_positions_push = [
        pre_push_target + push_vec * (i / steps_push)
        for i in range(1, steps_push + 1)
    ]

    # Combine both trajectories.
    traj_positions = traj_positions_approach + traj_positions_push

    # Use a fixed quaternion (identity) so that orientation is not controlled.
    fixed_quat = np.array([0, 0, 0, 1])
    gripper_closeness = 0  # Modify if necessary

    # Assemble the full 8D action for each trajectory point.
    traj_actions = [
        np.concatenate([pos, fixed_quat, [gripper_closeness]])
        for pos in traj_positions
    ]

    return traj_actions

def collect_demo_by_generating_push_trajectory(dataset_id, num_demos):
    env = XarmTableEnv()
    env = PushDemonstratorEnv(env)
    env = DataCollector(env, record_infos=False)

    successful_demo_count = 0
    while successful_demo_count < num_demos:
        print(f"collecting {successful_demo_count}th demo")
        observation, _ = env.reset()
        print(f"initial distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")

        traj = generate_push_trajectory(env.env)
        step_count = 0
        successful = False
        for action in traj:
            observation, _, _, _, _ = env.step(action)
            step_count += 1


        print(f"final distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")
        if np.linalg.norm(observation[0:2] - [0, -1]) > 0.9:
            successful = True

        if successful:
            successful_demo_count += 1
            print(f"demo successful. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")
        else:
            print(f"demo failed. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")

        print()

    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="push_trajectory_generator",
        author="Bora Toprak Temir"
    )


if __name__ == "__main__":
    dataset_id = "xarm_push_only_successful_50-v0"
    num_demos = 50
    collect_demo_by_generating_push_trajectory(dataset_id, num_demos)