import numpy as np
import mujoco
from stable_baselines3 import PPO
import gymnasium as gym
import torch
import os
import minari
import time

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv
from environments.XarmTableEnvironment import XarmTableEnv

from CustomPolicy import CustomPolicy


# --- ENVIRONMENT SETUP ---

non_robot_indices_in_observation = [0, 1, 2]
state_evaluator = StateEvaluator(len(non_robot_indices_in_observation))
state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/state_evaluator_02.18-19:46.pth"
state_evaluator.load_state_dict(torch.load(state_evaluator_path, map_location=torch.device('cpu')))

dataset = minari.load_dataset("xarm_push_3d_action_space_closer_1k-v0")
env = XarmTableEnv(control_option="ee_pos", render_mode="human")

env = InverseTrainerEnv(env, state_evaluator, dataset, non_robot_indices_in_obs=non_robot_indices_in_observation)

initial_model_path = "models/initial_policies/best_initial_policy_log_prob_02.27-03:16.pth"

initial_policy = CustomPolicy(env.observation_space, env.action_space)
pretrained_weights = torch.load(initial_model_path, map_location=torch.device('cpu'))
initial_policy.load_pretrained_weights(pretrained_weights)

# ----- MODEL SETUP -----

TEST_PRATRAINED = True
# TEST_PRATRAINED = False
if TEST_PRATRAINED:
    model = PPO(CustomPolicy, env=env, verbose=1, device="cpu")
    model.policy.load_state_dict(initial_policy.state_dict())

else:
    time = "02.21-01:54"
    model_dir_path = f"/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs/{time}"

    # OPTION = "latest"
    OPTION = "best"

    if OPTION == "latest":
        model_directory = os.listdir(model_dir_path)
        model_snapshots = [model for model in model_directory if model.startswith("rl_model")]

        max_i = 0
        max_step = 0
        for i in range(len(model_snapshots)):
            model_snapshots[i] = model_snapshots[i].split("_")
            step_count = int(model_snapshots[i][2])
            if step_count > max_step:
                max_step = step_count
                max_i = i
        model_path = os.path.abspath(f"{model_dir_path}/rl_model_{model_snapshots[max_i][2]}_steps.zip")

    elif OPTION == "best":
        model_path = os.path.abspath(f"{model_dir_path}/best_model.zip")

    model = PPO.load(model_path, env=env, device="cpu")

# ----- TESTING -----

model = env.env.model
data = env.env.data

while True:
    with torch.no_grad():
        episode_over = False
        i = 0
        initial_obs, _ = env.reset()
        initial_obs = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = initial_policy(initial_obs, deterministic=False)
        # action = list(dataset.iterate_episodes())[0].actions[-1]
        observation, _, _, _, _ = env.step(action)
        env.env.wait_until_ee_reaches_mocap()

        while not episode_over:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action, _, _ = initial_policy(observation, deterministic=False)
            action = action[0]
            observation, reward, terminated, truncated, reward_terms = env.step(action)

            episode_over = terminated or truncated or i > 1000 or np.linalg.norm(observation[:3] - [0, -1, 0]) > 1
            i += 1
