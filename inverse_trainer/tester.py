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

from CustomPPOPolicy import CustomPolicy


# --- ENVIRONMENT SETUP ---

non_robot_indices_in_observation = [0, 1, 2]
state_evaluator = StateEvaluator(len(non_robot_indices_in_observation))
state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/state_evaluator_03.05-16:50.pth"
state_evaluator.load_state_dict(torch.load(state_evaluator_path, map_location=torch.device('cpu')))

dataset = minari.load_dataset("xarm_push_4d_action_space_random_gripper_20-v0")
env = XarmTableEnv(control_option="ee_pos", render_mode="human")

env = InverseTrainerEnv(env, state_evaluator, dataset, non_robot_indices_in_obs=non_robot_indices_in_observation)

initial_model_path = "./models/initial_policies/best_initial_policy_log_prob_03.04-18:34.pth" # the one used in pulls_sometimes model
# initial_model_path = "./models/initial_policies/best_initial_policy_log_prob_03.05-16:57.pth"

initial_policy = CustomPolicy(env.observation_space, env.action_space)
pretrained_weights = torch.load(initial_model_path, map_location=torch.device('cpu'))
initial_policy.load_state_dict(pretrained_weights)

# ----- MODEL SETUP -----

model = PPO(CustomPolicy, env=env, verbose=1, device="cpu", use_sde=True)
model.policy.load_state_dict(initial_policy.state_dict())

pretrained_model = model
#--------------------

# model_dir_path = f"/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs/03.06-03:42"
model_dir_path = f"/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs/03.04-19:12_pulls_sometimes"

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
full_model = model

# model_option = "full_model"
model_option = "pretrained model"

if model_option == "full_model":
    model = full_model
else:
    model = pretrained_model

# ----- TESTING -----

# for rewarding as
from train_pure_PPO import PurePPOEnv
env = XarmTableEnv(control_option="ee_pos", render_mode="human")
env = PurePPOEnv(env)

pure_ppo_model = PPO.load("./models/pure_PPOs/03.07-20:35/best_model.zip")

# while True:
# for i, model_name in enumerate(["full_model", "pretrained_model", "pure_PPO"]):
# test_models = [full_model, pretrained_model, pure_ppo_model]
# model = test_models[i]

total_rewards = []
dist_rewards = []
ee_rewards = []
initial_distances = []
final_distances = []
for _ in range(1000):
    with torch.no_grad():

        episode_over = False
        i = 0
        initial_obs, _ = env.reset()
        initial_obs = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
        action, _ = model.predict(initial_obs)
        observation, reward, _, _, info = env.step(action)
        # env.env.wait_until_ee_reaches_mocap()

        total_reward = reward
        total_dist_reward = info["obj_to_base_rew"]
        total_ee_reward = info["ee_to_obj_rew"]
        while not episode_over:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action, _ = model.predict(observation)
            # action, _, _ = initial_policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_dist_reward += info["obj_to_base_rew"]
            total_ee_reward += info["ee_to_obj_rew"]

            if np.linalg.norm(reward - (info["obj_to_base_rew"] + info["ee_to_obj_rew"])) > 1e-5:
                print("There's a problem with the reward calculation")
                print(f"reward: {reward}, obj_to_base_rew: {info['obj_to_base_rew']}, ee_to_obj_rew: {info['ee_to_obj_rew']}")
                print()

            episode_over = terminated or truncated or i > 500
            i += 1

        total_rewards.append(total_reward)
        dist_rewards.append(total_dist_reward)
        ee_rewards.append(total_ee_reward)

        initial_pos = initial_obs[0, :3].numpy()
        initial_distance = np.linalg.norm(initial_pos - [0, -1, 0.04])
        initial_distances.append(initial_distance)

        final_pos = observation[:3]
        final_distance = np.linalg.norm(final_pos - [0, -1, 0.04])
        final_distances.append(final_distance)

        print(f"initial object pos: {initial_obs[0, :3].numpy()}, distance: {np.linalg.norm(initial_obs[0, :3].numpy() - [0, -1, 0.04])}")
        print(f"final object pos: {observation[:3]}, distance: {np.linalg.norm(observation[:3] - [0, -1, 0.04])}")
        print()

total_rewards = np.array(total_rewards)
dist_rewards = np.array(dist_rewards)
ee_rewards = np.array(ee_rewards)
initial_distances = np.array(initial_distances)
final_distances = np.array(final_distances)

# save_dir = "./models/final_reward_logs/"
# save_path = save_dir +  f"03.08_{model_name}_500_steps_distance_and_ee_reward.npz"
#
# np.savez(save_path, total_rewards=total_rewards, dist_rewards=dist_rewards, ee_rewards=ee_rewards, initial_distances=initial_distances, final_distances=final_distances)