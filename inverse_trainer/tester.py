from stable_baselines3 import PPO
import gymnasium as gym
import torch
import os
import minari

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv

from InitialPolicy import InitialPolicy
# from InitialPPO import InitialPolicy

from gymnasium.envs.registration import register
register(
    id="XarmPushEnv-v0",
    entry_point="environments.XarmTableEnvironment:XarmTableEnv",
    max_episode_steps=300,
)

model_dir_path = "/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs/02.11-13:25"

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

non_robot_indices_in_observation = [0, 1, 2]

state_evaluator = StateEvaluator(len(non_robot_indices_in_observation))
state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/best_state_evaluator_02.10-00:41.pth"
state_evaluator.load_state_dict(torch.load(state_evaluator_path, map_location=torch.device('cpu')))

dataset = minari.load_dataset("xarm_push_only_successful_1k-v0")
env = dataset.recover_environment().unwrapped
env = InverseTrainerEnv(state_evaluator, dataset, env, non_robot_indices_in_obs=non_robot_indices_in_observation)
env.env.render_mode = "human"



# model = PPO.load(model_path)

initial_model_path = "/Users/toprak/InverseRL/inverse_trainer/models/initial_policies/best_initial_policy_02.13-02:24.pth"

initial_policy = InitialPolicy(env.observation_space.shape[0], env.action_space.shape[0])
initial_policy.load_state_dict(torch.load(initial_model_path, map_location=torch.device('cpu')))
initial_policy_weights = initial_policy.state_dict()

# model = PPO(InitialPolicy, env=env, verbose=1, device="cpu")
# ppo_actor_network = model.policy.mlp_extractor.policy_net
# ppo_actor_network.load_state_dict(initial_policy_weights)

model = initial_policy

while True:
    observation, _ = env.reset()
    episode_over = False
    i = 0
    while not episode_over:
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32)
            action = model.forward(observation)
            observation, reward, terminated, truncated, reward_terms = env.step(action)

            episode_over = terminated or truncated or i > 300
            i += 1


