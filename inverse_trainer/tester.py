from stable_baselines3 import PPO
import gymnasium as gym
import torch
import os
import minari

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv

from InitialPPO import CustomPolicy

from gymnasium.envs.registration import register


# --- ENVIRONMENT SETUP ---

register(
    id="XarmPushEnv-v0",
    entry_point="environments.XarmTableEnvironment:XarmTableEnv",
    max_episode_steps=300,
)

non_robot_indices_in_observation = [0, 1, 2]

state_evaluator = StateEvaluator(len(non_robot_indices_in_observation))
state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/state_evaluator_02.18-19:46.pth"
state_evaluator.load_state_dict(torch.load(state_evaluator_path, map_location=torch.device('cpu')))

dataset = minari.load_dataset("xarm_push_directly_forward_50-v0")
env = dataset.recover_environment().unwrapped
print(env)
env = InverseTrainerEnv(state_evaluator, dataset, env, non_robot_indices_in_obs=non_robot_indices_in_observation)
env.env.render_mode = "human"


# ----- MODEL SETUP -----

TEST_PRATRAINED = True
if TEST_PRATRAINED:
    initial_model_path = "models/initial_policies/best_initial_policy_02.18-20:19.pth"

    initial_policy = CustomPolicy(env.observation_space, env.action_space)

    pretrained_weights = torch.load(initial_model_path, map_location=torch.device('cpu'))
    initial_policy.load_pretrained_weights(pretrained_weights)

    # model = PPO(CustomPolicy, env=env, verbose=1, device="cpu")
    # model.policy.load_state_dict(initial_policy.state_dict())

else:
    time = "02.17-22:39"
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

while True:
    observation, _ = env.reset()
    episode_over = False
    i = 0
    while not episode_over:
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action, _, _ = initial_policy(observation, deterministic=True)
            action = action[0]
            print(action)
            observation, reward, terminated, truncated, reward_terms = env.step(action)

            episode_over = terminated or truncated or i > 300
            i += 1
