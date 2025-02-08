from stable_baselines3 import PPO
import gymnasium as gym
import torch
import os
import minari

from StateEvaluator import StateEvaluator

from gymnasium.envs.registration import register
register(
    id="InverseTrainerEnvironment-v0",
    entry_point="inverse_trainer.InverseTrainerEnv:InverseTrainerEnv",
    max_episode_steps=300,
)

model_dir_path = "/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_logs"

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
    model_path = os.path.abspath(f"{model_dir_path}rl_model_{model_snapshots[max_i][2]}_steps.zip")

elif OPTION == "best":
    model_path = os.path.abspath(f"{model_dir_path}/best_model.zip")

model = PPO.load(model_path)

state_evaluator_path = "/Users/toprak/InverseRL/inverse_trainer/models/state_evaluators/state_evaluator_02.05-17:45.pth"
state_evaluator = StateEvaluator(14)
state_evaluator.load_state_dict(torch.load(state_evaluator_path))

dataset = minari.load_dataset("xarm_push_10k-v0")
simulation_environment = dataset.recover_environment().unwrapped
simulation_environment.render_mode = "human"

env = gym.make(
"InverseTrainerEnvironment-v0",
    render_mode="human",
    state_evaluator=state_evaluator,
    dataset=dataset,
    env=simulation_environment
)

# env = simulation_environment
# env.render_mode = "human"

for _ in range(1):
    observation, _ = env.reset()
    episode_over = False
    i = 0
    while not episode_over:
        print(f"observation: {observation}")
        action, _ = model.predict(observation)
        print(f"action: {action}")
        observation, reward, terminated, truncated, reward_terms = env.step(action)
        episode_over = terminated or truncated or i > 300
        i += 1


