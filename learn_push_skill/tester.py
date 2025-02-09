from stable_baselines3 import PPO
import gymnasium as gym
import os
import numpy as np

from gymnasium.envs.registration import register

register(
    id="XarmPushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=1000,
)

# OPTION = "latest"
OPTION = "best"

model_dir_path = "/Users/toprak/InverseRL/learn_push_skill/models/adjusted_gripper"

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

print(f"testing with {OPTION} model")
print(model_path)
model = PPO.load(model_path)

env = gym.make("XarmPushTrainer-v0", render_mode="human")
# model = PPO("MlpPolicy", env, verbose=1)

while True:
    observation, _ = env.reset()
    print(f"initial distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")
    episode_over = False
    i = 0
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        if np.linalg.norm(observation[0:2] - [0, -1]) > 0.9:
            episode_over = True
            print(f"episode over, distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")
            break
        if i % 1000 == 0:
            print(f"distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")

    print()