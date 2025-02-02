from stable_baselines3 import PPO
import gymnasium as gym
import os

from gymnasium.envs.registration import register

register(
    id="CustomPusher-v0",
    entry_point="environments.CustomPusherEnv:CustomPusherEnv",
    max_episode_steps=300,
)
register(
    id="XarmEnv-v0",
    entry_point="environments.XarmTableEnvironment:XarmTableEnv",
    max_episode_steps=300,
)
register(
    id="PushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=300,
)

OPTION = "latest"
# OPTION = "best"

if OPTION == "latest":
    model_directory = os.listdir("/Users/toprak/InverseRL/learn_push_skill/models/force_penalty_v4/")
    model_directory = sorted(model_directory)
    model_path = os.path.abspath(f"/Users/toprak/InverseRL/learn_push_skill/models/force_penalty_v4/{model_directory[-1]}")
    print(model_path)
elif OPTION == "best":
    model_path = os.path.abspath("models/force_penalty_v4/best_model.zip")

model = PPO.load(model_path)

env = gym.make("PushTrainer-v0", render_mode="human")
# model = PPO("MlpPolicy", env, verbose=1)

while True:
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated


