from stable_baselines3 import PPO
import gymnasium as gym
import os

from gymnasium.envs.registration import register

from environments.XarmTableEnvironment import XarmTableEnv

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
    entry_point="environments.PushTrainerEnv:PushTrainerEnv",
    max_episode_steps=300,
)

model_path = os.path.abspath("models/with_force_penalty/best_model.zip")
# model_path = os.path.abspath("models/rl_model_3600000_steps.zip")
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


