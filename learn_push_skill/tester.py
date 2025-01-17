from stable_baselines3 import PPO
import gymnasium as gym
import os

from gymnasium.envs.registration import register
register(
    id="CustomPusher-v0",
    entry_point="environments.CustomPusherEnv:CustomPusherEnv",
    max_episode_steps=300,
)

model_path = os.path.abspath("models/default_robot_trained/best_pusher")
model = PPO.load(model_path)

env = gym.make("CustomPusher-v0", render_mode="human")

while True:
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, reward_terms = env.step(action)
        episode_over = terminated or truncated


