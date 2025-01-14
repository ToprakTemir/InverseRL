from stable_baselines3 import PPO
import gymnasium as gym

from gymnasium.envs.registration import register
register(
    id="CustomPuller-v0",
    entry_point="environments.CustomPullerEnv:CustomPusherEnv",
    max_episode_steps=300,
)

# model_path = f"./model_logs/pusher_model-01.14-01:45:01"
model_path = "models/default_robot_trained/best_model.zip"
model = PPO.load(model_path)

env = gym.make("CustomPuller-v0", render_mode="human")

while True:
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, reward_terms = env.step(action)
        episode_over = terminated or truncated


