from stable_baselines3 import PPO
import gymnasium as gym

import minari
from minari import DataCollector

from gymnasium.envs.registration import register
register(
    id="CustomPusher-v0",
    entry_point="environments.CustomPusherEnv:CustomPusherEnv",
    max_episode_steps=300,
)


def collect_forward_demo(dataset_id, num_demos):
    forward_model_path = "../learn_push_skill/models/default_robot_trained/best_pusher.zip"
    forward_model = PPO.load(forward_model_path)

    env = gym.make("CustomPusher-v0")
    env = DataCollector(env, record_infos=False)

    for i in range(num_demos):
        print(f"collecting {i}th demo")
        observation, _ = env.reset()
        episode_over = False
        while not episode_over:
            action, _ = forward_model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated


    env.create_dataset(
        dataset_id = dataset_id,
        algorithm_name="PPO_pusher",
        author="Bora Toprak Temir",
        description="Standard push forward demo",
    )

if __name__ == "__main__":
    dataset_id = "pusher_demo_R08_large-v0"
    num_demos = 10000
    collect_forward_demo(dataset_id, num_demos)

