from stable_baselines3 import PPO
import gymnasium as gym

import minari
from minari import DataCollector

from gymnasium.envs.registration import register
register(
    id="XarmPushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=300,
)

def collect_forward_demo(dataset_id, num_demos):
    forward_model_path = "../learn_push_skill/models/adjusted_gripper/best_model.zip"
    forward_model = PPO.load(forward_model_path)

    env = gym.make("XarmPushTrainer-v0")
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
        algorithm_name="PPO_xarm_pusher",
        author="Bora Toprak Temir"
    )

if __name__ == "__main__":
    dataset_id = "xarm_push_10k-v0"
    num_demos = 10_000
    collect_forward_demo(dataset_id, num_demos)

