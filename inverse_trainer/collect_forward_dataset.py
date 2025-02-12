from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

import minari
from minari import DataCollector

from gymnasium.envs.registration import register

register(
    id="XarmPushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=100_000,
)


def collect_forward_demo(dataset_id, num_demos):
    forward_model_path = "../learn_push_skill/models/adjusted_gripper/best_model.zip"
    forward_model = PPO.load(forward_model_path)

    env = gym.make("XarmPushTrainer-v0")
    env = DataCollector(env, record_infos=False)

    successful_demo_count = 0
    while successful_demo_count < num_demos:
        print(f"collecting {successful_demo_count}th demo")

        observation, _ = env.reset()
        print(f"initial distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")

        episode_over = False
        successful = False
        step_count = 0
        while not episode_over:
            action, _ = forward_model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_over = terminated or truncated

            if np.linalg.norm(observation[0:2] - [0, -1]) > 0.9:
                episode_over = True
                successful = True

        if successful:
            successful_demo_count += 1
            print(
                f"demo successful. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")
        else:
            print(
                f"demo failed. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")

        print()

    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="PPO_xarm_pusher",
        author="Bora Toprak Temir"
    )


if __name__ == "__main__":
    dataset_id = "xarm_push_only_successful_50-v0"
    num_demos = 50
    collect_forward_demo(dataset_id, num_demos)
