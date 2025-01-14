import multiprocessing

import gymnasium as gym
import numpy as np
import datetime

from rich.progress_bar import ProgressBar
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, ProgressBarCallback

from gymnasium.envs.registration import register

register(
    id="CustomPusher-v0",
    entry_point="CustomPusherEnv:CustomPusherEnv",
    max_episode_steps=300,
)

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    NUM_ENVS = 6
    # env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = gym.make("CustomPusher-v0")

    model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="models/")
    eval_callback = EvalCallback(env, best_model_save_path="models/", log_path="models/", eval_freq=100_000)
    progressbar_callback = ProgressBarCallback()

    callback = [checkpoint_callback, eval_callback, progressbar_callback, ]
    model.learn(total_timesteps=100_000_000, callback=callback)

    model_path = f"models/pusher_model-{datetime.datetime.now().strftime('%m.%d-%H:%M:%S')}"
    model.save(model_path)

    env.close()

