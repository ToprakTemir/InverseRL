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
    id="PushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=300,
)

def make_env():
    return gym.make("PushTrainer-v0")

if __name__ == "__main__":
    num_envs = multiprocessing.cpu_count()
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./ppo_logs/")

    path = "models/adjusted_gripper_v2/"
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=path)
    eval_callback = EvalCallback(env, best_model_save_path=path, log_path=path, eval_freq=10_000)
    progressbar_callback = ProgressBarCallback()

    callback = [checkpoint_callback, eval_callback, progressbar_callback]
    model.learn(total_timesteps=50_000_000, callback=callback)

    model_path = f"models/pusher_model-{datetime.datetime.now().strftime('%m.%d-%H:%M:%S')}"
    model.save(model_path)

    env.close()

