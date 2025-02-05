from stable_baselines3 import PPO
import gymnasium as gym
import os
import numpy as np

from gymnasium.envs.registration import register

register(
    id="XarmEnv-v0",
    entry_point="environments.XarmTableEnvironment:XarmTableEnv",
    max_episode_steps=300,
)
register(
    id="XarmPushTrainer-v0",
    entry_point="environments.XarmPushTrainerEnv:PushTrainerEnv",
    max_episode_steps=300,
)


env = gym.make("XarmPushTrainer-v0", render_mode="human")
print("action space shape:" + str(env.action_space.shape))

obs = env.reset()
joint_angles = np.random.random(7)

i = 0
i_inc = 1
while True:
    if i == 255:
        i_inc = -1
    elif i == 0:
        i_inc = 1

    i += i_inc

    # concat joint angles with i
    action = np.concatenate((joint_angles, [i]))

    obs, reward, done, truncated, info = env.step(action)