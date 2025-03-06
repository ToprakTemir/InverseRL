
# spawn a cube directly ahead at 0.85 distance and run the pretrained policy step by step

import os
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
import gymnasium as gym
import minari
import time

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv
from environments.XarmTableEnvironment import XarmTableEnv

from CustomPPOPolicy import CustomPolicy

class spawn_object(gym.Wrapper):
    def __init__(self, env: XarmTableEnv):
        super().__init__(env)

    def reset(self, **kwargs):
        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()

        qpos[:3] = [0.0, -0.15, -0.01]
        qpos[3:7] = [1, 0, 0, 0]  # cube orientation

        self.env.set_state(qpos, qvel)

        return self.env._get_obs()

    def step(self, action):
        return self.env.step(action)


# --- ENVIRONMENT SETUP ---


env = XarmTableEnv(control_option="ee_pos", render_mode="human")
env = spawn_object(env)

path = "./models/initial_policies/best_initial_policy_log_prob_03.02-04:42.pth"
initial_policy = CustomPolicy(env.observation_space, env.action_space)
pretrained_weights = torch.load(path, map_location=torch.device('cpu'))
initial_policy.load_state_dict(pretrained_weights)

model = PPO(CustomPolicy, env=env, verbose=1, device="cpu", use_sde=True)
model.policy.load_state_dict(initial_policy.state_dict())

while True:
    obs = env.reset()
    done = False
    step_count = 0

    # set the mocap_pos to the cube's position and wait until the ee reaches the mocap_pos
    env.env.data.mocap_pos = env.env.get_body_com("object") - np.array([0, 0.03, 0])
    env.env.wait_until_ee_reaches_mocap()

    while not done:
        action, _ = model.predict(obs)
        obs, _, done, done2, _ = env.step(action)
        done = done or done2 or step_count > 400
        step_count += 1