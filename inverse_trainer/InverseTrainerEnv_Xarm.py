from typing import Dict, Union

import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from StateEvaluator import StateEvaluator
from StateCreator import StateCreator
from environments.XarmTableEnvironment import XarmTableEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class InverseTrainerEnv(XarmTableEnv):

    def __init__(
            self,
            state_evaluator : StateEvaluator = StateEvaluator(42),
            state_creator : StateCreator = StateCreator(42),
            state_dim = 42,
            action_dim = 7,
            state_reward_weight = 1,
            distance_reward_weight = 0.5,
            control_penalty_weight: float = 0.1,
    ):
        super(InverseTrainerEnv, self).__init__()
        self.state_evaluator = state_evaluator
        self.state_creator = state_creator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_reward_weight = state_reward_weight
        self.distance_reward_weight = distance_reward_weight
        self.control_penalty_weight = control_penalty_weight

        self.MIN_ANGLE = np.pi/6
        self.MIN_R = 0.4
        self.MAX_R = 0.8

        self.object_init_pos = np.zeros(3)
        self.goal_pos = np.array([0, -1, 0]) # IMPORTANT: hard-coded as the base of the robot

        self.metadata = XarmTableEnv.metadata


    def _get_obs(self):
        super()._get_obs()

    def step(self, action):
        super().step(action)

    def reset_model(self, push_direction=None):
        qpos = self.init_qpos
        qvel = self.init_qvel

        # X is to the right, Y is into the screen, Z is up
        # the square table is centered at (0, 0) with side length 2
        # the robot base is at (0, -1)

        R = np.random.uniform(self.MIN_R, self.MAX_R)

        # randomize the object in a circle around the robot
        angle = np.random.uniform(self.MIN_ANGLE, np.pi - self.MIN_ANGLE)
        obj_xy = self.robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])

        init_obj_pos = np.concatenate([obj_xy, [0]])

        self.object_init_pos = init_obj_pos

        qpos[:3] = init_obj_pos
        qpos[3:7] = [1, 0, 0, 0]

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_rew(self, action):

        current_state = self._get_obs()
        current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

        # CALCULATING STATE REWARD USING STATE EVALUATOR
        state_reward = -self.state_evaluator(current_state) # - sign rewards negative outputs from evaluator
        state_reward = state_reward.item()

        distance_to_object = self.get_body_com("object") - self.get_body_com("tips_arm")
        distance_reward = (1 / np.linalg.norm(distance_to_object)) * self.distance_reward_weight

        ctrl_penalty = -np.square(action).sum() * self.control_penalty_weight

        reward = state_reward + distance_reward + ctrl_penalty

        reward_info = {
            "state_reward": state_reward,
            "ctrl_penalty": ctrl_penalty
        }
        return reward, reward_info
