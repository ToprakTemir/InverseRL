from typing import Dict, Union

import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from StateEvaluator import StateEvaluator
from StateCreator import StateCreator

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class InverseTrainerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
            self,
            state_evaluator : StateEvaluator,
            state_creator : StateCreator,
            state_dim = 23,
            action_dim = 7,
            state_reward_weight = 1,
            reward_dist_weight = 0.5,
            reward_control_weight: float = 0.1,
            xml_file: str = "pusher_v5.xml",
            frame_skip: int = 5,
            default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_control_weight,
            **kwargs,
        )
        self.state_evaluator = state_evaluator
        self.state_creator = state_creator
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._reward_control_weight = reward_control_weight
        self.reward_dist_weight = reward_dist_weight
        self._reward_state_weight = state_reward_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):

        current_state = self._get_obs()
        current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

        # CALCULATING STATE REWARD USING STATE EVALUATOR
        state_reward = -self.state_evaluator(current_state) # - rewards getting negative output from evaluator
        state_reward = state_reward.item()

        # CALCULATING STATE REWARD USING STATE CREATOR
        # sample a goal state and give reward based on the distance to the goal state
        # goal_state = self.state_creator.sample(torch.tensor([-1.0])).flatten().detach().numpy()
        #
        # current_state_without_robot = current_state[-6:]
        # goal_state_without_robot = goal_state[-6:]
        # distance_to_goal_state = np.linalg.norm(current_state_without_robot - goal_state_without_robot)
        # state_reward = (1 / distance_to_goal_state) * self._reward_state_weight

        distance_to_object = self.get_body_com("object") - self.get_body_com("tips_arm")
        distance_reward = (1 / np.linalg.norm(distance_to_object)) * self.reward_dist_weight

        ctrl_penalty = -np.square(action).sum() * self._reward_control_weight

        reward = state_reward + distance_reward + ctrl_penalty

        reward_info = {
            "state_reward": state_reward,
            "ctrl_penalty": ctrl_penalty
        }
        return reward, reward_info

    def reset_model(self, push_direction=None):

        predicted_start = self.state_creator.sample(torch.tensor([1.0])).flatten().detach().numpy()

        qpos = self.init_qpos

        qpos[-4:-2] = predicted_start[17:19] # IMPORTANT: only valid for the gym pusher environment
        qpos[-2:] = predicted_start[20:22] # IMPORTANT: only valid for the gym pusher environment

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )