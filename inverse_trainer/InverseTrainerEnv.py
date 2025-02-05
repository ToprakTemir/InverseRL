from typing import Dict, Union

import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import minari

from StateEvaluator import StateEvaluator

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
            dataset: minari.MinariDataset,
            env: MujocoEnv,
            state_reward_weight = 1,
            reward_dist_weight = 0.5,
            reward_control_weight: float = 0.1,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            state_evaluator,
            dataset,
            env,
            state_reward_weight,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )

        self.state_evaluator = state_evaluator
        self.dataset = dataset
        self.env = env

        self._reward_control_weight = reward_control_weight
        self.reward_dist_weight = reward_dist_weight
        self._reward_state_weight = state_reward_weight

        MujocoEnv.__init__(
            self,
            model_path=env.fullpath,
            frame_skip=env.frame_skip,
            observation_space=env.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
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

        obs, _, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._get_rew(obs, action)

        return obs, reward, terminated, truncated, info | reward_info

    def _get_rew(self, obs, action):

        obs = torch.tensor(obs, dtype=torch.float32)
        state_reward = - self.state_evaluator(obs).item() # - sign makes negative output from evaluator desirable

        # distance_to_object = self.get_body_com("object") - self.get_body_com("tips_arm")
        # distance_reward = (1 / np.linalg.norm(distance_to_object)) * self.reward_dist_weight

        ctrl_penalty = np.square(action).sum() * self._reward_control_weight

        reward = state_reward - ctrl_penalty

        reward_info = {
            "state_reward": state_reward,
            "ctrl_penalty": ctrl_penalty
        }
        return reward, reward_info

    def reset_model(self):
        """
        sets the environment to a random initial state similar to a moment in the dataset.
        """

        sampled_episode = list(self.dataset.sample_episodes(1))[0]
        sample_time = np.random.randint(0, len(sampled_episode))

        # TODO: check if always started from a given point (this method) results in high enough variety.
        # if not, a solution could be training a CNMP to learn the forward skill, and then sample an episode from it.

        self.env.reset()
        for i in range(sample_time-1):
            self.env.step(sampled_episode.actions[i])

        obs, _, _, _, _ = self.env.step(sampled_episode.actions[sample_time-1])

        return obs
