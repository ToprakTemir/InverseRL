import numpy as np
import torch

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import minari
from numpy.f2py.auxfuncs import isint1
from tensorflow.python.ops.numpy_ops.np_dtypes import object_

from StateEvaluator import StateEvaluator


class InverseTrainerEnv(gym.Wrapper):

    def __init__(
            self,
            state_evaluator : StateEvaluator,
            dataset: minari.MinariDataset,
            env: MujocoEnv,
            non_robot_indices_in_obs: list = None,
            max_episode_steps = 2000,
            state_reward_weight = 1,
            reward_control_weight = 0.1,
            **kwargs,
    ):
        super().__init__(env)

        self.state_evaluator = state_evaluator
        self.dataset = dataset
        self.non_robot_indices_in_obs = non_robot_indices_in_obs

        self.reward_control_weight = reward_control_weight
        self.reward_state_weight = state_reward_weight

        self.episode_step = 0
        self.max_episode_steps = max_episode_steps


    def step(self, action):
        """
        Overrides the step method of the environment, only changes the reward calculation.
        """

        self.episode_step += 1

        obs, _, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._get_rew(obs, action)

        if self.episode_step >= self.max_episode_steps:
            truncated = True
            terminated = True

        return obs, reward, terminated, truncated, info | reward_info

    def _get_rew(self, obs, action):
        obs = torch.tensor(obs, dtype=torch.float32)

        if self.non_robot_indices_in_obs is not None:
            obs = obs[self.non_robot_indices_in_obs]

        # state_evaluator's output is between 0 and 1, and we want to minimize it, so we give its negative as reward
        state_reward = 1 - self.state_evaluator(obs).item() * self.reward_state_weight

        # distance_to_object = self.get_body_com("object") - self.get_body_com("tips_arm")
        # distance_reward = (1 / np.linalg.norm(distance_to_object)) * self.reward_dist_weight

        ctrl_penalty = np.square(action).sum() * self.reward_control_weight

        reward = state_reward - ctrl_penalty

        reward_info = {
            "state_reward": state_reward,
            "ctrl_penalty": ctrl_penalty
        }
        return reward, reward_info

    def reset(self, **kwargs):
        """
        sets the environment to a random initial state similar to a moment in the dataset.
        """
        assert isinstance(self.env, MujocoEnv)

        self.episode_step = 0

        sampled_episode = list(self.dataset.sample_episodes(1))[0]
        sample_time = np.random.randint(0, len(sampled_episode))
        object_obs_at_sample_time = sampled_episode.observations[sample_time][self.non_robot_indices_in_obs]

        # TODO: check if always started from a given point (this method) results in high enough variety.
        # if not, a solution could be training a CNMP to learn the forward skill, and then sample an episode from it.

        obs = self.env.reset()
        obs = obs[0] # obs is given as a tuple of the observation as a list and the extra information dictionary. We only need the observation list

        qpos = self.env.init_qpos.copy()
        qpos[self.non_robot_indices_in_obs] = object_obs_at_sample_time # WARNING: this indexes aren't necessarily coincide for every environment. But it is correct for my xarm test scene
        self.env.set_state(qpos, self.env.init_qvel.copy())

        obs[self.non_robot_indices_in_obs] = object_obs_at_sample_time

        return obs, {}
