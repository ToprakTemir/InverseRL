import numpy as np
import torch

import mujoco
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import minari

from models.StateEvaluator import StateEvaluator


class InverseTrainerEnv(gym.Wrapper):

    def __init__(
            self,
            env: MujocoEnv,
            state_evaluator : StateEvaluator,
            dataset: minari.MinariDataset,
            non_robot_indices_in_obs: list = None,
            max_episode_steps = 500,
            reward_control_weight = 0.001,
            **kwargs,
    ):
        super().__init__(env)


        self.state_evaluator = state_evaluator
        self.dataset = dataset
        self.non_robot_indices_in_obs = non_robot_indices_in_obs

        self.reward_control_weight = reward_control_weight

        self.episode_step = 0
        self.max_episode_steps = max_episode_steps

        self.previous_reward = 0.5
        self.previous_obs = None
        self.terminated = False
        self.truncated = False

        self.goal_lowering_frequency = 10
        self.passed_treshold_count = 0
        self.goal_treshold = 0.31

    def step(self, action):
        """
        Overrides the step method of the environment, only changes the reward calculation.
        """

        if self.terminated or self.truncated:
            return self.env._get_obs(), 0, self.terminated, self.truncated, {}

        self.episode_step += 1
        low = torch.tensor(self.action_space.low, dtype=torch.float32)
        high = torch.tensor(self.action_space.high, dtype=torch.float32)
        action = action.clip(low, high)

        obs, _, terminated, truncated, info = self.env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        if self.non_robot_indices_in_obs is not None:
            obs_tensor = obs_tensor[self.non_robot_indices_in_obs]

        reward, reward_info = self._get_rew(obs_tensor, action)

        def fallen_from_table(obj_pos):
            x_out = obj_pos[0] < -1 or obj_pos[0] > 1
            y_out = obj_pos[1] < -1 or obj_pos[1] > 1
            return x_out or y_out

        if fallen_from_table(self.env.get_body_com("object")):
            reward -= 50
            self.terminated = True

        if self.state_evaluator(obs_tensor).item() < self.goal_treshold:
            reward += 50
            self.terminated = True
            self.passed_treshold_count += 1
            if self.passed_treshold_count == self.goal_lowering_frequency:
                self.passed_treshold_count = 0
                self.goal_treshold = max(self.goal_treshold - 0.01, 0.1)
            print(f"Reached the goal!, pos={obs_tensor.numpy()}, new goal treshold: {self.goal_treshold}")

        if self.episode_step >= self.max_episode_steps:
            self.truncated = True

        return obs, reward, self.terminated, self.truncated, info | reward_info

    def _get_rew(self, obs, action):
        # state_evaluator's output is between 0 and 1, and we want to minimize it, so we give its negative as reward
        # state_reward = 1/2 - self.state_evaluator(obs).item()
        if self.previous_obs is None:
            self.previous_obs = obs


        state_reward = 0.5 - self.state_evaluator(obs).item()

        # def rew(observation):
        #     return 0.5 - self.state_evaluator(observation).item()
        # state_reward = rew(obs) - rew(self.previous_obs) * 0.95 # delta state reward
        # state_reward += max(0, rew(obs)) # positive reward if more than halfway towards the goal
        # state_reward *= 10

        self.previous_obs = obs

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
        self.truncated = False
        self.terminated = False
        self.previous_reward = 0.5
        self.previous_obs = None

        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()

        R = 0.8
        MIN_ANGLE = np.pi/6
        # MIN_ANGLE = 87 * (np.pi / 180)  # directly forward
        robot_base_xy = np.array([0, -1])
        angle = np.random.uniform(MIN_ANGLE, np.pi - MIN_ANGLE)

        obj_xy = robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])
        # noise = np.random.normal(0, 0.05, 2)
        # obj_xy = [-0.1, -0.3] + noise # WARNING: TEST

        init_cube_pos = np.concatenate([obj_xy, [-0.01]])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]  # cube orientation

        self.env.set_state(qpos, qvel)

        # FOR SETTING THE ARM POSITION ABOVE AND BEHIND THE OBJECT
        # obj_pos = self.env.get_body_com("object").copy()
        # obj_direction = obj_pos - np.array([0, -1, 0])
        # obj_direction /= np.linalg.norm(obj_direction)
        # self.env.data.mocap_pos = obj_pos + obj_direction * 0.2 + np.array([0, 0, 0.3])
        # self.env.data.mocap_quat = [0, 0, 0, 1]
        # self.env.wait_until_ee_reaches_mocap()

        # FOR SETTING THE ARM POSITION IN PUSH POSITION
        # obj_pos = self.env.get_body_com("object").copy()
        # obj_direction = obj_pos - np.array([0, -1, 0])
        # obj_direction /= np.linalg.norm(obj_direction)
        # self.env.data.mocap_pos = obj_pos + obj_direction * -0.1
        # self.env.data.mocap_quat = [0, 0, 0, 1]
        # self.env.wait_until_ee_reaches_mocap()

        # STANDARD RESETTING WITHOUT HELPING THE ROBOT
        self.env.data.mocap_pos = self.env.get_ee_pos()
        self.env.data.mocap_pos[:, 2] += 0.01
        self.env.data.mocap_quat = [0, 0, 0, 1]
        for i in range(self.env.frame_skip):
            mujoco.mj_step(self.env.model, self.env.data)
            mujoco.mj_kinematics(self.env.model, self.env.data)
            mujoco.mj_forward(self.env.model, self.env.data)

        return self.env._get_obs(), None

        # Old version, places the object to a random position sampled from the dataset.
        # ---------------------------------------
        # sampled_episode = list(self.dataset.sample_episodes(1))[0]
        # sample_time = np.random.randint(0, len(sampled_episode))
        # object_obs_at_sample_time = sampled_episode.observations[sample_time][self.non_robot_indices_in_obs]
        #
        # # TODO: check if always started from a given point (this method) results in high enough variety.
        # # if not, a solution could be training a CNMP to learn the forward skill, and then sample an episode from it.
        #
        # obs = self.env.reset()
        # obs = obs[0] # obs is given as a tuple of the observation as a list and the extra information dictionary. We only need the observation list
        #
        # qpos = self.env.init_qpos.copy()
        # qpos[self.non_robot_indices_in_obs] = object_obs_at_sample_time # WARNING: this indexes doesn't necessarily coincide for every environment. But it is correct for my xarm test scene
        # self.env.set_state(qpos, self.env.init_qvel.copy())
        #
        # obs[self.non_robot_indices_in_obs] = object_obs_at_sample_time
        #
        # obs = self.env._get_obs()
        # return obs, {}

    # def reset(self, **kwargs):
    #     qpos = self.env.init_qpos.copy()
    #     qvel = self.env.init_qvel.copy()
    #
    #     # X is to the right, Y is into the screen, Z is up.
    #     # The square table is centered at (0, 0) with side length 2,
    #     # and the robot base is at (0, -1).
    #     R = np.random.uniform(self.MIN_R, self.MAX_R)
    #
    #     # Randomize the object in a circle around the robot.
    #     angle = np.random.uniform(self.MIN_ANGLE, np.pi - self.MIN_ANGLE)
    #     obj_xy = self.env.robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])
    #
    #     init_cube_pos = np.concatenate([obj_xy, [-0.01]])
    #     qpos[:3] = init_cube_pos
    #     qpos[3:7] = [1, 0, 0, 0]  # cube orientation
    #
    #     self.env.set_state(qpos, qvel)
    #
    #     ee_pos = self.get_ee_pos()
    #     self.env.data.mocap_pos = ee_pos
    #     self.env.data.mocap_pos[:, 2] += 0.01
    #     self.env.data.mocap_quat = [0, 0, 0, 1]
    #     mujoco.mj_step(self.env.model, self.env.data)
    #     mujoco.mj_kinematics(self.env.model, self.env.data)
    #     mujoco.mj_forward(self.env.model, self.env.data)
    #
    #     return self.env._get_obs(), None
