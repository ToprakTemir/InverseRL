import os.path

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
from scipy.ndimage import distance_transform_edt

from environments.XarmTableEnvironment import XarmTableEnv


class PushTrainerEnv(XarmTableEnv):

    def __init__(
        self,
        distance_weight=1.0,     # weight for inverse distance reward
        control_penalty_weight=0.1,    # penalty factor for large control actions
        force_penalty_weight = 0.1,
        render_mode = None,
    ):
        super().__init__(
            distance_weight=distance_weight,
            control_penalty_weight=control_penalty_weight,
            force_penalty_weight=force_penalty_weight,
            render_mode=render_mode,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(42,),
            dtype=np.float32,
        )

        self.robot_base_xy = np.array([0, -1])

        self.MIN_ANGLE = np.pi/6
        self.MIN_R = 0.4
        self.MAX_R = 0.8

        self.metadata["render_modes"] = [
            "human",
            "rgb_array",
            "depth_array",
        ]

    def step(self, action):
        return super().step(action)

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel

        # X is to the right, Y is into the screen, Z is up
        # the square table is centered at (0, 0) with side length 1
        # the robot base is at (0, -1)

        R = np.random.uniform(self.MIN_R, self.MAX_R)

        # randomize the object in a circle around the robot
        angle = np.random.uniform(self.MIN_ANGLE, np.pi - self.MIN_ANGLE)
        obj_xy = self.robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])

        init_cube_pos = np.concatenate([obj_xy, [-0.04]])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]

        self.set_state(qpos, qvel)
        return self._get_obs()


    def calculate_collision_penalty(self):
        total_force_on_table = 0
        contact_forces = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name

            if "table" in geom1 or "table" in geom2:
                mujoco.mj_contactForce(self.model, self.data, i, contact_forces)

                force_magnitude = np.linalg.norm(contact_forces[:3])
                total_force_on_table += force_magnitude

        force_penalty = total_force_on_table * self.force_penalty_weight

        return force_penalty


    def _compute_reward(self, action):

        # reward = push_reward + distance_reward - control_penalty

        ee_pos = super().get_ee_pos()
        obj_pos = super().get_object_position()
        distance_reward = (- np.linalg.norm(ee_pos - obj_pos)) * self.distance_weight

        # set goal as a far point in the same line with the robot and the object
        line_unit = (obj_pos[:2] - self.robot_base_xy) / np.linalg.norm(obj_pos[:2] - self.robot_base_xy)
        goal = obj_pos[:2] + line_unit * 2

        push_reward = (1 / np.linalg.norm(obj_pos[:2] - goal)) * self.distance_weight * 10

        control_penalty = np.linalg.norm(action) * self.control_penalty_weight

        force_on_table = super().compute_total_force_on_table()
        force_penalty = force_on_table * self.force_penalty_weight

        reward = push_reward + distance_reward - control_penalty - force_penalty

        return reward







