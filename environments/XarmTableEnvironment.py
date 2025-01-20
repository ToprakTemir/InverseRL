import os.path

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

class XarmTableEnv(MujocoEnv, EzPickle):
    """
    A tabletop environment with an xArm7 manipulator placed on one edge
    of a 1x1x1 table, and a 5cm cube on the table. The robot is position-controlled.
    Collisions between the arm, table, and cube are enabled.

    Observations (dim=20):
        0-6   : joint positions [7]
        7-13  : joint velocities [7]
        14-16 : end-effector xyz [3]
        17-19 : cube xyz [3]

    Reward:
        - The inverse of the distance between the end-effector and the cube
        - Minus a small penalty for large control actions

    Episode termination:
        - If a strong collision with the table occurs (fail flag = True)
        - After max_steps
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file="xarm7_tabletop.xml",
        frame_skip=5,
        distance_weight=1.0,     # weight for inverse distance reward
        control_penalty_weight=0.1,    # penalty factor for large control actions
        force_penalty_weight = 0.1,
        render_mode = None,
    ):
        EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            distance_weight,
            control_penalty_weight,
            render_mode,
        )

        self.distance_weight = distance_weight
        self.control_penalty_weight = control_penalty_weight
        self.force_penalty_weight = force_penalty_weight

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config={"trackbodyid": -1, "distance": 2.5},
            # If you're controlling positions directly, you might want to
            # override other MujocoEnv settings, or ensure the xArm model
            # is set up for position control properly.
            observation_space=self.observation_space,
            render_mode=render_mode,
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
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self._compute_reward(action)

        collusion = False
        if self._check_collision_with_table():
            collusion = True

        info = {
            "collusion": collusion,
        }
        return obs, reward, False, False, info

    def compute_total_force_on_table(self):
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

        return total_force_on_table

    def _compute_reward(self, action):
        # Distance-based reward
        ee_pos = self.get_ee_pos()
        cube_pos = self.get_object_position()
        distance = np.linalg.norm(ee_pos - cube_pos)
        distance_reward = (1 / (distance + 1e-6)) * self.distance_weight

        # Control penalty
        ctrl_cost = np.sum(np.square(action)) * self.control_penalty_weight

        # table collision penalty
        total_force_on_table = 0
        for i in range(self.model.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom_id2name(contact.geom1)
            geom2 = self.model.geom_id2name(contact.geom2)

            if "table" in geom1 or "table" in geom2:
                total_force_on_table += np.linalg.norm(contact.f)

        force_penalty = total_force_on_table * self.force_penalty_weight

        reward = distance_reward - ctrl_cost - force_penalty

        return reward

    def _check_collision_with_table(self):
        return False

    def _get_obs(self):
        """
        Returns a 42D observation:

        qpos: 20d
        - 7d cube position and orientation
        - 7d joint positions
        - 6d gripper joint positions

        qvel: 20d
        - 6d cube linear and angular velocities
        - 7d joint velocities
        - 6d gripper joint velocities

        ee_pos: 3d
        - 3d end-effector position
        """
        qpos = self.data.qpos[:].copy()
        qvel = self.data.qvel[:].copy()
        ee_pos = self.get_ee_pos()
        return np.concatenate([qpos, qvel, ee_pos]).astype(np.float32)

    def get_ee_pos(self):
        left_finger_pos = self.data.body("left_finger").xpos.copy()
        right_finger_pos = self.data.body("right_finger").xpos.copy()
        return (left_finger_pos + right_finger_pos) / 2

    def get_object_position(self):
        return self.data.body("object").xpos.copy()

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel

        # X is to the right, Y is into the screen, Z is up
        # the square table is centered at (0, 0) with side length 1
        # the robot base is at (0, -1)

        init_cube_pos = np.array([0, -0.5, 0.04])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:14] = [0, 0, 0, 0, 0, 0, 0]

        qvel[:7] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()