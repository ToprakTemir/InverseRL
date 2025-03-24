import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

class XarmTableEnv(MujocoEnv, EzPickle):
    """
    A tabletop environment with an xArm7 manipulator placed on one edge
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
        control_option="ee_pos", # joint_angles / ee_pos / discrete_ee_pos
        frame_skip=5,
        distance_weight=1.0,
        control_penalty_weight=0.1,    # penalty factor for large control actions
        force_penalty_weight = 0.01,   # penalty factor for collisions
        max_episode_steps = None,
        render_mode = None,
    ):
        EzPickle.__init__(
            self,
            xml_file,
            control_option,
            frame_skip,
            distance_weight,
            control_penalty_weight,
            force_penalty_weight,
            render_mode,
        )

        self.distance_weight = distance_weight
        self.control_penalty_weight = control_penalty_weight
        self.force_penalty_weight = force_penalty_weight

        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

        self.observation_dim = 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
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

        self.control_option = control_option
        self.robot_base = np.array([0, -1, 0])
        if self.control_option == "ee_pos":
            R = 0.9 # the max reach of robot

            self.robot_base = np.array([0, -1, 0, 0])
            # (x, y, z, gripper)
            self.action_space = spaces.Box(
                low=np.array([-R, 0, -0.04, -1e-10]) + self.robot_base, # I don't let going backward or underground, hence min y and z is 0
                high=np.array([R, R, R, 1 + 1e-10]) + self.robot_base,
                dtype=np.float32,
            )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.robot_base_xy = np.array([0, -1])


    def step(self, action):
        if action.shape[0] == 1:
            action = action[0]
        if self.control_option == "joint_angles":
            action[7] *= 255
            action[7].clip(0, 255)
            self.do_simulation(action, self.frame_skip)

        elif self.control_option == "ee_pos":
            action[3] *= 255
            action[3].clip(0, 255)

            self.data.mocap_pos = action[:3]
            self.data.mocap_quat = [0, 0, 0, 1]
            self.data.ten_length = action[3]

            for i in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

        elif self.control_option == "discrete_ee_pos":
            action_mapping = {
                0: np.array([0.05, 0, -0.034, 0]),  # Move +X
                1: np.array([-0.05, 0, -0.034, 0]),  # Move -X
                2: np.array([0, 0.05, -0.034, 0]),  # Move +Y
                3: np.array([0, -0.05, -0.034, 0]),  # Move -Y
                4: np.array([0, 0, 0, 255]),  # Close gripper
                5: np.array([0, 0, 0, 0]),  # Open gripper
            }

            # assert action is a one hot encoded vector
            assert np.sum(action) == 1, "Action must be one hot encoded"
            action = np.argmax(action)
            action = action_mapping[action]

            self.data.mocap_pos = action[:3]
            self.data.mocap_quat = [0, 0, 0, 1]
            self.data.ten_length = action[3]
            for i in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)


        else:
            raise ValueError(f"Invalid control option: {self.control_option}")

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.max_episode_steps is not None:
            self.episode_steps += 1
            if self.episode_steps > self.max_episode_steps:
                truncated = True

        return obs, reward, False, False, info

    def wait_until_ee_reaches_mocap(self):
        mocap_pos = self.get_mocap_pos()
        ee_pos = self.get_ee_pos()
        model = self.model
        data = self.data
        # print(f"distance between mocap and ee: {np.linalg.norm(mocap_pos - ee_pos)}")
        MAX_WAIT = 200
        cur = 0
        while np.linalg.norm(mocap_pos - ee_pos) > 0.05:
            mujoco.mj_step(model, data)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_forward(model, data)
            if self.render_mode == "human":
                self.render()

            ee_pos = self.get_ee_pos()
            # print(f"distance between mocap and ee: {np.linalg.norm(mocap_pos - ee_pos)}")

            cur += 1
            if cur > MAX_WAIT:
                break

        # print("action completed")

    def reset(self):
        obs = self.reset_model()
        self.episode_steps = 0
        return obs

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel

        # X is to the right, Y is into the screen, Z is up
        # the square table is centered at (0, 0) with side length 2
        # the robot base is at (0, -1)

        init_cube_pos = np.array([0, -0.5, 0.04])

        # qpos: 0:3 cube position, 3:7 cube orientation, 7:14 joint positions
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:14] = [0, 0, 0, 0, 0, 0, 0]
        qvel[:7] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    # def compute_total_force_on_table(self):
    #     total_force_on_table = 0
    #     contact_forces = np.zeros(6)
    #
    #     for i in range(self.data.ncon):
    #         contact = self.data.contact[i]
    #         geom1 = self.model.geom(contact.geom1).name
    #         geom2 = self.model.geom(contact.geom2).name
    #
    #         if "table" in geom1 or "table" in geom2:
    #             mujoco.mj_contactForce(self.model, self.data, i, contact_forces)
    #
    #             force_magnitude = np.linalg.norm(contact_forces[:3])
    #             total_force_on_table += force_magnitude
    #
    #     return total_force_on_table


    def _get_obs(self):
        """
        Returns a 14D observation:
        0:2 3d cube position
        3:9 7d joint positions
        10 1d gripper closeness (0: fully open, 255: fully closed)
        11:13 3d end-effector position
        """
        qpos = self.data.qpos[:].copy()

        cube_pos = qpos[:3]
        joint_pos = qpos[7:14]
        gripper_closeness = self.model.tendon_length0 / 255
        ee_pos = self.get_ee_pos()

        # important: high/low level observation settings here
        if self.observation_dim == 14:
            return np.concatenate([cube_pos, joint_pos, gripper_closeness, ee_pos]).astype(np.float32)
        elif self.observation_dim == 6:
            return np.concatenate([cube_pos, ee_pos]).astype(np.float32)
        else:
            raise ValueError(f"Invalid observation dimension: {self.observation_dim}")

    def get_ee_pos(self):
        # left_finger_pos = self.data.body("left_finger").xpos.copy()
        # right_finger_pos = self.data.body("right_finger").xpos.copy()
        # return (left_finger_pos + right_finger_pos) / 2
        return self.data.body("end_effector").xpos.copy()

    def get_mocap_pos(self):
        return self.data.mocap_pos.copy()

    def get_object_pos(self):
        return self.data.body("object").xpos.copy()


if __name__ == "__main__":
    env = XarmTableEnv(render_mode="human", control_option="ee_pos")
    print(env.action_space)
    while True:
        obs = env.reset()
        for _ in range(100000):
            obs, _, end1, end2, _ = env.step(env.action_space.sample())
