import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import glfw
from scipy.spatial.transform import Rotation as R

###############################################################################
# Utility functions.
###############################################################################
def rotation(theta_x=0, theta_y=0, theta_z=0):
    """Return a rotation matrix for rotations about x, then y, then z."""
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])
    rot_y = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    return rot_x.dot(rot_y).dot(rot_z)

def quat2euler(quat):
    """
    Convert a quaternion in MuJoCo's (w, x, y, z) order to Euler angles (XYZ).
    (scipy expects quaternions in (x, y, z, w) order.)
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler('XYZ')

###############################################################################
# Direction definitions.
###############################################################################
class Direction:
    POS = 1
    NEG = -1

###############################################################################
# Controller for keyboard control using the mocap.
###############################################################################
class Controller:
    MAX_SPEED = 1.0
    MIN_SPEED = 0.0
    SPEED_CHANGE_PERCENT = 0.2

    def __init__(self, env):
        """
        env: The environment instance. We expect the env to have attributes
             'model' and 'data' (from MujocoEnv).
        """
        self.env = env
        self._speeds = np.array([0.01, 0.1])
        # Find the mocap body index.
        mocap_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        self.mocap_idx = self.env.model.body_mocapid[mocap_body_id]

    @property
    def pos_speed(self):
        return self._speeds[0]

    @property
    def rot_speed(self):
        return self._speeds[1]

    def speed_up(self):
        self._speeds = np.minimum(
            self._speeds * (1 + self.SPEED_CHANGE_PERCENT), self.MAX_SPEED
        )

    def speed_down(self):
        self._speeds = np.maximum(
            self._speeds * (1 - self.SPEED_CHANGE_PERCENT), self.MIN_SPEED
        )

    def move_x(self, direction: int):
        current_pos = self.env.data.mocap_pos[self.mocap_idx].copy()
        delta = np.array([self.pos_speed * direction, 0, 0])
        new_pos = current_pos + delta
        self.env.data.mocap_pos[self.mocap_idx] = new_pos
        mujoco.mj_forward(self.env.model, self.env.data)

    def move_y(self, direction: int):
        current_pos = self.env.data.mocap_pos[self.mocap_idx].copy()
        delta = np.array([0, self.pos_speed * direction, 0])
        new_pos = current_pos + delta
        self.env.data.mocap_pos[self.mocap_idx] = new_pos
        mujoco.mj_forward(self.env.model, self.env.data)

    def move_z(self, direction: int):
        current_pos = self.env.data.mocap_pos[self.mocap_idx].copy()
        delta = np.array([0, 0, self.pos_speed * direction])
        new_pos = current_pos + delta
        self.env.data.mocap_pos[self.mocap_idx] = new_pos
        mujoco.mj_forward(self.env.model, self.env.data)

    def rot_x(self, direction: int):
        current_quat = self.env.data.mocap_quat[self.mocap_idx].copy()
        euler_angles = quat2euler(current_quat)
        new_angles = (euler_angles[0] + self.rot_speed * direction,
                      euler_angles[1],
                      euler_angles[2])
        rot_matrix = rotation(*new_angles)
        new_quat_scipy = R.from_matrix(rot_matrix).as_quat()  # (x,y,z,w)
        new_quat = np.array([new_quat_scipy[3],
                             new_quat_scipy[0],
                             new_quat_scipy[1],
                             new_quat_scipy[2]])
        self.env.data.mocap_quat[self.mocap_idx] = new_quat
        mujoco.mj_forward(self.env.model, self.env.data)

    def rot_y(self, direction: int):
        current_quat = self.env.data.mocap_quat[self.mocap_idx].copy()
        euler_angles = quat2euler(current_quat)
        new_angles = (euler_angles[0],
                      euler_angles[1] + self.rot_speed * direction,
                      euler_angles[2])
        rot_matrix = rotation(*new_angles)
        new_quat_scipy = R.from_matrix(rot_matrix).as_quat()
        new_quat = np.array([new_quat_scipy[3],
                             new_quat_scipy[0],
                             new_quat_scipy[1],
                             new_quat_scipy[2]])
        self.env.data.mocap_quat[self.mocap_idx] = new_quat
        mujoco.mj_forward(self.env.model, self.env.data)

    def rot_z(self, direction: int):
        current_quat = self.env.data.mocap_quat[self.mocap_idx].copy()
        euler_angles = quat2euler(current_quat)
        new_angles = (euler_angles[0],
                      euler_angles[1],
                      euler_angles[2] + self.rot_speed * direction)
        rot_matrix = rotation(*new_angles)
        new_quat_scipy = R.from_matrix(rot_matrix).as_quat()
        new_quat = np.array([new_quat_scipy[3],
                             new_quat_scipy[0],
                             new_quat_scipy[1],
                             new_quat_scipy[2]])
        self.env.data.mocap_quat[self.mocap_idx] = new_quat
        mujoco.mj_forward(self.env.model, self.env.data)

###############################################################################
# The Gymnasium Mujoco environment.
###############################################################################
class XarmTableEnv(MujocoEnv, EzPickle):
    """
    A tabletop environment with an xArm7 manipulator placed on one edge
    of a 2x2x1 table, and a 5cm cube on the table. The robot is position-controlled.
    Collisions between the arm, table, and cube are enabled.
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
        distance_weight=1.0,
        control_penalty_weight=0.1,    # penalty factor for large control actions
        force_penalty_weight=0.01,     # penalty factor for collisions
        render_mode=None,
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
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config={"trackbodyid": -1, "distance": 2.5},
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

        self.robot_base_xy = np.array([0, -1])

        # If using human render mode, initialize keyboard control.
        if render_mode == "human":
            self.controller = Controller(self)
            self._keyboard_callback_set = False

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
        return 0

    def _check_collision_with_table(self):
        return False

    def _get_obs(self):
        """
        Returns a 14D observation:
          0:2   -> 3D cube position
          3:9   -> 7D joint positions
          10    -> 1D gripper closeness (0: fully open, 255: fully closed)
          11:13 -> 3D end-effector position
        """
        qpos = self.data.qpos[:].copy()
        cube_pos = qpos[:3]
        joint_pos = qpos[7:14]
        # Here we assume gripper closeness is represented by a single tendon length.
        gripper_closeness = self.model.tendon_length0
        ee_pos = self.get_ee_pos()
        return np.concatenate([cube_pos, joint_pos, gripper_closeness, ee_pos]).astype(np.float32)

    def get_ee_pos(self):
        left_finger_pos = self.data.body("left_finger").xpos.copy()
        right_finger_pos = self.data.body("right_finger").xpos.copy()
        return (left_finger_pos + right_finger_pos) / 2

    def get_object_position(self):
        return self.data.body("object").xpos.copy()

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # The cube is placed on the table; adjust as needed.
        init_cube_pos = np.array([0, -0.5, 0.04])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:14] = [0, 0, 0, 0, 0, 0, 0]

        qvel[:7] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _keyboard_callback(self, window, key, scancode, action, mods):
        """Keyboard callback to update the mocap-controlled end-effector."""

        print("Key event:", key, action)  # Debug print to check for key events

        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                self.controller.move_z(Direction.POS)
            elif key == glfw.KEY_DOWN:
                self.controller.move_z(Direction.NEG)
            elif key == glfw.KEY_RIGHT:
                self.controller.move_y(Direction.POS)
            elif key == glfw.KEY_LEFT:
                self.controller.move_y(Direction.NEG)
            elif key == glfw.KEY_B:
                self.controller.move_x(Direction.NEG)
            elif key == glfw.KEY_F:
                self.controller.move_x(Direction.POS)
            elif key == glfw.KEY_Q:
                self.controller.rot_x(Direction.POS)
            elif key == glfw.KEY_W:
                self.controller.rot_x(Direction.NEG)
            elif key == glfw.KEY_A:
                self.controller.rot_y(Direction.POS)
            elif key == glfw.KEY_S:
                self.controller.rot_y(Direction.NEG)
            elif key == glfw.KEY_Z:
                self.controller.rot_z(Direction.POS)
            elif key == glfw.KEY_X:
                self.controller.rot_z(Direction.NEG)
            elif key == glfw.KEY_MINUS:
                self.controller.speed_down()
            elif key == glfw.KEY_EQUAL:
                self.controller.speed_up()

    def render(self):
        if self.render_mode == "human" and not self._keyboard_callback_set:
            # Launch the viewer manually if it doesn’t exist
            mujoco.viewer.launch(self.model, self.data)

            # Attach the key callback after launching the viewer
            if glfw.get_current_context() is not None:  # Ensure there is an active OpenGL context
                glfw.set_key_callback(glfw.get_current_context(), self._keyboard_callback)
                self._keyboard_callback_set = True


if __name__ == "__main__":
    env = XarmTableEnv(render_mode="human")
    obs, _ = env.reset()
    # Create a zero-action vector (assumes action_space is a Box)
    no_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    while True:
        obs, reward, done, truncated, info = env.step(no_action)