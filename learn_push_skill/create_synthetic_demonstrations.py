import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.optimize import Bounds

from environments.XarmTableEnvironment import XarmTableEnv

# For this example we assume that the robot joint angles are stored
# in qpos indices 7 to 13 (7 values) so we define:
robot_joint_indices = slice(3, 10)

class PushDemonstratorEnv(XarmTableEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, render_mode=None):
        super(PushDemonstratorEnv, self).__init__(render_mode=render_mode)
        self.MIN_ANGLE = np.pi / 6
        self.MIN_R = 0.4
        self.MAX_R = 0.8
        model = mujoco.MjModel.from_xml_path("/Users/toprak/InverseRL/environments/assets/xml_files/xarm7_tabletop.xml")
        data = mujoco.MjData(model)

    def step(self, action):

        self.data.mocap_pos = action[:3]
        self.data.mocap_quat = action[3:7]
        self.data.tendon_length[0] = action[7]

        obs = self._get_obs()

        reward = self._compute_reward(action)
        info = {}
        return obs, reward, False, False, info

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
        return super()._get_obs()

    def get_ee_pos(self):
        return super().get_ee_pos()

    def _compute_reward(self, action):
        return 0

    def get_object_position(self):
        return self.data.body("object").xpos.copy()

    def reset_model(self):
        # Get a copy of the initial state.
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # X is to the right, Y is into the screen, Z is up.
        # The square table is centered at (0, 0) with side length 2,
        # and the robot base is at (0, -1).
        R = np.random.uniform(self.MIN_R, self.MAX_R)

        # Randomize the object in a circle around the robot.
        angle = np.random.uniform(self.MIN_ANGLE, np.pi - self.MIN_ANGLE)
        obj_xy = self.robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])

        init_cube_pos = np.concatenate([obj_xy, [-0.04]])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]  # cube orientation

        self.set_state(qpos, qvel)
        return self._get_obs()


# Global variables for the demonstration policy
inPushingPosition = False
robot_base = np.array([0, -1, 0])

def demonstration_policy(observation, env):

    obj_xyz = observation[:3]
    movement_direction = obj_xyz - robot_base
    movement_direction /= np.linalg.norm(movement_direction)
    movement_direction *= 0.001

    target_pos = obj_xyz + movement_direction

    target_quat = [0, 0.707, 0.707, 0]
    gripper_closeness = 0
    action = np.concatenate([target_pos, target_quat, [gripper_closeness]])
    return action

if __name__ == "__main__":
    env = PushDemonstratorEnv(render_mode="human")
    observation = env.reset()
    for _ in range(1000):
        action = demonstration_policy(observation, env)
        observation, _, _, _, _ = env.step(action)
