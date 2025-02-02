import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
from scipy.optimize import minimize

from scipy.optimize import Bounds

from environments.XarmTableEnvironment import XarmTableEnv

# For this example we assume that the robot joint angles are stored
# in qpos indices 7 to 13 (7 values) so we define:
robot_joint_indices = slice(7, 14)

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
initial_joint_angles = None
robot_base = np.array([0, -1, 0])

# Load the model and data from the MuJoCo XML.
model = mujoco.MjModel.from_xml_path("/Users/toprak/InverseRL/environments/assets/necessary_gymnasium_xml_files/xarm7_tabletop.xml")
data = mujoco.MjData(model)

def forward_kinematics(model, data, joint_angles):
    # Update the relevant portion of qpos with the candidate joint angles
    qpos = data.qpos.copy()
    qpos[robot_joint_indices] = joint_angles[:7]
    data.qpos[:] = qpos
    # Update simulation state based on the new qpos
    mujoco.mj_forward(model, data)
    # Now compute the average position of the fingers
    left_finger_pos = data.body("left_finger").xpos.copy()
    right_finger_pos = data.body("right_finger").xpos.copy()
    return (left_finger_pos + right_finger_pos) / 2


def ik_objective(joint_angles, model, data, target_position):
    """
    IK objective: the Euclidean distance between the current end-effector
    position (given joint_angles) and the target_position.
    """
    ee_pos = forward_kinematics(model, data, joint_angles)
    loss = np.linalg.norm(ee_pos - target_position)

    # mujoco.mj_forward(model, data)
    # contact_penalty = 0
    # for i in range(data.ncon):  # Loop through contacts
    #     contact = data.contact[i]
    #     force = np.zeros(6)
    #     mujoco.mj_contactForce(model, data, i, force)
    #     contact_penalty += np.linalg.norm(force)  # Penalize collisions

    return loss

def calculate_inverse_kinematics(model, data, target_position, initial_guess):
    """
    Solves for robot joint angles that drive the end-effector toward target_position
    while avoiding collisions.
    """

    # Define joint limits as constraints
    joint_lower_bounds = model.jnt_range[:, 0][robot_joint_indices]
    joint_upper_bounds = model.jnt_range[:, 1][robot_joint_indices]

    bounds = Bounds(joint_lower_bounds, joint_upper_bounds)

    result = minimize(
        ik_objective,
        initial_guess,
        args=(model, data, target_position),
        method="trust-constr",  # Better for constrained optimization
        bounds=bounds,
        options={"disp": True}
    )

    return result.x

def calculate_initial_joint_angles(initial_observation):
    """
    Calculates initial joint angles for the demonstration by computing IK for a target
    end-effector position based on the object's initial position.
    """
    obj_xyz = initial_observation[:3]
    # Raise the end-effector slightly above the object
    ee_xyz = obj_xyz
    ee_xyz[2] += 0.05 # 5 cm above the object

    # Use the current robot joint angles as the initial guess.
    initial_guess = data.qpos[robot_joint_indices].copy()

    # Solve for joint angles using the custom IK solver.
    joint_angles = calculate_inverse_kinematics(model, data, ee_xyz, initial_guess[:7])
    gripper_closeness = 0
    action = np.concatenate([joint_angles, [gripper_closeness]])
    return action

def demonstration_policy(observation):
    """
    Provides a demonstration policy action based on the current observation.
    When not in a pushing position, the policy first computes initial joint angles.
    Afterwards, it computes a new action to gently push the object.
    """
    global inPushingPosition
    global initial_joint_angles

    if not inPushingPosition:
        print("initial position not taken yet!")
        if initial_joint_angles is None:
            initial_joint_angles = calculate_initial_joint_angles(observation)
            print("target: " , forward_kinematics(model, data, initial_joint_angles))
            print("cube: ", observation[:3])
            print("initial joint angles: ", initial_joint_angles)
            return initial_joint_angles

        current_joint_angles = data.qpos[robot_joint_indices]
        print("current joint angles: ", current_joint_angles)
        if np.linalg.norm(current_joint_angles - initial_joint_angles[:7]) < 0.001:
            inPushingPosition = True

        return initial_joint_angles

    print("taken initial position successfully!")

    # If already in pushing position, compute a new target position by moving
    # slightly in the direction from the robot base toward the object.
    obj_xyz = observation[:3]
    movement_direction = obj_xyz - robot_base
    movement_direction /= np.linalg.norm(movement_direction)
    movement_direction *= 0.05

    target_position = obj_xyz + movement_direction
    # Use the previously solved joint angles as the initial guess.
    joint_angles = calculate_inverse_kinematics(model, data, target_position, initial_joint_angles[:7])
    gripper_closeness = 0

    action = np.concatenate([joint_angles, [gripper_closeness]])
    return action

if __name__ == "__main__":
    env = PushDemonstratorEnv(render_mode="human")
    env.reset()
    # Start with a random action (this is only for initialization).
    action = env.action_space.sample()
    for _ in range(1000):
        observation, _, _, _, _ = env.step(action)
        action = demonstration_policy(observation)