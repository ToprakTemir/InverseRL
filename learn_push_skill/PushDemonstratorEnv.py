import gymnasium as gym
import minari
import numpy as np
import mujoco
from gymnasium import Wrapper
from gymnasium.envs.mujoco import MujocoEnv
from minari import DataCollector
from minari.data_collector import EpisodeBuffer

from environments.XarmTableEnvironment import XarmTableEnv

from gymnasium.envs import register
register(
    id="XarmTableEnv-v0",
    entry_point="environments.XarmTableEnvironment:XarmTableEnv",
    max_episode_steps=2000,
)


class PushDemonstratorEnv(Wrapper):
    """
    A wrapper for XarmTableEnv that modifies the object spawn location and
    provides a push policy function that uses mocap control for the robot to push the object.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, env: XarmTableEnv):
        super().__init__(env)

        # HYPERPARAMETERS
        self.MIN_ANGLE = np.pi / 6
        self.MIN_R = 0.4
        self.MAX_R = 0.8

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()

        # X is to the right, Y is into the screen, Z is up.
        # The square table is centered at (0, 0) with side length 2,
        # and the robot base is at (0, -1).
        R = np.random.uniform(self.MIN_R, self.MAX_R)

        # Randomize the object in a circle around the robot.
        angle = np.random.uniform(self.MIN_ANGLE, np.pi - self.MIN_ANGLE)
        obj_xy = self.env.robot_base_xy + R * np.array([np.cos(angle), np.sin(angle)])

        init_cube_pos = np.concatenate([obj_xy, [-0.01]])
        qpos[:3] = init_cube_pos
        qpos[3:7] = [1, 0, 0, 0]  # cube orientation

        self.env.set_state(qpos, qvel)

        ee_id = self.env.model.body("end_effector").id
        self.env.data.mocap_pos = self.env.data.xpos[ee_id].copy()
        self.env.data.mocap_pos[:, 2] += 0.01
        self.env.data.mocap_quat = [0, 0, 0, 1]
        mujoco.mj_step(self.env.model, self.env.data)
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_forward(self.env.model, self.env.data)

        return self.env._get_obs(), None

    def step(self, action):

        self.env.data.mocap_pos = action[:3]
        self.env.data.mocap_quat = action[3:7]
        self.env.data.ten_length = action[7]

        mujoco.mj_step(self.env.model, self.env.data)
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_forward(self.env.model, self.env.data)

        obs = self.env._get_obs()
        reward = 0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info


    def get_object_position(self):
        return self.data.body("object").xpos.copy()



def generate_push_trajectory(env, speed=0.002, pre_push_offset=0.1, push_distance=1, table_z=-0.03):
    """
    Generates a trajectory (list of action points) for pushing an object away,
    moving at a fixed speed (distance per step).

    The trajectory has two phases:
      1. Approach Phase: Moves from the current end-effector position to a pre-push position
         located behind the object (relative to the push direction).
      2. Push Phase: Moves along the push direction for a specified push distance.

    Each action is an 8D vector:
       [target_x, target_y, target_z, target_quat_x, target_quat_y, target_quat_z, target_quat_w, gripper_closeness]

    Note: We use a fixed orientation (identity)

    Parameters:
      env: The environment instance (assumed to have attributes 'data' and method '_get_obs()').
      speed: The distance to move per step (default 0.01 m per step).
      pre_push_offset: The distance behind the object for the pre-push position.
      push_distance: The distance to push the object.
      table_z: The table's z-coordinate (default 0.0); the end-effector is kept just above it.

    Returns:
      traj_actions: A list of action points (numpy arrays of shape (8,)).
    """
    # Ensure the end-effector stays just above the table.
    desired_z = table_z + 0.000

    # Get the current end-effector position from the mocap (flatten if necessary)
    current_pos = env.data.mocap_pos.copy()
    if current_pos.ndim > 1:
        current_pos = current_pos[0]
    current_pos[2] = desired_z

    # Get the object's position from the observation.
    obs = env._get_obs()
    object_pos = obs[:3].copy()
    object_pos[2] = desired_z

    # Define the robot base position (assumed fixed; adjust if needed)
    robot_base = np.array([0, -1, desired_z])

    # Compute the push direction (from robot base to object, projected on the table)
    push_direction = object_pos - robot_base
    push_direction[2] = 0  # ignore vertical component
    norm = np.linalg.norm(push_direction)
    push_direction = push_direction / norm

    # Compute the pre-push target (a point behind the object along the push direction)
    pre_push_target = object_pos - push_direction * pre_push_offset
    pre_push_target[2] = desired_z

    d0 = np.linalg.norm(object_pos - robot_base)
    robot_reach = 1.17
    push_displacement = max(0, robot_reach - d0)
    final_target = pre_push_target + push_direction * push_displacement
    final_target[2] = desired_z

    # Generate the approach trajectory at fixed speed.
    approach_vec = pre_push_target - current_pos
    approach_distance = np.linalg.norm(approach_vec)
    steps_approach = int(np.ceil(approach_distance / speed))
    traj_positions_approach = [
        current_pos + approach_vec * (i / steps_approach)
        for i in range(1, steps_approach + 1)
    ]

    # Generate the push trajectory at fixed speed.
    push_vec = final_target - pre_push_target
    push_distance_actual = np.linalg.norm(push_vec)
    steps_push = int(np.ceil(push_distance_actual / speed))
    traj_positions_push = [
        pre_push_target + push_vec * (i / steps_push)
        for i in range(1, steps_push + 1)
    ]

    # Combine both trajectories.
    traj_positions = traj_positions_approach + traj_positions_push

    # Use a fixed quaternion (identity) so that orientation is not controlled.
    fixed_quat = np.array([0, 0, 0, 1])
    gripper_closeness = 0  # Modify if necessary

    # Assemble the full 8D action for each trajectory point.
    traj_actions = [
        np.concatenate([pos, fixed_quat, [gripper_closeness]])
        for pos in traj_positions
    ]
    traj_actions = np.array(traj_actions).astype(np.float32)

    return traj_actions


def collect_push_demo(dataset_id, num_demos):

    base_env = XarmTableEnv()
    push_env = PushDemonstratorEnv(base_env)
    data_collector_env = DataCollector(push_env, record_infos=False)

    successful_demo_count = 0
    while successful_demo_count < num_demos:
        print(f"collecting {successful_demo_count}th demo")
        observation, _ = data_collector_env.reset()
        print(f"initial distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")

        traj = generate_push_trajectory(push_env.env)
        step_count = 0
        successful = False
        for action in traj:
            observation, _, _, _, _ = data_collector_env.step(action)
            step_count += 1

        print(f"final distance to robot: {np.linalg.norm(observation[0:2] - [0, -1])}")
        if np.linalg.norm(observation[0:2] - [0, -1]) > 0.9:
            successful = True

        if successful:
            successful_demo_count += 1
            print(
                f"demo successful. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")
        else:
            print(
                f"demo failed. object distance to robot is {np.linalg.norm(observation[0:2] - [0, -1])}, step count is: {step_count}")

        print()

    data_collector_env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="push_trajectory_generator",
        author="Bora Toprak Temir"
    )


if __name__ == "__main__":

    num_demos = 1000
    dataset_id = f"xarm_synthetic_push_50-v0"

    # collect_push_demo(dataset_id, num_demos)

    # FILTERING THE DEMO

    dataset = minari.load_dataset(dataset_id)

    episode_buffers = []
    for episode in dataset.iterate_episodes():
        final_obs = episode.observations[-1]
        if np.linalg.norm(final_obs[0:2] - [0, -1]) > 0.9:
            episode_buffer = EpisodeBuffer(
                observations=episode.observations,
                actions=episode.actions,
                rewards=episode.rewards,
                terminations=episode.terminations,
                truncations=episode.truncations
            )
            episode_buffers.append(episode_buffer)

    print(f"original episode count: {dataset.total_episodes}")
    print(f"filtered episode count: {len(episode_buffers)}")

    dataset_env = gym.make("XarmTableEnv-v0")
    dataset_env = PushDemonstratorEnv(dataset_env)
    minari.delete_dataset(dataset_id)
    minari.create_dataset_from_buffers(
        dataset_id,
        episode_buffers,
        algorithm_name="push_trajectory_generator",
        author="Bora Toprak Temir",
        env=dataset_env.env
    )
    dataset = minari.load_dataset(dataset_id)
    print(f"env_spec after recreation: {dataset.env_spec}")
