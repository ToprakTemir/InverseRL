from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments.XarmTableEnvironment import XarmTableEnv


class PurePPOEnv(gym.Wrapper):
    def __init__(self, env: XarmTableEnv):
        super().__init__(env)
        self.max_episode_steps = 500
        self.episode_steps = 0

    def reset(self, **kwargs):
        self.episode_steps = 0

        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()

        R = 0.8
        MIN_ANGLE = np.pi/6
        angle = np.random.uniform(MIN_ANGLE, np.pi - MIN_ANGLE)
        robot_base_xy = np.array([0, -1])

        # obj_x = robot_base_xy[0] + R * np.cos(angle)
        # obj_y = robot_base_xy[1] + R * np.sin(angle)

        noise = np.random.normal(0, 0.05, 2)
        obj_x = -0.1 + noise[0]  # WARNING: TEST
        obj_y = -0.3 + noise[1]  # WARNING: TEST

        qpos[:3] = [obj_x, obj_y, -0.01]
        qpos[3:7] = [1, 0, 0, 0]

        self.env.set_state(qpos, qvel)

        return self.env._get_obs(), {}

    def step(self, action):

        robot_base = [0, -1, 0]
        obj_to_base = np.linalg.norm(self.env.get_body_com("object") - robot_base)
        ee_to_obj = np.linalg.norm(self.env.get_ee_pos() - self.env.get_body_com("object"))

        obj_to_base_rew = 1 / obj_to_base
        ee_to_obj_rew = 1 / ee_to_obj
        reward = obj_to_base_rew + ee_to_obj_rew
        info = {"obj_to_base_rew": obj_to_base_rew, "ee_to_obj_rew": ee_to_obj_rew}

        obs, _, terminated, truncated, _ = self.env.step(action)

        if self.max_episode_steps is not None:
            self.episode_steps += 1
            if self.episode_steps >= self.max_episode_steps:
                truncated = True

        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------

def make_env():
    env = XarmTableEnv(control_option="ee_pos", max_episode_steps=500)
    env = PurePPOEnv(env)
    return env

def train(model_dir):
    env = SubprocVecEnv([make_env for _ in range(8)])

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    save_freq = 64_000
    report_freq = 1000

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_dir)
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=save_freq, verbose=1)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=report_freq,
        callback_after_eval=stop_callback
    )
    callback = [checkpoint_callback, eval_callback]
    model.learn(total_timesteps=80_000_000, callback=callback)

if __name__ == "__main__":
    time = datetime.now().strftime("%m.%d-%H:%M")
    save_dir = f"logs/model_logs/pure_PPOs/{time}"
    train(model_dir=save_dir)

    # -- TEST --
    # time = "03.06-03:42"
    #
    # # option = "latest"
    # option = "best"
    # if option == "best":
    #     model = PPO.load(f"./model_logs/pure_PPOs/{time}/best_model.zip")
    # else:
    #     model = PPO.load(f"./model_logs/pure_PPOs/{time}/28173645918276.zip")
    #
    #
    # while True:
    #     obs, _ = env.reset()
    #     done = False
    #     while not done:
    #         action = model.predict(obs)
    #         obs, _, done1, done2, _ = env.step(action)
    #         done = done1 or done2