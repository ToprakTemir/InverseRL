import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import minari
from minari import MinariDataset
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv
from tensorflow.python.keras.utils.version_utils import training

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv
from CustomPolicy import CustomPolicy
from environments.XarmTableEnvironment import XarmTableEnv

from plot_logs import plot_initial_policy_guesses_compared_to_reverse_trajectory

from gymnasium.envs.registration import register

register(
    id="InverseTrainerEnv-v0",
    entry_point="inverse_trainer.InverseTrainerEnv:InverseTrainerEnv",
    max_episode_steps=300,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


class InverseAgent(nn.Module):

    def __init__(self,
                 env: MujocoEnv,
                 dataset: MinariDataset,
                 validation_dataset: MinariDataset = None,
                 non_robot_indices_in_obs: list = None
                 ):
        super(InverseAgent, self).__init__()

        self.dataset = dataset
        self.state_dim = dataset.observation_space.shape[0]
        self.action_dim = dataset.action_space.shape[0]

        if env is not None:
            self.env = env
        else:
            self.env = dataset.recover_environment().unwrapped

        self.validation_dataset = validation_dataset
        self.validation_error_state_evaluator = np.inf
        self.validation_error_initial_policy = np.inf

        self.non_robot_indices_in_obs = non_robot_indices_in_obs

        self.state_evaluator = None
        self.state_evaluator_trained = False
        self.initial_policy = None
        self.initial_policy_trained = False

        self.state_evaluator_optimizer = None
        self.initial_policy_optimizer = None

        self.inverse_trainer_environment: InverseTrainerEnv | None = None
        self.inverse_model: PPO | None = None

        # HYPERPARAMETERS
        self.lr = 0.001
        self.mse_loss = nn.MSELoss().to(device)

        # total_steps_in_dataset = sum([len(ep.observations) for ep in self.dataset.iterate_episodes()])
        self.batch_size = 32

        self.num_epochs_for_state_evaluator = 1_600_000 // self.batch_size
        self.num_epochs_for_initial_policy = 32_000_000 // self.batch_size

        self.total_steps_for_inverse_skill = 10_000_000


    def remove_robot_indices(self, obs) -> np.ndarray:
        """
        only uses the object indices in the observation, removing the robot indices
        """
        if self.non_robot_indices_in_obs is None:
            # print("object_indices_in_obs is None, returning the whole observation")
            return obs
        else:
            return obs[self.non_robot_indices_in_obs]


    def create_state_evaluator(self, state_evaluator_path=None, device=device):
        # Create a state evaluator
        if self.non_robot_indices_in_obs is None:
            print("object_indices_in_obs is None, using the whole observation for StateEvaluator")
            self.state_evaluator = StateEvaluator(self.state_dim).to(device)
        else:
            self.state_evaluator = StateEvaluator(len(self.non_robot_indices_in_obs)).to(device)

        # Load weights if path is specified
        if state_evaluator_path is not None:
            if device == torch.device('cpu'):
                weights = torch.load(state_evaluator_path, map_location=torch.device('cpu'))
            else:
                weights = torch.load(state_evaluator_path)
            self.state_evaluator.load_state_dict(weights)
            self.state_evaluator_trained = True


    def train_state_evaluator(self, load_from_path=None, device=None):
        """
        trains the state evaluator to predict the timestamp of a given point in the trajectory
        """

        # TODO: right now, the inverse RL agent will try to get to a state so that its observation will be given 0 (initial state) by state evaluator
        # but state evaluator is only trained by the demonstrations, what if the RL agent gets to a state that is not in the demonstrations at all,
        # and state evaluator gives a random point, possibly close to 0? Then RL agent will get a reward for that, which is not what we want.

        self.create_state_evaluator(state_evaluator_path=load_from_path, device=device)

        if self.state_evaluator_trained:
            print("state evaluator is already trained")
            return

        optimizer = optim.Adam(self.state_evaluator.parameters(), lr=self.lr)
        self.state_evaluator_optimizer = optimizer

        t0 = datetime.now()
        time_id = datetime.now().strftime('%m.%d-%H:%M')
        log_path = f"./logs/state_evaluator_differences_{time_id}.npy"
        training_logs = []
        model_save_path = f"./models/state_evaluators/state_evaluator_{time_id}.pth"
        best_model_path = f"./models/state_evaluators/best_state_evaluator_{time_id}.pth"

        for i in range(self.num_epochs_for_state_evaluator):
            indexes = np.random.choice(self.dataset.episode_indices, size=self.batch_size, replace=True)
            episodes = self.dataset.iterate_episodes(indexes)

            points_list = []  # To hold the predictions
            targets_list = []  # To hold the corresponding targets
            for ep in episodes:
                obs_idx = np.random.randint(0, len(ep.observations))
                obs = ep.observations[obs_idx]
                obs = self.remove_robot_indices(obs)
                obs = torch.tensor(obs, dtype=torch.float32, device=device)
                points_list.append(obs)
                targets_list.append(torch.tensor(obs_idx / len(ep.observations)))

            batch_points = torch.stack(points_list).to(device)
            batch_targets = torch.stack(targets_list).to(device)

            predicted_timestamps = self.state_evaluator(batch_points).squeeze(-1)

            mse_loss = self.mse_loss(predicted_timestamps, batch_targets)
            total_variation_loss = torch.mean(torch.abs(predicted_timestamps[1:] - predicted_timestamps[:-1]))

            loss = mse_loss + 0.0 * total_variation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- validation and saving best model ---

            if i % 100 == 0 and self.validation_dataset is not None:
                with torch.no_grad():
                    current_error = 0
                    for j in range(100):
                        ep = list(self.validation_dataset.sample_episodes(1))[0]
                        obs_idx = np.random.randint(0, len(ep.observations))
                        obs = ep.observations[obs_idx]
                        obs = self.remove_robot_indices(obs)
                        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        predicted_timestamp = self.state_evaluator(obs).squeeze(-1)
                        current_error += self.mse_loss(predicted_timestamp, torch.tensor(ep.observations[obs_idx] / len(ep.observations), dtype=torch.float32, device=device))

                    if current_error < self.validation_error_state_evaluator:
                        self.validation_error_state_evaluator = current_error
                        self.save_state_evaluator(best_model_path)
                        print(f"new validation best. error: {self.validation_error_state_evaluator}")
                    else:
                        print(f"validation error: {current_error}")


            # --- logging ---
            training_logs.append({
                "step": i,
                "difference": abs(predicted_timestamps[0] - batch_targets[0]),
                "predicted": predicted_timestamps[0],
                "actual": batch_targets[0],
            })

            if i % 1000 == 0:
                print(f"step: {i}, time: {datetime.now() - t0}")
                print(
                    f"prediction: {predicted_timestamps[0]}, actual: {batch_targets[0]}, difference: {abs(predicted_timestamps[0] - batch_targets[0])}")
                print()
                np.save(log_path, training_logs)
                self.save_state_evaluator(path=model_save_path)

        self.state_evaluator_trained = True

    def save_state_evaluator(self, path=None):
        time = datetime.now().strftime('%m.%d-%H:%M')
        if path is None:
            torch.save(self.state_evaluator.state_dict(), f"./models/state_evaluators/state_evaluator_{time}.pth")
        else:
            torch.save(self.state_evaluator.state_dict(), path)

    def create_initial_policy(self, initial_policy_path=None, device=device):
        self.initial_policy = CustomPolicy(self.env.observation_space, self.env.action_space).to(device)
        if initial_policy_path is not None:
            self.initial_policy.load_state_dict(torch.load(initial_policy_path))
            self.initial_policy_trained = True

    def pretrain_policy(self, load_from_path=None, device=device):
        """
        Create an initial policy by training the robot to do the demonstrations rewound in time

        initial_policy input: observation, output: joint angles
        """

        self.create_initial_policy(initial_policy_path=load_from_path, device=device)

        if self.initial_policy_trained:
            print("initial policy is already trained")
            return

        optimizer = optim.Adam(self.initial_policy.parameters(), lr=self.lr)
        self.initial_policy_optimizer = optimizer

        t0 = datetime.now()
        time_id = datetime.now().strftime('%m.%d-%H:%M')
        log_path = f"./logs/initial_policy_differences_{time_id}.npy"
        training_logs = []

        # loss_option = "MSE"
        loss_option = "log_prob"
        model_save_path = f"./models/initial_policies/initial_policy_{loss_option}_{time_id}.pth"
        best_model_path = f"./models/initial_policies/best_initial_policy_{loss_option}_{time_id}.pth"

        # obs = list(self.dataset.iterate_episodes())[0].observations[0]
        # target = torch.tensor([0, -1, 0], dtype=torch.float32, device=device)
        # obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        # target = target.unsqueeze(0)
        # for i in range(100):
        #     dist, _ = self.initial_policy._get_dist_and_value(obs)
        #     log_prob = dist.log_prob(target)
        #     loss = -log_prob.mean()
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     print(f"dist mean: {dist.mean}, target: {target}, loss = {loss.item()}")


        for i in range(self.num_epochs_for_initial_policy):
            episode = list(self.dataset.sample_episodes(1))[0]
            indexes = np.random.choice(len(episode.observations), size=self.batch_size, replace=True)

            states = episode.observations[indexes]
            target_actions = episode.actions[indexes - 10]

            states = torch.tensor(states, dtype=torch.float32, device=device)
            target_actions = torch.tensor(target_actions, dtype=torch.float32, device=device)

            if loss_option == "log_prob":
                dist, _ = self.initial_policy._get_dist_and_value(states)
                log_prob = dist.log_prob(target_actions)
                loss = -log_prob.mean()
            elif loss_option == "MSE":
                predicted_actions, values, log_probs = self.initial_policy(states)
                if target_actions.dim() == 1:
                    target_actions = target_actions.unsqueeze(-1)
                loss = self.mse_loss(predicted_actions, target_actions)
            else:
                raise ValueError("loss_option should be log_prob or MSE")

            optimizer.zero_grad()
            # for name, param in self.initial_policy.named_parameters():
            #     if param.grad is None:
            #         print(f"param {name} has no grad")
            #     else:
            #         print(f"param {name} grad norm: {param.grad.norm()}")
            loss.backward()
            optimizer.step()

            # --- validation and saving best model ---

            if i % 100 == 0 and self.validation_dataset is not None:
                with torch.no_grad():
                    current_error = 0
                    log_prob_loss_sum = 0
                    mse_loss_sum = 0
                    for j in range(100):
                        ep = list(self.validation_dataset.sample_episodes(1))[0]
                        idx = np.random.randint(0, len(ep.observations))
                        prev_action, state = ep.actions[idx - 1], ep.observations[idx]
                        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        prev_action = torch.tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)

                        dist, _ = self.initial_policy._get_dist_and_value(state)
                        log_prob = dist.log_prob(prev_action)
                        log_prob_loss = -log_prob.mean()

                        predicted_action, _, _ = self.initial_policy(state)
                        mse_loss = self.mse_loss(predicted_action, prev_action)

                        log_prob_loss_sum += log_prob_loss
                        mse_loss_sum += mse_loss

                        if loss_option == "log_prob":
                            current_error += log_prob_loss
                        else:
                            current_error += mse_loss

                    if current_error < self.validation_error_initial_policy:
                        self.validation_error_initial_policy = current_error
                        self.save_pretrained_policy(best_model_path)
                        print(f"new validation best. error: {self.validation_error_initial_policy}")

                        # LOGGING NEW SAVED MODEL
                        # plot_initial_policy_guesses_compared_to_reverse_trajectory(path=best_model_path)

                    else:
                        print(f"validation error: {current_error}")

            # --- logging ---

            training_logs.append({
                "step": i,
                "loss": loss.item(),
            })

            if i % 1000 == 0:
                print(f"step: {i}, time: {datetime.now() - t0}")
                print(f"prediction: {dist.sample()}, target: {target_actions[0]} loss: {loss.item()}")
                print()
                np.save(log_path, training_logs)
                self.save_pretrained_policy(path=model_save_path)

        self.initial_policy_trained = True

    def save_pretrained_policy(self, path=None):
        time = datetime.now().strftime('%m.%d-%H:%M')
        if path is None:
            torch.save(self.initial_policy.state_dict(), f"./models/initial_policies/initial_policy_{time}.pth")
        else:
            torch.save(self.initial_policy.state_dict(), path)


    def create_inverse_trainer_environment(self, max_episode_steps=1000):
        """
        Creates the RL environment that will teach the inverse skill by controlling the reward using the trained state evaluator.
        """
        assert isinstance(self.env, MujocoEnv), "InverseTrainerEnv should be a MujocoEnv"

        trainer_env = InverseTrainerEnv(self.env, self.state_evaluator, self.dataset, self.non_robot_indices_in_obs)
        trainer_env = TimeLimit(trainer_env, max_episode_steps=max_episode_steps)
        self.inverse_trainer_environment = trainer_env

        return trainer_env

    def train_inverse_model(self):

        if not self.state_evaluator_trained:
            raise Exception("State evaluator is not trained yet.")

        if not self.initial_policy_trained:
            raise Exception("Initial policy is not trained yet.")


        num_envs = multiprocessing.cpu_count()
        print(f"num envs: {num_envs}")
        env = SubprocVecEnv([self.create_inverse_trainer_environment for _ in range(num_envs)])

        inverse_model = PPO(CustomPolicy, env=env, verbose=1, device="cpu")
        inverse_model.policy.load_state_dict(self.initial_policy.state_dict())
        # inverse_model.policy.optimizer.load_state_dict(self.initial_policy_optimizer.state_dict())

        time = datetime.now().strftime('%m.%d-%H:%M')
        model_path = f"./models/inverse_model_logs/{time}"
        os.mkdir(model_path)

        total_timesteps = self.total_steps_for_inverse_skill
        save_freq = 64_000
        report_freq = 1000

        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path)
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=save_freq, verbose=1)
        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_path,
            log_path=model_path,
            eval_freq=report_freq,
            callback_after_eval=stop_callback
        )
        callback = [checkpoint_callback, eval_callback]

        inverse_model.learn(total_timesteps=total_timesteps, callback=callback)
        self.inverse_model = inverse_model

    def save_inverse_model(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        self.inverse_model.save(f"./models/inverse_model_{time}")


if __name__ == "__main__":
    dataset = minari.load_dataset("xarm_push_3d_action_space_closer_1k-v0")
    validation_dataset = minari.load_dataset("xarm_push_3d_action_space_closer_5k-v0")

    # dataset = minari.load_dataset("xarm_push_3d_directly_forward_1-v0")
    # validation_dataset = None

    env = XarmTableEnv(control_option="ee_pos")
    inverse_agent = InverseAgent(env, dataset, validation_dataset=validation_dataset, non_robot_indices_in_obs=[0, 1, 2])

    path = "models/state_evaluators/best_state_evaluator_02.27-04:56.pth"
    # path = None
    inverse_agent.train_state_evaluator(load_from_path=path, device=device)

    path = "./models/initial_policies/best_initial_policy_log_prob_02.27-03:16.pth"
    # path = None
    inverse_agent.pretrain_policy(load_from_path=path, device=device)

    inverse_agent.train_inverse_model()