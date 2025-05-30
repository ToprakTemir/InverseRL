import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
import minari
from minari import MinariDataset
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv

# from StateEvaluator import StateEvaluator
from StateEvaluator_probabilistic import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv

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
                 dataset: MinariDataset,
                 validation_dataset: MinariDataset = None,
                 object_indices_in_obs: list = None
                 ):
        super(InverseAgent, self).__init__()

        self.dataset = dataset
        self.state_dim = dataset.observation_space.shape[0]
        self.action_dim = dataset.action_space.shape[0]

        self.object_indices_in_obs = object_indices_in_obs

        self.validation_dataset = validation_dataset
        self.validation_error = np.inf

        self.state_evaluator = None
        self.state_evaluator_trained = False

        self.inverse_trainer_environment: InverseTrainerEnv | None = None
        self.inverse_model: PPO | None = None

        # HYPERPARAMETERS
        self.lr = 0.001
        self.mse_loss = nn.MSELoss().to(device)

        self.num_steps_for_state_evaluator = 1_000_000
        self.batch_size = 128

        self.num_steps_for_inverse_skill = 30_000_000

        self.classifier_loss_coeff = 1
        self.creator_loss_coeff = 1
        self.cycle_consistency_loss_coeff = 1

    def remove_robot_indices(self, obs) -> np.ndarray:
        """
        only uses the object indices in the observation, removing the robot indices
        """
        if self.object_indices_in_obs is None:
            # print("object_indices_in_obs is None, returning the whole observation")
            return obs
        else:
            return obs[self.object_indices_in_obs]

    def train_state_evaluator(self):
        """
        trains the state evaluator to predict the timestamp of a given point in the trajectory
        """

        if self.object_indices_in_obs is None:
            print("object_indices_in_obs is None, using the whole observation for StateEvaluator")
            self.state_evaluator = StateEvaluator(self.state_dim).to(device)
        else:
            self.state_evaluator = StateEvaluator(len(self.object_indices_in_obs)).to(device)

        optimizer = optim.Adam(self.state_evaluator.parameters(), lr=self.lr)

        t0 = datetime.now()
        time_id = datetime.now().strftime('%m.%d-%H:%M')
        log_path = f"./logs/state_evaluator_differences_{time_id}.npy"
        training_logs = []
        model_save_path = f"./model_logs/state_evaluators/probabilistic_state_evaluator_{time_id}.pth"
        best_model_path = f"./model_logs/state_evaluators/best_probabilistic_state_evaluator_{time_id}.pth"

        for i in range(self.num_steps_for_state_evaluator):
            episodes = list(self.dataset.sample_episodes(self.batch_size))
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

            output = self.state_evaluator(batch_points)
            if torch.isnan(output).any():
                print(f"NaN encountered in state evaluator output at step {i}, skipping this step")
                continue

            mean = output[:, 0]
            std = output[:, 1]
            dist = torch.distributions.Normal(mean, std)

            loss = -dist.log_prob(batch_targets).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- validation and saving best model ---

            if i % 1000 == 0 and self.validation_dataset is not None:
                current_error = 0
                for j in range(100):
                    ep = list(self.validation_dataset.sample_episodes(1))[0]
                    obs_idx = np.random.randint(0, len(ep.observations))
                    obs = ep.observations[obs_idx]
                    obs = self.remove_robot_indices(obs)
                    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    output = self.state_evaluator(obs)
                    if torch.isnan(output).any():
                        print(f"NaN encountered in state evaluator output at validation of step {i}, skipping this step")
                        continue

                    mean = output[:, 0]
                    std = output[:, 1]
                    dist = torch.distributions.Normal(mean, std)
                    current_error += -dist.log_prob(torch.tensor(obs_idx / len(ep.observations))).item()

                if current_error < self.validation_error:
                    self.validation_error = current_error
                    self.save_state_evaluator(path=best_model_path)
                    print(f"new validation best. error: {current_error}")


            # --- logging ---
            training_logs.append({
                "step": i,
                "predicted_mean": mean[0].item(),
                "predicted_std": std[0].item(),
                "actual": batch_targets[0].item(),
                "loss": loss.item()
            })

            if i % 1000 == 0:
                print(f"step: {i}, time: {datetime.now() - t0}")
                print(f"predicted mean: {mean[0].item()}, predicted std: {std[0].item()}, actual: {batch_targets[0].item()}, loss: {loss.item()}")
                print()
                np.save(log_path, training_logs)
                self.save_state_evaluator(path=model_save_path)

        self.state_evaluator_trained = True

    def save_state_evaluator(self, path=None):
        time = datetime.now().strftime('%m.%d-%H:%M')
        if path is None:
            torch.save(self.state_evaluator.state_dict(), f"./model_logs/state_evaluators/state_evaluator_{time}.pth")
        else:
            torch.save(self.state_evaluator.state_dict(), path)

    def load_state_evaluator(self, state_evaluator_path):
        weights = torch.load(state_evaluator_path)
        self.state_evaluator.load_state_dict(weights)
        self.state_evaluator_trained = True

    def create_inverse_RL_environment(self):
        """
        Creates the RL environment that will teach the inverse skill by controlling the reward using the trained state evaluator.
        """
        if not self.state_evaluator_trained:
            raise Exception("State evaluator is not trained yet.")

        simulation_environment = self.dataset.recover_environment().unwrapped

        trainer_env = gym.make(
            "InverseTrainerEnv-v0",
            state_evaluator=self.state_evaluator,
            dataset=self.dataset,
            env=simulation_environment
        )
        self.inverse_trainer_environment = trainer_env
        return trainer_env

    def make_env(self):
        return self.create_inverse_RL_environment()

    def train_inverse_model(self):

        self.create_inverse_RL_environment()

        num_envs = multiprocessing.cpu_count()
        env = SubprocVecEnv([self.make_env for _ in range(num_envs)])

        inverse_model = PPO("MlpPolicy", env=env, verbose=1, device="cpu")

        total_timesteps = self.num_steps_for_inverse_skill
        save_freq = 100_000
        report_freq = 1000

        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="../model_logs/inverse_model_logs/")
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=save_freq, verbose=1)
        eval_callback = EvalCallback(
            env,
            best_model_save_path="../model_logs/inverse_model_logs/",
            log_path="../model_logs/inverse_model_logs/",
            eval_freq=report_freq,
            callback_after_eval=stop_callback
        )
        callback = [checkpoint_callback, eval_callback]
        inverse_model.learn(total_timesteps=total_timesteps, callback=callback)

        self.inverse_model = inverse_model

    def save_inverse_model(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        self.inverse_model.save(f"./model_logs/inverse_model_{time}")


if __name__ == "__main__":
    dataset = minari.load_dataset("xarm_push_only_successful_1k-v0")
    validation_dataset = minari.load_dataset("xarm_push_only_successful_5k-v0")

    inverse_agent = InverseAgent(dataset, validation_dataset=validation_dataset, object_indices_in_obs=[0, 1, 2])

    inverse_agent.train_state_evaluator()
    inverse_agent.save_state_evaluator()
    # inverse_agent.load_state_evaluator("./model_logs/state_evaluators/state_evaluator_02.05-17:45.pth")

    # inverse_agent.train_inverse_model()
    # inverse_agent.save_inverse_model()
