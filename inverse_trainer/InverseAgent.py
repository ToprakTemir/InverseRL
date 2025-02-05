import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minari import MinariDataset
from stable_baselines3 import PPO
import minari
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv

from StateEvaluator import StateEvaluator
from InverseTrainerEnv import InverseTrainerEnv


class InverseAgent(nn.Module):
    """
    InverseAgent is trained to do the inverse action of the skill.

    - It first trains the StateEvaluator on the skill, demanding demonstrations until it is trained sufficiently.
    - Concurrently, it trains the StateCreator to be the inverse function of the StateEvaluator.
    - It then learns the inverse skill using the StateCreator and StateEvaluator, detailed below:
        It takes in the final state of the skill, and outputs the action that would take the environment to the initial state.
        It achieves this by sampling final states using StateCreator, and receives reward by some metric of how much it got closer to the initial state, which is judged by the StateClassifier.
    """

    def __init__(self,
                 dataset : MinariDataset,
                 ):
        super(InverseAgent, self).__init__()

        self.dataset = dataset
        self.state_dim = dataset.observation_space.shape[0]
        self.action_dim = dataset.action_space.shape[0]

        self.state_evaluator = StateEvaluator(self.state_dim)
        self.state_evaluator_trained = False

        self.inverse_trainer_environment: InverseTrainerEnv | None = None
        self.inverse_model: PPO | None = None

        # HYPERPARAMETERS
        self.lr = 0.001
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # TODO: tune these parameters
        self.total_steps = 1_000_000
        # self.batch_size = 128

        self.num_steps_for_inverse_skill = 30_000_000

        self.classifier_loss_coeff = 1
        self.creator_loss_coeff = 1
        self.cycle_consistency_loss_coeff = 1


    def train_state_evaluator(self):
        """
        trains the state evaluator to predict the timestamp of a given point in the trajectory
        """

        # TODO: ask: right now, the inverse RL agent will try to get to a state so that its observation will be given 0 (initial state) by state evaluator
        # but state evaluator is only trained by the demonstrations, what if the RL agent gets to a state that is not in the demonstrations at all,
        # and state evaluator gives a random point, possibly close to 0? Then RL agent will get a reward for that, which is not what we want.

        t0 = datetime.now()
        time_id = datetime.now().strftime('%m.%d-%H:%M')
        differences_log_path = f"./logs/state_evaluator_differences_{time_id}.npy"
        difference_logs = np.zeros(self.total_steps)

        for i in range(self.total_steps):
            episode = next(iter(self.dataset.sample_episodes(1)))
            points = episode.observations

            point_idx = np.random.randint(0, len(points))
            point = torch.tensor(points[point_idx], dtype=torch.float32)

            predicted_timestamp = self.state_evaluator(point)
            real_timestamp = torch.tensor(point_idx / len(points), dtype=torch.float32)
            loss = self.mse_loss(predicted_timestamp, real_timestamp)

            difference_logs[i] = torch.abs(predicted_timestamp - real_timestamp)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 1000 == 0:
                print(f"step: {i}, loss: {loss}, time: {datetime.now() - t0}")
            if i % 1000 == 0:
                np.save(differences_log_path, difference_logs)

        self.state_evaluator_trained = True



    def save_state_evaluator(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        torch.save(self.state_evaluator.state_dict(), f"./models/state_evaluators/state_evaluator_{time}.pth")

    def load_state_evaluator(self, state_evaluator_path):
        weights = torch.load(state_evaluator_path)
        self.state_evaluator.load_state_dict(weights)
        self.state_evaluator_trained = True

    def create_inverse_RL_environment(self):
        """
        Creates the RL environment that will teach the inverse skill using the trained state evaluator.
        """
        if not self.state_evaluator_trained:
            raise Exception("State evaluator is not trained yet.")

        simulation_environment = self.dataset.recover_environment().unwrapped
        trainer_env = InverseTrainerEnv(self.state_evaluator, self.dataset, simulation_environment)
        self.inverse_trainer_environment = trainer_env
        return trainer_env

    def make_env(self):
        return self.inverse_trainer_environment

    def train_inverse_model(self):

        self.create_inverse_RL_environment()

        num_envs = multiprocessing.cpu_count()
        env = SubprocVecEnv([self.make_env for _ in range(num_envs)])

        inverse_model = PPO("MlpPolicy", env=env, verbose=1)

        total_timesteps = self.num_steps_for_inverse_skill
        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps//100, save_path="./models/inverse_model_logs/")
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=total_timesteps//100, verbose=1)
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./models/inverse_model_logs/",
            log_path="./models/inverse_model_logs/",
            eval_freq=total_timesteps // 10_000,
            callback_after_eval=stop_callback
        )
        callback = [checkpoint_callback, eval_callback]
        inverse_model.learn(total_timesteps=total_timesteps, callback=callback)

        self.inverse_model = inverse_model

    def save_inverse_model(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        self.inverse_model.save(f"./models/inverse_model_{time}")


if __name__ == "__main__":

    dataset = minari.load_dataset("xarm_push_10k-v0")

    inverse_agent = InverseAgent(dataset)

    # inverse_agent.train_state_evaluator()
    # inverse_agent.save_state_evaluator()
    inverse_agent.load_state_evaluator("./models/state_evaluators/state_evaluator_02.05-17:45.pth")

    inverse_agent.train_inverse_model()
    inverse_agent.save_inverse_model()
