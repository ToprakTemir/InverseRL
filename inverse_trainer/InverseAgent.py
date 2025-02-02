from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
import gymnasium as gym
import minari
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from torch.distributions import MultivariateNormal

from StateEvaluator import StateEvaluator
from StateCreator import StateCreator

from gymnasium.envs.registration import register
register(
    id="CustomPusher-v0",
    entry_point="environments.CustomPusherEnv:CustomPusherEnv",
    max_episode_steps=300,
)
register(
    id="CustomPuller-v0",
    entry_point="environments.CustomPullerEnv:CustomPullerEnv",
    max_episode_steps=300,
)
register(
    id="InverseTrainerEnvironment-v0",
    entry_point="inverse_trainer.InverseTrainerEnv:InverseTrainerEnv",
    max_episode_steps=300,
)

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
                 state_dim,
                 action_dim,
                 dataset : minari.MinariDataset,
                 inverse_trainer_environment,
                 forward_model = None,
                 ):
        super(InverseAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dataset = dataset
        self.forward_model = forward_model
        self.inverse_trainer_environment = inverse_trainer_environment

        self.state_evaluator = StateEvaluator(state_dim)
        self.state_creator = StateCreator(state_dim)

        self.inverse_model: PPO | None = None

        # HYPERPARAMETERS
        self.lr = 0.001
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.num_epochs_for_state_networks = dataset.total_episodes
        self.batch_size = self.num_epochs_for_state_networks // 5
        self.num_steps_for_inverse_skill = 300_000_0

        self.classifier_loss_coeff = 1
        self.creator_loss_coeff = 1
        self.cycle_consistency_loss_coeff = 1

    def load_state_evaluator(self, state_evaluator):
        self.state_evaluator = state_evaluator

    def load_state_creator(self, state_creator):
        self.state_creator = state_creator

    def train_state_networks(self):

        # GET THE INITIAL AND FINAL STATES FROM DATASET

        num_episodes = self.dataset.total_episodes

        initial_states = np.zeros((num_episodes, state_dim))
        uncertain_states = np.zeros((2*num_episodes, state_dim))
        final_states = np.zeros((num_episodes, state_dim))

        episodes = self.dataset.iterate_episodes()
        i = 0
        for episode in episodes:
            demo = episode.observations
            initial_states[i] = demo[0]
            final_states[i] = demo[-1]

            j1 = int(np.random.uniform(len(demo) // 10, 9 * len(demo) // 10))
            j2 = int(np.random.uniform(len(demo) // 10, 9 * len(demo) // 10))

            uncertain_states[2*i] = demo[j1]
            uncertain_states[2*i+1] = demo[j2]
            i += 1

        # label initial states with -1, uncertain states with 0, final states with 1
        initial_states = np.concatenate((initial_states, -np.ones((num_episodes, 1))), axis=1)
        uncertain_states = np.concatenate((uncertain_states, np.zeros((2*num_episodes, 1))), axis=1)
        final_states = np.concatenate((final_states, np.ones((num_episodes, 1))), axis=1)

        labeled_states = np.concatenate((initial_states, uncertain_states, final_states))
        np.random.shuffle(labeled_states)


        # TRAINING

        for epoch in range(self.num_epochs_for_state_networks):
            indices = np.random.choice(labeled_states.shape[0], self.batch_size)
            batch = labeled_states[indices]

            states = torch.tensor(batch[:, :-1], dtype=torch.float32)
            states += torch.normal(mean=0, std=0.04, size=states.shape) # add noise for robustness

            labels = torch.tensor(batch[:, -1], dtype=torch.float32)
            labels = labels.unsqueeze(1)

            # STATE EVALUATOR LOSS
            output_labels = self.state_evaluator(states)
            state_evaluator_loss = self.mse_loss(output_labels, labels)

            # STATE CREATOR LOSS
            mu, L, D = self.state_creator(labels)
            Sigma = L @ L.transpose(-2, -1) + torch.diag_embed(D)

            # use the log likelihood of the original states as the loss
            try:
                distribution = MultivariateNormal(mu, covariance_matrix=Sigma)
                state_creator_loss = -distribution.log_prob(states).mean()
            except Exception as e:
                print(f"Failed to compute distribution log_prob at epoch {epoch}: {e}")
                continue

            # CYCLE CONSISTENCY LOSS
            # Reconstructed State: StateCreator(StateEvaluator(states))

            reconstructed_labels = self.state_evaluator(self.state_creator.sample(labels))
            reconstructed_states = self.state_creator.sample(self.state_evaluator(states))
            cycle_consistency_loss = (self.mse_loss(reconstructed_labels, labels) + self.mse_loss(reconstructed_states, states)) / 2

            total_loss = (self.classifier_loss_coeff * state_evaluator_loss +
                          self.creator_loss_coeff * state_creator_loss +
                          self.cycle_consistency_loss_coeff * cycle_consistency_loss)


            # BACKWARDS PASS
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


    def save_state_networks(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        torch.save(self.state_evaluator.state_dict(), f"./models/state_evaluators/state_evaluator_{time}.pth")
        torch.save(self.state_creator.state_dict(), f"./models/state_creators/state_creator_{time}.pth")

    def train_inverse_skill(self):
        total_timesteps = self.num_steps_for_inverse_skill
        env = self.inverse_trainer_environment

        # inverse_model = PPO("MlpPolicy", env, verbose=1)
        inverse_model = PPO.load("/Users/toprak/InverseRL/inverse_trainer/models/inverse_model_01.16-13:25", env=env, verbose=1)

        if self.forward_model is not None:
            inverse_model.set_parameters(self.forward_model.get_parameters()) # IMPORTANT: it should derive an initial inverse policy from the forward model, not directly set the forward model as the initial policy

        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps//10, save_path="./models/inverse_model_logs/")
        eval_callback = EvalCallback(env, best_model_save_path="./models/inverse_model_logs/", log_path="./models/inverse_model_logs/", eval_freq=total_timesteps//100)

        callback = [checkpoint_callback, eval_callback]
        inverse_model.learn(total_timesteps=total_timesteps, callback=callback)

        self.inverse_model = inverse_model

    def save_inverse_skill(self):
        time = datetime.now().strftime('%m.%d-%H:%M')
        self.inverse_model.save(f"./models/inverse_model_{time}")

    def load_inverse_skill(self, inverse_model):
        self.inverse_model = inverse_model



if __name__ == "__main__":

    # initializations
    state_dim = 23
    action_dim = 7
    dataset = minari.load_dataset("pusher_demo_R08_large-v0")

    # forward_model_path = "/Users/toprak/InverseRL/learn_push_skill/models/default_robot_trained/best_pusher.zip"
    # forward_model = PPO.load(forward_model_path)

    inverse_agent = InverseAgent(state_dim, action_dim, dataset, None, forward_model=None) # the inverse trainer env will be set after the training of the state networks

    # train the state networks
    # inverse_agent.train_state_networks()
    # inverse_agent.save_state_networks()

    # Or instead load already trained and tested state networks
    state_creator = StateCreator(state_dim)
    state_evaluator = StateEvaluator(state_dim)
    state_creator.load_state_dict(torch.load("./models/state_creators/state_creator_01.16-06:22.pth"))
    state_evaluator.load_state_dict(torch.load("./models/state_evaluators/state_evaluator_01.16-06:22.pth"))
    inverse_agent.load_state_creator(state_creator)
    inverse_agent.load_state_evaluator(state_evaluator)

    # construct the inverse trainer environment, which is dependent on the state networks
    state_evaluator = inverse_agent.state_evaluator
    state_creator = inverse_agent.state_creator
    trainer_env = gym.make("InverseTrainerEnvironment-v0", state_evaluator=state_evaluator, state_creator=state_creator)
    inverse_agent.inverse_trainer_environment = trainer_env

    # train with the inverse trainer environment
    inverse_agent.train_inverse_skill()
    inverse_agent.save_inverse_skill()



