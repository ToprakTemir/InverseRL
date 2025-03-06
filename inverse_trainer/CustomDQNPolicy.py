import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from environments.XarmTableEnvironment import XarmTableEnv


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=6):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Convert everything to CPU numpy or store raw;
        # we’ll move to device later in training
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim=6,
        n_actions=6,
        exploration_strategy="epsilon_greedy",  # or "softmax"
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        softmax_temp=1.0,
        softmax_temp_min=0.1,
        softmax_decay=0.99,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        target_update_freq=1000,
        device=None
    ):
        """
        :param state_dim: dimension of observations (6 or 14).
        :param n_actions: number of discrete actions (6).
        :param exploration_strategy: "epsilon_greedy" or "softmax".
        :param epsilon: initial epsilon (if using epsilon-greedy).
        :param epsilon_min: minimal epsilon.
        :param epsilon_decay: factor by which epsilon is multiplied periodically.
        :param softmax_temp: initial temperature (if using softmax).
        :param softmax_temp_min: minimal temperature.
        :param softmax_decay: factor by which temperature is multiplied periodically.
        :param gamma: discount factor.
        :param lr: learning rate for the Adam optimizer.
        :param batch_size: how many samples to draw from replay buffer each update.
        :param buffer_size: capacity of the replay buffer.
        :param target_update_freq: how often to copy weights to target network.
        :param device: "cuda" or "cpu" (if None, will auto-detect).
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.exploration_strategy = exploration_strategy

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.temp = softmax_temp
        self.temp_min = softmax_temp_min
        self.temp_decay = softmax_decay

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_net = DQN(self.state_dim, self.n_actions).to(self.device)
        self.target_net = DQN(self.state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Counters
        self.learn_step_counter = 0

    def select_action(self, state):
        """
        state: np.array of shape (state_dim,)
        Returns an integer action index in [0..n_actions-1].
        """
        # Turn state into a PyTorch tensor on the correct device
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Evaluate Q(s, a)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)  # shape: [n_actions]

        # Exploration Strategy 1: epsilon-greedy
        if self.exploration_strategy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            else:
                return torch.argmax(q_values).item()

        # Exploration Strategy 2: softmax (Boltzmann) sampling
        elif self.exploration_strategy == "softmax":
            # The higher the temperature, the more uniform the distribution
            # The lower the temperature, the more greedy it behaves
            scaled_Q = q_values / max(self.temp, 1e-8)  # Avoid div by zero
            probs = torch.softmax(scaled_Q, dim=-1)
            # Sample an action according to the probabilities
            action = torch.multinomial(probs, 1).item()
            return action

        else:
            raise ValueError("Unknown exploration_strategy")

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        # If not enough data in buffer, skip
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to PyTorch
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute current Q(s,a)
        q_values = self.q_net(states_t)
        q_selected = q_values.gather(1, actions_t).squeeze(-1)  # shape: [batch_size]

        # Double DQN target
        with torch.no_grad():
            # argmax action using online network
            next_q_values = self.q_net(next_states_t)
            next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)  # shape: [batch_size, 1]

            # Evaluate those actions using target network
            target_q_values = self.target_net(next_states_t)
            target_q_selected = target_q_values.gather(1, next_actions).squeeze(-1)

            # if done, no future reward
            td_target = rewards_t + self.gamma * target_q_selected * (1 - dones_t)

        # Compute loss
        loss = nn.MSELoss()(q_selected, td_target)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay exploration
        self._decay_exploration()

    def _decay_exploration(self):
        # For epsilon-greedy
        if self.exploration_strategy == "epsilon_greedy":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # For softmax
        elif self.exploration_strategy == "softmax":
            self.temp = max(self.temp_min, self.temp * self.temp_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

# ----------------------------------------------------------
# 4. Example usage / training loop
# ----------------------------------------------------------
if __name__ == "__main__":
    # Create the environment with 6D obs
    # (ensuring that inside the XarmTableEnv __init__, we set self.observation_dim=6)
    env = XarmTableEnv(
        xml_file="xarm7_tabletop.xml",
        control_option="discrete_ee_pos",
        frame_skip=5,
        render_mode=None
    )

    # By default, that environment returns a 14D observation, so confirm we pick the 6D branch
    # e.g., you might do:
    env.observation_dim = 6  # Force 6D observation if not already set

    agent = DQNAgent(
        state_dim=6,             # 6D obs
        n_actions=6,            # +x, -x, +y, -y, close, open
        exploration_strategy="softmax",  # or "epsilon_greedy"
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        softmax_temp=1.0,
        softmax_temp_min=0.1,
        softmax_decay=0.99,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=20000,
        target_update_freq=1000
    )

    n_episodes = 500
    max_steps_per_ep = 200  # or whichever max steps

    for episode in range(n_episodes):
        state = env.reset()  # shape (6,) if observation_dim=6
        done = False
        ep_reward = 0.0

        for t in range(max_steps_per_ep):
            action_idx = agent.select_action(state)

            one_hot = np.zeros(agent.n_actions, dtype=np.float32)
            one_hot[action_idx] = 1

            next_state, reward, done1, done2, _ = env.step(one_hot)
            done = (done1 or done2)

            agent.store_transition(state, action_idx, reward, next_state, done)

            agent.update()

            state = next_state
            ep_reward += reward
            if done:
                break

        print(f"Episode {episode} | Reward = {ep_reward}")

    agent.save("stochastic_dqn_xarm.pth")

    # You can later load via:
    # agent.load("stochastic_dqn_xarm.pth")