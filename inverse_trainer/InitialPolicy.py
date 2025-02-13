import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class InitialPolicy(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(InitialPolicy, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )


    def forward(self, x):
        return self.sequential(x)

