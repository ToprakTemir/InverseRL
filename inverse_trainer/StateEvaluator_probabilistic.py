import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEvaluator(nn.Module):
    """
    StateClassifier is trained on a specific skill, and takes in a state and outputs at what percentage the state is close to the final state of the skill.
    """

    def __init__(self, state_dim):
        super(StateEvaluator, self).__init__()

        self.state_dim = state_dim

        self.sequential = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        output = self.sequential(state)
        mean = output[:, 0].unsqueeze(1)
        std = output[:, 1].unsqueeze(1)
        mean = torch.sigmoid(mean) # make sure mean is between 0 and 1
        std = F.softplus(std) + 1e-6
        return torch.stack((mean, std), dim=1)

    def get_distribution(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist
