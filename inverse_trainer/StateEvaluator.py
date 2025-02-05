import torch
import torch.nn as nn

class StateEvaluator(nn.Module):
    """
    StateClassifier is trained on a specific skill, and takes in a state and outputs at what percentage the state is close to the final state of the skill.
    """

    def __init__(self, state_dim):
        super(StateEvaluator, self).__init__()

        self.state_dim = state_dim

        self.forward = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        x = self.forward(state)
        x = torch.sigmoid(x)

