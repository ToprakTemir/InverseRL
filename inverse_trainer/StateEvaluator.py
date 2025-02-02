import torch
import torch.nn as nn

class StateEvaluator(nn.Module):
    """
    StateClassifier takes is trained on a specific skill, and takes in a state and outputs whether it is the initial
    state of the skill, denoted by -1, or the final state of the skill, denoted by 1, or a random state in between, denoted by 0.
    """

    def __init__(self, state_dim):
        super(StateEvaluator, self).__init__()

        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x)
        return x