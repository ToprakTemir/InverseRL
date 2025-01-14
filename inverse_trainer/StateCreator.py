import torch
import torch.nn as nn

class StateCreator(nn.Module):
    """
    StateCreator is trained as the inverse of StateClassifier, and takes in a label, and outputs a state that
    is the initial state of the skill if the label is -1, or the final state of the skill if the label is 1.
    """

    def __init__(self, state_dim):
        super(StateCreator, self).__init__()

        self.state_dim = state_dim

        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)

    def forward(self, label):
        assert -1 <= label <= 1

        x = torch.relu(self.fc1(label))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
