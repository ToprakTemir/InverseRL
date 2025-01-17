import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import MultivariateNormal

class StateCreator(nn.Module):
    """
    StateCreator is trained as the inverse of StateClassifier, and takes in a label, and outputs mu and covariance matrix
    parameters for a normal state distribution.
    The state distribution is supposed to represent the different states that the environment can be in, given the label.

    the state distribution is represented as the mean and the low-rank approximation of the covariance matrix.
    """

    def __init__(self, state_dim, rank=5):
        super(StateCreator, self).__init__()

        self.state_dim = state_dim
        self.rank = rank

        self.fc_mu = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        self.fc_L = nn.Sequential(                  # Low rank matrix L that approximates the covariance matrix
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * rank)
        )
        self.fc_D = nn.Sequential(
            nn.Linear(1, 64), # Diagonal D of the covariance matrix
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, labels):
        mu = self.fc_mu(labels)
        L_flat = self.fc_L(labels)

        D = self.fc_D(labels)
        D = softplus(D)

        L = L_flat.view(-1, self.state_dim, self.rank)
        return mu, L, D

    def sample(self, labels):
        mu, L, D = self.forward(labels)
        Sigma = L @ L.transpose(-2, -1) + torch.diag_embed(D)  # Covariance matrix
        samples = MultivariateNormal(mu, Sigma).sample()
        return samples


