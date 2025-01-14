import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class InverseAgent(nn.Module):
    """
    InverseAgent is trained to do the inverse action of the skill.

    - It first trains the StateEvaluator on the skill, demanding demonstrations until it is trained sufficiently.
    - Concurrently, it trains the StateCreator to be the inverse function of the StateEvaluator.
    - It then learns the inverse skill using the StateCreator and StateEvaluator, detailed below:
        It takes in the final state of the skill, and outputs the action that would take the environment to the initial state.
        It achieves this by sampling final states using StateCreator, and receives reward by some metric of how much it got closer to the initial state, which is judged by the StateClassifier.
    """

    def __init__(self, state_dim, action_dim, state_evaluator, state_creator):
        super(InverseAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_evaluator = state_evaluator
        self.state_creator = state_creator

        # HYPERPARAMETERS
        lr = 0.001

        self.optimizer = optim.Adam(self.parameters(), lr=lr)







