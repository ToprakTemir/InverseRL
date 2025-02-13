import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp


class InitialPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule=lambda _: 3e-4, **kwargs):
        # You can pass a custom net_arch if you wish via kwargs.
        super(InitialPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Define the actor network.
        # Note: self.features_dim is set by the parent class based on your features extractor.
        self.actor = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n if hasattr(action_space, "n") else action_space.shape[0])
        )

        # Define the critic network.
        self.critic = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # (Optional)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs, deterministic=False):

        # Extract features using the parent class extractor.
        features = self.extract_features(obs)

        # Pass features through the critic network to obtain the state value.
        value = self.critic(features)

        # Pass features through the actor network to get action logits.
        logits = self.actor(features)

        # Get the distribution for actions using the helper method from ActorCriticPolicy.
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, value, log_prob