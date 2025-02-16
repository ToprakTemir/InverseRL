import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution


class CustomMLPExtractor(nn.Module):
    """
    Simple MLP feature extractor that outputs separate latent codes
    for the policy (actor) and the value function (critic).
    """
    def __init__(self, features_dim: int, net_arch=(128, 128), activation_fn=nn.ReLU):
        super().__init__()
        # Actor MLP
        actor_layers = []
        last_layer_dim = features_dim
        for layer_size in net_arch:
            actor_layers.append(nn.Linear(last_layer_dim, layer_size))
            actor_layers.append(activation_fn())
            last_layer_dim = layer_size
        self.actor_mlp = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = []
        last_layer_dim = features_dim
        for layer_size in net_arch:
            critic_layers.append(nn.Linear(last_layer_dim, layer_size))
            critic_layers.append(activation_fn())
            last_layer_dim = layer_size
        self.critic_mlp = nn.Sequential(*critic_layers)

        # Store the output dimensions
        self.latent_dim_pi = last_layer_dim
        self.latent_dim_vf = last_layer_dim

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.actor_mlp(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.critic_mlp(features)

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


class CustomPolicy(ActorCriticPolicy):
    """
    A custom policy that manually creates actor/critic nets.
    Must return (actions, value, log_prob) when called.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=(128, 128),
        activation_fn=nn.ReLU,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        # Features dim is determined by the parent class's extractor (default MLP/CNN).
        extractor_input_dim = self.features_dim

        # Build custom MLP that splits into actor & critic latents
        self.mlp_extractor = CustomMLPExtractor(
            features_dim=extractor_input_dim,
            net_arch=net_arch,
            activation_fn=activation_fn
        )

        # Create the action distribution
        self.dist = make_proba_distribution(action_space)

        # Actor head
        if isinstance(action_space, spaces.Box):
            # Continuous
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])
            self.log_std = nn.Parameter(th.zeros(action_space.shape[0]))
        else:
            # Discrete
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.n)
            self.log_std = None

        # Critic head
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=th.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Compute the policy’s action, its value estimate, and the log probability
        of the selected action, given observations.
        SB3 expects: actions, values, log_probs
        """
        distribution, value = self._get_dist_and_value(obs)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)
        return actions, value, log_probs

    def _get_dist_and_value(self, obs: th.Tensor):
        # Extract features using the parent class's feature extractor
        features = self.extract_features(obs)
        pi_latent = self.mlp_extractor.forward_actor(features)
        vf_latent = self.mlp_extractor.forward_critic(features)

        dist = self._get_action_dist_from_latent(pi_latent)
        value = self.value_net(vf_latent)
        return dist, value

    def _get_action_dist_from_latent(self, pi_latent: th.Tensor):
        if self.log_std is not None:
            # Continuous actions
            mean = self.action_net(pi_latent)
            return self.dist.proba_distribution(mean, self.log_std.exp())
        else:
            # Discrete actions
            logits = self.action_net(pi_latent)
            return self.dist.proba_distribution(logits=logits)

    def _get_value(self, obs: th.Tensor):
        """
        Returns the estimated value (critic output) for given observations.
        Used internally by SB3 for e.g. rollout buffer calculations.
        """
        features = self.extract_features(obs)
        vf_latent = self.mlp_extractor.forward_critic(features)
        return self.value_net(vf_latent)

    def _predict(self, observation: th.Tensor, deterministic: bool = False):
        """
        Used by `.predict()` for evaluation. Only returns the actions.
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def load_pretrained_weights(self, weights: dict):
        """
        Load pretrained weights for the actor network.
        """
        self.load_state_dict(weights, strict=False)