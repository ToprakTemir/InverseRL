import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import make_proba_distribution


class CustomMLPExtractor(nn.Module):
    """
    Simple MLP feature extractor that outputs separate latent codes
    for the policy (actor) and the value function (critic).
    In practice, you can make this as simple or as complex as you like.
    """
    def __init__(self, features_dim: int, net_arch=(128, 128), activation_fn=nn.ReLU):
        super().__init__()
        # Build a small MLP for actor
        actor_layers = []
        last_layer_dim = features_dim
        for layer_size in net_arch:
            actor_layers.append(nn.Linear(last_layer_dim, layer_size))
            actor_layers.append(activation_fn())
            last_layer_dim = layer_size
        self.actor_mlp = nn.Sequential(*actor_layers)

        # Build a small MLP for critic
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


class CustomPolicy(ActorCriticPolicy):
    """
    A custom policy that manually creates actor/critic nets,
    allowing you to pretrain the actor if desired.
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
            # For advanced usage, you can pass features_extractor_class, etc.
            # or let the default CNN/MLP feature extractor handle it.
            **kwargs
        )

        # If your policy does not rely on a separate "features_extractor",
        # you can interpret self.features_dim as already being the size
        # of the flattened observation. Or define your own features_extractor_class.
        extractor_input_dim = self.features_dim

        # Build a custom MLP that splits into actor & critic latents
        self.mlp_extractor = CustomMLPExtractor(
            features_dim=extractor_input_dim,
            net_arch=net_arch,
            activation_fn=activation_fn
        )

        # Create the action distribution
        self.dist = make_proba_distribution(action_space)

        # Actor head: from actor MLP output -> distribution parameters
        #   e.g. for continuous, output means (and log_stds are separate).
        #   for discrete, output logits.
        if isinstance(action_space, spaces.Box):
            # Continuous
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])
            # We usually store log_std as a separate parameter
            # (SB3 does that inside DiagGaussianDistribution)
            self.log_std = nn.Parameter(th.zeros(action_space.shape[0]))
        else:
            # Discrete
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.n)
            self.log_std = None

        # Critic head: from critic MLP output -> scalar value
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # If you want orthogonal initialization or any custom init, do it here
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # Example orthogonal init
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=th.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Returns actions and value function for given observations.
        This is the method SB3 calls in the rollout collection phase.
        """
        distribution, value = self._get_dist_and_value(obs)
        if deterministic:
            actions = distribution.get_mode()
        else:
            actions = distribution.sample()
        return actions, value

    def _get_dist_and_value(self, obs: th.Tensor):
        features = self.extract_features(obs)  # from ActorCriticPolicy
        pi_latent = self.mlp_extractor.forward_actor(features)
        vf_latent = self.mlp_extractor.forward_critic(features)

        # Create the distribution
        dist = self._get_action_dist_from_latent(pi_latent)
        value = self.value_net(vf_latent)
        return dist, value

    def _get_action_dist_from_latent(self, pi_latent: th.Tensor):
        # For continuous actions, the action_net outputs means, log_std is separate
        if self.log_std is not None:
            mean = self.action_net(pi_latent)
            # Construct diagonal Gaussian from (mean, log_std)
            return self.dist.proba_distribution(mean, self.log_std.exp())
        else:
            # Discrete actions: action_net outputs logits
            logits = self.action_net(pi_latent)
            return self.dist.proba_distribution(logits=logits)

    def _get_value(self, obs: th.Tensor):
        """
        Returns the estimated value (critic) for the given observations.
        """
        features = self.extract_features(obs)
        vf_latent = self.mlp_extractor.forward_critic(features)
        return self.value_net(vf_latent)

    def _predict(self, observation: th.Tensor, deterministic: bool = False):
        """
        This is used by `.predict()` outside of training (e.g., for evaluation).
        """
        actions, _ = self.forward(observation, deterministic=deterministic)
        return actions

    # -------------------------------------------------
    # Example method to do some "pretraining" on the actor
    # before starting the PPO updates. In practice, you'd
    # define your own data loader, loss, etc.
    # -------------------------------------------------
    # def pretrain_actor(self, obs: th.Tensor, target_actions: th.Tensor, optimizer: th.optim.Optimizer, n_epochs=1):
    #     """
    #     Simple example of "pretraining" the actor with some supervised data:
    #      - obs: a batch of observations
    #      - target_actions: the "correct" actions for those observations
    #      - optimizer: an optimizer for the actor’s parameters
    #      - n_epochs: how many epochs to run
    #     """
    #     for _ in range(n_epochs):
    #         optimizer.zero_grad()
    #
    #         # Pass forward
    #         features = self.extract_features(obs)
    #         pi_latent = self.mlp_extractor.forward_actor(features)
    #
    #         # For continuous: assume MSE on the means
    #         if self.log_std is not None:
    #             predicted_means = self.action_net(pi_latent)
    #             loss = F.mse_loss(predicted_means, target_actions)
    #
    #         else:
    #             print("discrete case hasn't implemented yet")
    #             raise NotImplementedError
    #
    #         loss.backward()
    #         optimizer.step()

    def load_pretrained_actor(self, weights: dict):
        """
        Load pretrained weights for the actor network.
        """
        self.load_state_dict(weights, strict=False)