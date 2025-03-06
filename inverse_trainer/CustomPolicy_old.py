from typing import Tuple

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution
from tensorflow.python.framework.func_graph import override_func_graph_name_scope


# class CustomMLPExtractor(nn.Module):
#     """
#     Simple MLP feature extractor that outputs separate latent codes
#     for the policy (actor) and the value function (critic).
#     """
#     def __init__(self, features_dim: int, net_arch=(128, 128), activation_fn=nn.ReLU):
#         super().__init__()
#         # Actor MLP
#         actor_layers = []
#         last_layer_dim = features_dim
#         for layer_size in net_arch:
#             actor_layers.append(nn.Linear(last_layer_dim, layer_size))
#             actor_layers.append(activation_fn())
#             last_layer_dim = layer_size
#         self.actor_mlp = nn.Sequential(*actor_layers)
#
#         # Critic MLP
#         critic_layers = []
#         last_layer_dim = features_dim
#         for layer_size in net_arch:
#             critic_layers.append(nn.Linear(last_layer_dim, layer_size))
#             critic_layers.append(activation_fn())
#             last_layer_dim = layer_size
#         self.critic_mlp = nn.Sequential(*critic_layers)
#
#         # Store the output dimensions
#         self.latent_dim_pi = last_layer_dim
#         self.latent_dim_vf = last_layer_dim
#
#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.actor_mlp(features)
#
#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.critic_mlp(features)
#
#     def forward(self, features: th.Tensor):
#         return self.forward_actor(features), self.forward_critic(features)

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
        use_sde=True,
        net_arch=(128, 128),
        activation_fn=nn.ReLU,
        **kwargs
    ):

        super().__init__(
            observation_space = observation_space,
            action_space = action_space,
            lr_schedule = lr_schedule,
            # We do NOT want to build the default MlpExtractor,
            # so pass net_arch=[], or net_arch=None,
            # and override _build_mlp_extractor() so it won't create anything.
            net_arch=[],
            activation_fn=activation_fn,
            ortho_init=False,
            use_sde=use_sde,
            **kwargs
        )
        #
        # # Features dim is determined by the parent class's extractor (default MLP/CNN).
        # extractor_input_dim = self.features_dim
        #
        # # Build custom MLP that splits into actor & critic latents
        # self.mlp_extractor = CustomMLPExtractor(
        #     features_dim=extractor_input_dim,
        #     net_arch=net_arch,
        #     activation_fn=activation_fn
        # )
        #
        # # Actor head
        # dim = action_space.shape[0]
        # self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, dim)
        # self.covariance_net = nn.Sequential(
        #     nn.Linear(self.mlp_extractor.latent_dim_pi, dim),
        #     nn.Softplus()
        # )
        # # Critic head
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)


    def _build_mlp_extractor(self) -> None:
        """
        We override this method to prevent the parent class from building
        the default MLP extractor. We will build our own custom extractor.
        """
        pass

    def _make_dist(self, mean: th.Tensor, std: th.Tensor) -> None:
        """
        No-op override to avoid building an SB3 distribution object
        like DiagGaussianDistribution. We create our own distribution below.
        """
        return None

    def _build(self, lr_schedule) -> None:
        """
        Called automatically at the end of the parent's __init__.
        Here we can define all submodules, now that self.features_dim is known.
        Then define self.optimizer with the parent's logic.
        """

        obs_dim = self.features_dim
        action_dim = self.action_space.shape[0]

        hidden_sizes = (128, 128)

        actor_layers = []
        last_layer_dim = obs_dim
        for layer_size in hidden_sizes:
            actor_layers.append(nn.Linear(last_layer_dim, layer_size))
            actor_layers.append(nn.ReLU())
            last_layer_dim = layer_size
        actor_layers.append(nn.Linear(last_layer_dim, action_dim))
        self.actor_mlp = nn.Sequential(*actor_layers)

        self.log_std = nn.Parameter(th.zeros(action_dim))

        critic_layers = []
        last_layer_dim = obs_dim
        for layer_size in hidden_sizes:
            critic_layers.append(nn.Linear(last_layer_dim, layer_size))
            critic_layers.append(nn.ReLU())
            last_layer_dim = layer_size
        self.critic_mlp = nn.Sequential(*critic_layers)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # ===========================================
    # The main methods that SB3 calls in PPO
    # ===========================================

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Given observations and actions, return (value, log_prob, entropy).
        This is what PPO needs for its loss function.
        """
        features = self._extract_features(obs)
        pi_latent = self.actor_mlp(features)
        vf_latent = self.critic_mlp(features)

        dist = self._get_action_dist_from_latent(pi_latent)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self.value_net(vf_latent).flatten()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        """
        Return the distribution for the given observations.
        Called by SB3 (e.g. in rollout sampling or `_predict()`).
        """
        features = self._extract_features(obs)
        pi_latent = self.actor_mlp(features)
        return self._get_action_dist_from_latent(pi_latent)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Used by SB3 to get the estimated V(s).
        """
        features = self._extract_features(obs)
        vf_latent = self.critic_mlp(features)
        return self.value_net(vf_latent).flatten()

    # ===========================================
    # Overriding the distribution logic
    # ===========================================

    def _get_action_dist_from_latent(self, pi_latent: th.Tensor) -> th.distributions.MultivariateNormal:
        """
        Build a MultivariateNormal distribution from the actor output:
         - self.action_net(pi_latent) => mean
         - self.cov_net(pi_latent) => diagonal elements of covariance
        """
        mean = self.action_net(pi_latent)
        diag = self.covariance_net(pi_latent) + 1e-5
        cov_matrix = th.diag_embed(diag)
        return th.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)

    # ===========================================
    # Misc: forward(...) and _predict(...)
    # ===========================================

    def forward(
            self,
            obs: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Compute (actions, values, log_probs) just by calling the distribution
        and sampling or taking mean.
        Note: PPO does NOT call this method internally for training,
        but you or other code might call `policy(obs)` directly.
        """
        dist = self.get_distribution(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        values = self.predict_values(obs)
        return actions, values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Called by `.predict()`. By default, the parent calls `get_distribution(obs)`
        and samples actions. We'll keep it consistent with `forward()`.
        """
        dist = self.get_distribution(observation)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()
        return actions


    # def forward(self, obs: th.Tensor, deterministic: bool = False):
    #     """
    #     Compute the policy’s action, its value estimate, and the log probability
    #     of the selected action, given observations.
    #     SB3 expects: actions, values, log_probs
    #     """
    #     distribution, value = self._get_dist_and_value(obs)
    #     distribution: th.distributions.multivariate_normal.MultivariateNormal
    #     if deterministic:
    #         actions = distribution.mean
    #     else:
    #         actions = distribution.sample()
    #
    #     log_probs = distribution.log_prob(actions)
    #     return actions, value, log_probs
    #
    #
    # def _get_dist_and_value(self, obs: th.Tensor):
    #
    #     pi_latent = self.mlp_extractor.forward_actor(obs)
    #     vf_latent = self.mlp_extractor.forward_critic(obs)
    #
    #     dist = self._get_action_dist_from_latent(pi_latent)
    #     value = self.value_net(vf_latent)
    #     return dist, value
    #
    # def _get_action_dist_from_latent(self, pi_latent: th.Tensor):
    #     # Continuous actions
    #     mean = self.action_net(pi_latent)
    #     diag = self.cov_net(pi_latent)
    #     diag = diag + 1e-5
    #     cov_matrix = th.diag_embed(diag)
    #     dist = th.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)
    #     return dist
    #
    #     # Old code to get complete covariance matrix from lower triangular part
    #     # dim = mean.shape[-1]
    #     # batch_size = mean.shape[0]
    #     # construct full matrix from lower triangular part
    #     # L_elements = self.cov_net(pi_latent)
    #     # L = th.zeros((batch_size, dim, dim), device=pi_latent.device)
    #     # tril_indices = th.tril_indices(row=dim, col=dim, offset=0)
    #     # L[:, tril_indices[0], tril_indices[1]] = L_elements
    #     # cov_matrix = L @ L.transpose(-1, -2)
    #     # small_identity = th.eye(dim) * 1e-5
    #     # cov_matrix = cov_matrix + small_identity
    #
    # # override
    # def _extract_features(self, obs: th.Tensor) -> th.Tensor:
    #     """
    #     Overriden method that extracts features from the observations.
    #     This is used by SB3 for e.g. rollout buffer calculations.
    #     """
    #     return obs
    #
    #
    # def _get_value(self, obs: th.Tensor):
    #     """
    #     Returns the estimated value (critic output) for given observations.
    #     Used internally by SB3 for e.g. rollout buffer calculations.
    #     """
    #     features = self.extract_features(obs)
    #     vf_latent = self.mlp_extractor.forward_critic(features)
    #     return self.value_net(vf_latent)
    #
    # def _predict(self, observation: th.Tensor, deterministic: bool = False):
    #     """
    #     Used by `.predict()` for evaluation. Only returns the actions.
    #     """
    #     actions, _, _ = self.forward(observation, deterministic=deterministic)
    #     return actions

    def load_pretrained_weights(self, weights: dict):
        """
        Load pretrained weights for the actor network.
        """
        self.load_state_dict(weights, strict=False)