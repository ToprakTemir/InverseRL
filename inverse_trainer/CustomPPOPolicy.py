import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution, StateDependentNoiseDistribution
from stable_baselines3.common.type_aliases import Schedule

class CustomPolicy(ActorCriticPolicy):
    """
    A custom policy that uses SB3's built-in StateDependentNoiseDistribution
    for SDE-based exploration, without manually building a separate covariance net.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule: Schedule = lambda _: 3e-4,
        net_arch=(128, 128),
        activation_fn=nn.ReLU,
        # The following are typical SDE-related arguments in SB3:
        use_sde: bool = True,           # use StateDependentNoiseDistribution
        log_std_init: float = -3.0,    # initial log-std
        full_std: bool = True,         # shape of std param
        use_expln: bool = False,       # use expln() transform vs. exp()
        **kwargs
    ):
        # Dist kwargs for SDE
        dist_kwargs = dict(
            full_std=full_std,
            use_expln=use_expln,
            squash_output=False,   # or True if you want to tanh() the mean
        )

        assert use_sde is True, "USE SDE"

        # Pass everything to ActorCriticPolicy, enabling SDE
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[],                # skip default MlpExtractor
            activation_fn=nn.Identity,  # we will build our own MLP
            use_sde=use_sde,              # <--- The big switch
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=False,       # or True if you want to tanh() the mean
            **kwargs
        )

        # Store your custom net arch so we can build it in _build().
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln

    def _build_mlp_extractor(self) -> None:
        """
        Do nothing so the parent doesn't create a default MlpExtractor.
        """
        pass

    def _make_dist(self, action_space: spaces.Space):
        """
        Return SB3's standard distribution with use_sde=True -> StateDependentNoiseDistribution
        """
        return make_proba_distribution(
            action_space,
            use_sde=True,
            dist_kwargs=self.dist_kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Called at end of parent __init__. Create your own networks,
        then build the actor & log_std for SDE via `proba_distribution_net`.
        """
        obs_dim = self.features_dim  # typically shape of Box obs
        act_dim = self.action_space.shape[0]

        # --- Actor MLP (before the distribution layers) ---
        actor_layers = []
        last_dim = obs_dim
        for size in self.net_arch:
            actor_layers.append(nn.Linear(last_dim, size))
            actor_layers.append(self.activation_fn())
            last_dim = size
        self.actor_mlp = nn.Sequential(*actor_layers)

        # --- Critic MLP ---
        critic_layers = []
        last_dim = obs_dim
        for size in self.net_arch:
            critic_layers.append(nn.Linear(last_dim, size))
            critic_layers.append(self.activation_fn())
            last_dim = size
        critic_layers.append(nn.Linear(last_dim, 1))
        self.value_net = nn.Sequential(*critic_layers)

        # The SDE distribution needs us to define:
        # self.action_net, self.log_std = self.action_dist.proba_distribution_net(...)
        # which returns two modules or parameters for the final step.
        latent_dim_pi = last_dim  # final size of actor layers
        latent_sde_dim = latent_dim_pi  # typically same for SDE
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            latent_sde_dim=latent_sde_dim,
            log_std_init=self.log_std_init
        )

        # Create optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """
        Called by PPO to get (values, log_prob, entropy).
        """
        features = self.extract_features(obs, self.features_extractor)
        pi_latent = self.actor_mlp(features)
        if th.isnan(pi_latent).any():
            print("NAN in pi_latent")
        values = self.value_net(features).flatten()

        # Build distribution from actor-latent:
        dist = self._get_action_dist_from_latent(pi_latent)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        """
        Used by SB3 for sampling actions in rollouts.
        """
        features = self.extract_features(obs, self.features_extractor)
        pi_latent = self.actor_mlp(features)
        return self._get_action_dist_from_latent(pi_latent)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Returns V(s) for the given observation.
        """
        features = self.extract_features(obs, self.features_extractor)
        return self.value_net(features).flatten()

    def _get_action_dist_from_latent(self, pi_latent: th.Tensor):
        """
        The parent's SDE distribution expects:
         dist = self.action_dist.proba_distribution(mean_actions, self.log_std, sde_latent)
        """
        if th.isnan(pi_latent).any():
            print("NAN in pi_latent")
        mean_actions = self.action_net(pi_latent)
        if th.isnan(mean_actions).any():
            print("NAN in mean actions")
        if self.log_std is None:
            print("log_std is None")
        if th.isnan(self.log_std).any():
            print("NAN in log_std")
        if th.isinf(self.log_std).any():
            print("INF in log_std")

        return self.action_dist.proba_distribution(mean_actions, self.log_std, pi_latent)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        If you call `policy(obs)`, returns (actions, values, log_prob).
        PPO does NOT call this internally for training, but it's nice to have for debug.
        """
        dist: StateDependentNoiseDistribution = self.get_distribution(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.predict_values(obs)
        return actions, values, log_prob

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Called by `policy.predict(...)`; returns just the action.
        """
        dist = self.get_distribution(obs)
        return dist.get_actions(deterministic=deterministic)