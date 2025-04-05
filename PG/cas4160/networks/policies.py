import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cas4160.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = ptu.from_numpy(obs[None]) if obs.ndim == 1 else ptu.from_numpy(obs)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        action = ptu.to_numpy(action.squeeze(0)) if obs.ndim == 1 else ptu.to_numpy(action)
        return action

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # HINT: use torch.distributions.Categorical to define the distribution.
            logits = self.logits_net(obs)
            dist = distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # HINT: use torch.distributions.Normal to define the distribution.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            dist = distributions.Normal(mean, std)
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # HINT: don't forget to do `self.optimizer.step()`!
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        if self.discrete:
            log_probs = log_probs
        else:
            log_probs = log_probs.sum(-1)

        loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # TODO: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        dist = self.forward(obs)
        logp = dist.log_prob(actions) #Calculating logp
        if not self.discrete:
            logp = logp.sum(-1)
        
        ratio = torch.exp(logp - old_logp)  
        clip_adv = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange) * advantages    #Clipping using torch.clamp
        loss = -torch.min(ratio * advantages, clip_adv).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"PPO Loss": ptu.to_numpy(loss)}
