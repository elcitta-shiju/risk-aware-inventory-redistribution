"""Parameter-shared (permutation-equivariant) actor-critic policy for per-store actions.

The observation is N blocks of k per-store features followed by g global features. Each store's
action is produced by the SAME small network applied to [own features, pooled network context,
globals] (a deep-sets style design), so a single store-level decision rule is learned from all
N stores at once instead of N separate output heads of a flat MLP. The critic uses the pooled
context. The log-std remains one parameter per action dimension, as in standard SB3 PPO.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class PerStoreExtractor(nn.Module):
    def __init__(self, n_stores: int, k: int, n_global: int, hidden: int):
        super().__init__()
        self.n, self.k, self.g = n_stores, k, n_global
        self.embed = nn.Sequential(nn.Linear(k, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(k + 2 * hidden + n_global, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(2 * hidden + n_global, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.latent_dim_pi = n_stores * hidden
        self.latent_dim_vf = hidden

    def _context(self, obs: torch.Tensor):
        x = obs[:, : self.n * self.k].reshape(-1, self.n, self.k)
        g = obs[:, self.n * self.k:]
        e = self.embed(x)
        pooled = torch.cat([e.mean(1), e.max(1).values], dim=-1)
        return x, g, pooled

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        x, g, pooled = self._context(obs)
        ctx = torch.cat([pooled, g], -1).unsqueeze(1).expand(-1, self.n, -1)
        return self.actor(torch.cat([x, ctx], -1)).reshape(obs.shape[0], -1)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        _, g, pooled = self._context(obs)
        return self.critic(torch.cat([pooled, g], -1))

    def forward(self, obs: torch.Tensor):
        return self.forward_actor(obs), self.forward_critic(obs)


class PerStoreHead(nn.Module):
    """Shared linear head: per-store latent (hidden) -> one action mean per store."""

    def __init__(self, n_stores: int, hidden: int):
        super().__init__()
        self.n = n_stores
        self.lin = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.lin.weight, gain=0.01)
        nn.init.zeros_(self.lin.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.lin(latent.reshape(latent.shape[0], self.n, -1)).squeeze(-1)


class PerStorePolicy(ActorCriticPolicy):
    def __init__(self, *args, n_stores: int, per_store_features: int, n_global: int = 2, hidden: int = 64, **kwargs):
        self.n_stores, self.k, self.n_global, self.hidden = n_stores, per_store_features, n_global, hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PerStoreExtractor(self.n_stores, self.k, self.n_global, self.hidden)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = PerStoreHead(self.n_stores, self.hidden)
        # rebuild the optimizer so it tracks the replaced head
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
