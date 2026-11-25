"""Residual listwise ranking over a fixed recall candidate list."""
import torch
from torch import nn
from torch.nn import functional as F


class CrossLayer(nn.Module):
    """Low-rank DCN V2 cross layer, with x0 carried across layers."""
    def __init__(self, features, rank):
        super().__init__()
        self.down = nn.Linear(features, rank, bias=False)
        self.up = nn.Linear(rank, features)

    def forward(self, original, current):
        return current + original * self.up(self.down(current))


class ChannelRanker(nn.Module):
    def __init__(self, mean, std, width, cross_rank=None):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        features = mean.numel()
        self.deep = nn.Sequential(nn.Linear(features, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU())
        self.cross = nn.ModuleList(
            CrossLayer(features, cross_rank) for _ in range(2)
        ) if cross_rank else nn.ModuleList()
        self.score = nn.Linear(width + features * bool(self.cross), 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, features):
        x = ((features[..., :self.mean.numel()] - self.mean) / self.std).clamp(-6, 6)
        deep = self.deep(x)
        if self.cross:
            crossed = x
            for layer in self.cross:
                crossed = layer(x, crossed)
            deep = torch.cat((deep, crossed), dim=-1)
        return self.score(deep).squeeze(-1)


def list_loss(prior, residual, positive, weights, penalty):
    """One target per user; alternatives compete only within the same list."""
    losses = F.cross_entropy(prior + residual, positive, reduction="none")
    return (losses * weights).sum() / weights.sum() + penalty * residual.square().mean()
