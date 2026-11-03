"""Explicit next-event supervision policies and sampled contrastive losses."""

import torch
import torch.nn.functional as F


def supervision_weights(actions, objective, exposure_weight=0.1):
    if objective not in ['all_bce', 'weighted_bce', 'click_bce', 'click_infonce', 'all_infonce']:
        raise ValueError(objective)
    if exposure_weight < 0:
        raise ValueError('Exposure weight must be nonnegative.')
    weights = (actions == 2).float()
    if objective in ['all_bce', 'all_infonce']:
        weights = weights + (actions == 1).float()
    elif objective == 'weighted_bce':
        weights = weights + (actions == 1).float() * exposure_weight
    # Padding and unknown actions never provide supervised labels.
    return weights


def weighted_bce(positive, negative, weights):
    pos = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction='none')
    neg = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction='none')
    if negative.ndim == positive.ndim + 1:
        neg = neg.mean(-1)
    if neg.shape != pos.shape or weights.shape != pos.shape:
        raise ValueError('Scores and supervision weights have incompatible shapes.')
    if not (weights.sum() > 0):
        raise ValueError('No supervised positions.')
    return ((pos + neg) * weights).sum() / weights.sum()


def sampled_info_nce(positive, negative, temperature=1.0):
    if temperature <= 0 or negative.ndim != 2 or positive.ndim != 1:
        raise ValueError('Expected [positions], [positions, negatives] scores and positive temperature.')
    logits = torch.cat([positive.unsqueeze(-1), negative], dim=-1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(positive), dtype=torch.long, device=positive.device))
