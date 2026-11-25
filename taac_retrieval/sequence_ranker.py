"""Candidate-conditioned scoring with shared history and per-candidate tokens."""
import torch
from torch import nn
from torch.nn import functional as F

from .channel_features import BASIC, FEATURES


def heads(x, count):
    return x.unflatten(-1, (count, -1)).transpose(-2, -3)


def feed_forward(width):
    return nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))


class HistoryBlock(nn.Module):
    def __init__(self, width, count):
        super().__init__()
        self.count = count
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), feed_forward(width))

    def forward(self, sequence, valid):
        q, k, v = (heads(x, self.count) for x in self.qkv(self.norm(sequence)).chunk(3, dim=-1))
        length = sequence.size(1)
        diagonal = torch.eye(length, device=sequence.device, dtype=torch.bool)
        mask = (diagonal.tril() & valid[:, None, :]) | diagonal
        update = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        hidden = sequence + self.output(update.transpose(1, 2).flatten(-2))
        return hidden + self.ffn(hidden), k, v


class JointCandidateBlock(nn.Module):
    """Each candidate attends to shared history and its own ordered feature tokens."""
    def __init__(self, width, count, groups=2):
        super().__init__()
        self.count = count
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.ModuleList(nn.Linear(width, 3 * width) for _ in range(groups))
        self.output = nn.Linear(width, width)
        self.ffn = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), feed_forward(width))
                                 for _ in range(groups))

    def forward(self, tokens, history_k, history_v, valid):
        batch, candidates, groups, width = tokens.shape
        normalized = self.norm(tokens)
        projected = torch.stack([layer(normalized[:, :, i]) for i, layer in enumerate(self.qkv)], dim=2)
        q, k, v = (x.unflatten(-1, (self.count, -1)).permute(0, 3, 1, 2, 4)
                   for x in projected.chunk(3, dim=-1))
        scale = q.size(-1) ** -0.5
        history_logits = (q.flatten(2, 3) @ history_k.transpose(-1, -2)).unflatten(2, (candidates, groups))
        local_logits = q @ k.transpose(-1, -2)
        causal = torch.ones(groups, groups, device=tokens.device, dtype=torch.bool).tril()
        logits = torch.cat((history_logits.masked_fill(~valid[:, None, None, None, :], -torch.inf),
                            local_logits.masked_fill(~causal, -torch.inf)), dim=-1)
        history_weights, local_weights = (logits * scale).softmax(-1).split((history_k.size(2), groups), dim=-1)
        global_value = (history_weights.flatten(2, 3) @ history_v).unflatten(2, (candidates, groups))
        mixed = global_value + local_weights @ v
        hidden = tokens + self.output(mixed.permute(0, 2, 3, 1, 4).reshape(batch, candidates, groups, width))
        updates = torch.stack([layer(hidden[:, :, i]) for i, layer in enumerate(self.ffn)], dim=2)
        return hidden + updates


class HistoryRanker(nn.Module):
    def __init__(self, variant, width, count, max_history, mean, std, content_dim):
        super().__init__()
        self.variant = variant
        self.count = count
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.history_content = nn.Linear(content_dim, width)
        self.actions = nn.Embedding(3, width, padding_idx=0)
        self.position = nn.Embedding(max_history, width)
        self.age = nn.Linear(1, width, bias=False)
        self.history = HistoryBlock(width, count)
        self.item = nn.Linear(content_dim + len(BASIC), width)
        self.context = nn.Linear(len(FEATURES) - len(BASIC), width)
        self.history_kv = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width))
        if variant == "history_attention":
            self.query = nn.Linear(2 * width, width)
            self.combine = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
            self.score = nn.Linear(width, 1)
        elif variant == "joint_transformer":
            self.blocks = nn.ModuleList(JointCandidateBlock(width, count) for _ in range(2))
            self.score = nn.Linear(2 * width, 1)
        else:
            raise ValueError(variant)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, features, items, history, actions, ages, keep=None):
        positions = torch.arange(history.size(1), device=history.device)
        valid = actions != 0
        if keep is not None:
            valid = valid & keep
        sequence = self.history_content(history) + self.actions(actions) + self.position(positions)
        sequence = sequence + self.age(ages.unsqueeze(-1) / 10)
        encoded, raw_k, raw_v = self.history(sequence, valid)
        k, v = (heads(x, self.count) for x in self.history_kv(encoded).chunk(2, dim=-1))
        x = ((features - self.mean) / self.std).clamp(-6, 6)
        item = self.item(torch.cat((x[..., :len(BASIC)], items), dim=-1))
        context = self.context(x[..., len(BASIC):])
        if self.variant == "history_attention":
            q = heads(self.query(torch.cat((item, context), dim=-1)), self.count)
            update = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None])
            update = torch.where(valid.any(1)[:, None, None, None], update, 0)
            output = self.combine(torch.cat((item, context, update.transpose(1, 2).flatten(-2)), dim=-1))
        else:
            pair = torch.stack((item, context), dim=2)
            first = self.blocks[0](pair, raw_k, raw_v, valid)
            output = self.blocks[1](first, k, v, valid).flatten(-2)
        return self.score(output).squeeze(-1)
