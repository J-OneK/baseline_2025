"""Candidate-relative features for channel-aware ranking."""
import numpy as np
import torch
from torch.nn import functional as F


BASIC = ("retrieval_z", "ctr_z", "train_events_log", "early_clicks_log", "content_present")
FEATURES = BASIC + (
    "added_channel", "click_match_cosine", "click_match_present", "exposure_match_cosine",
    "exposure_match_present", "best_match_from_click", "best_match_age_log",
    "matched_fields", "matched_posting_size_log", "best_content_cosine"
)


def recent_context(data, rows, length):
    """Left-pad known events strictly before the original validation boundary."""
    items = np.zeros((len(rows), length), np.int32)
    actions = np.zeros_like(items)
    ages = np.zeros((len(rows), length), np.float32)
    for i, row in enumerate(rows):
        cut = int(data.val_cut[row])
        observed = np.flatnonzero(np.isin(data.actions[row, :cut], (1, 2)))[-length:]
        if not len(observed):
            continue
        positions = slice(-len(observed), None)
        items[i, positions] = data.items[row, observed]
        actions[i, positions] = data.actions[row, observed]
        elapsed = data.timestamps[row, cut - 1] - data.timestamps[row, observed]
        ages[i, positions] = np.log1p(np.maximum(elapsed, 0) / 60)
    return items, actions, ages


@torch.inference_mode()
def channel_features(items, history, actions, ages, vector, attributes, posting_sizes, source, click_bonus):
    """Side features use only observed history and frozen catalog attributes."""
    queries = vector[history]
    similarity = torch.bmm(vector[items], queries.transpose(1, 2)).clamp(-1, 1)
    valid = (queries.square().sum(-1) > 0)[:, None] & (vector[items].square().sum(-1) > 0)[:, :, None]
    left, right = attributes[items], attributes[history]
    same = (left[:, :, None, :] == right[:, None, :, :]) & (left[:, :, None] > 0)
    match = same.any(-1) & valid
    click = actions == 2
    exp = actions == 1
    groups = []
    for mask in (click, exp):
        eligible = match & mask[:, None]
        found = eligible.any(2)
        value = similarity.masked_fill(~eligible, -2).amax(2)
        groups.extend([torch.where(found, value, 0), found.float()])
    found = match.any(2)
    preferred = (similarity + click_bonus * click[:, None]).masked_fill(~match, -2).argmax(2)
    best_action = click.gather(1, preferred).float()
    age = ages.gather(1, preferred)
    match_fields = same.gather(2, preferred[:, :, None, None].expand(-1, -1, 1, attributes.shape[1])).squeeze(2)
    size = posting_sizes[items]
    min_size = size.masked_fill(~match_fields, 2_000_000).amin(-1)
    any_value = similarity.masked_fill(~valid, -2).amax(2).clamp(min=0)
    return torch.stack((
        source.float(), *groups, best_action * found, age * found,
        match_fields.float().sum(-1) * found, min_size.float().log1p() * found, any_value
    ), dim=-1)


def vector_tables(data, fields, device):
    count = data.train_counts
    content = torch.as_tensor(np.load(data.root / "item_mm81.npy"), device=device)
    unit = F.normalize(content, dim=1, eps=1e-8)
    columns = [data.metadata["item_fields"].index(field) for field in fields]
    values = np.load(data.root / "item_sparse.npy", mmap_mode="r")[:, columns].copy()
    size = np.empty_like(values, dtype=np.int32)
    eligible = (count <= 5) & np.load(data.root / "mm_present.npy")
    eligible[0] = False
    for j in range(len(columns)):
        bucket_count = np.bincount(values[eligible, j], minlength=int(values[:, j].max()) + 1)
        size[:, j] = bucket_count[values[:, j]]
    return unit, torch.as_tensor(values, device=device), torch.as_tensor(size, device=device)


def normalize_to_original(scores, changed, replacements):
    """Compute z-scores in the original retrieval pool and extend to add-ons."""
    mu, sigma = scores.mean(1, keepdims=True), scores.std(1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    original = (scores - mu) / sigma
    result = original.copy()
    rows = np.where(changed)[0]
    if len(rows):
        result[changed] = (replacements - mu[rows, 0]) / sigma[rows, 0]
    return original, result
