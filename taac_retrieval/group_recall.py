"""Retrieve rare items through shared attributes and text similarity."""
import numpy as np


class RareAttributeIndex:
    def __init__(self, attributes, content, frequency, columns, max_posting):
        self.attributes = attributes
        self.content = content
        self.frequency = frequency
        supported = (frequency <= 5) & (np.linalg.norm(content, axis=1) > 0)
        supported[0] = False
        ids = np.flatnonzero(supported).astype(np.int32)
        self.postings = {}
        for column in columns:
            values = attributes[ids, column]
            order = np.argsort(values, kind="stable")
            keys, starts, sizes = np.unique(values[order], return_index=True, return_counts=True)
            ordered_ids = ids[order]
            self.postings[column] = {
                int(key): ordered_ids[start:start + size]
                for key, start, size in zip(keys, starts, sizes)
                if key > 0 and size <= max_posting
            }

    def propose(self, history, actions, recent_events, count, click_bonus):
        """Return mean and matched-exemplar proposals, grouped by frequency."""
        known = (history > 0) & np.isin(actions, (1, 2))
        items, actions = history[known][-recent_events:], actions[known][-recent_events:]
        valid = np.any(self.content[items] != 0, axis=1)
        items, actions = items[valid], actions[valid]
        proposals = np.zeros((2, 2, count), dtype=np.int32)
        if not len(items):
            return proposals
        source = self.content[items]
        center = np.average(source, axis=0, weights=1 + (actions == 2)).astype(np.float32)
        center /= max(float(np.linalg.norm(center)), 1e-8)
        candidate_parts = []
        score_parts = []
        for column, postings in self.postings.items():
            keys = self.attributes[items, column]
            for key in np.unique(keys):
                posting = postings.get(int(key))
                if posting is None:
                    continue
                mask = keys == key
                similarities = self.content[posting] @ source[mask].T
                similarities += click_bonus * (actions[mask] == 2)
                candidate_parts.append(posting)
                score_parts.append(similarities.max(1))
        if not candidate_parts:
            return proposals
        unique, inverse = np.unique(np.concatenate(candidate_parts), return_inverse=True)
        matched = np.full(len(unique), -np.inf, dtype=np.float32)
        np.maximum.at(matched, inverse, np.concatenate(score_parts))
        keep = ~np.isin(unique, history)
        unique, matched = unique[keep], matched[keep]
        scores = (self.content[unique] @ center, matched)
        for mode, score in enumerate(scores):
            for group, cold in enumerate((True, False)):
                mask = (self.frequency[unique] == 0) == cold
                ids, values = unique[mask], score[mask]
                order = np.lexsort((ids, -values))[:count]
                proposals[mode, group, :len(order)] = ids[order]
        return proposals


def merge_candidates(base, protected, frequency, proposals, quotas):
    """Only replace the weakest unprotected main candidates under a fixed budget."""
    output = base.copy()
    removable = np.flatnonzero(~protected & (frequency[base] > 5))[::-1]
    seen = set(base)
    added = []
    for group, quota in zip(proposals, quotas):
        used = 0
        for item in group:
            if used >= quota or len(added) == len(removable):
                break
            if item > 0 and item not in seen:
                seen.add(item)
                added.append(item)
                used += 1
    output[removable[:len(added)]] = added
    return output
