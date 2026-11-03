"""Faiss raw-inner-product retrieval with explicit ID mapping/history filtering."""

import numpy as np


def lift_database(vectors, radius_squared):
    """Asymmetric MIPS-to-L2 lift; unlike normalization, it preserves IP order."""
    norms = np.square(vectors.astype(np.float64)).sum(1)
    if (norms > radius_squared + 1e-6).any():
        raise ValueError('Lift radius is smaller than a database vector norm.')
    extra = np.sqrt(np.maximum(radius_squared - norms, 0)).astype(np.float32)
    return np.ascontiguousarray(np.column_stack([vectors, extra]), dtype=np.float32)


class InnerProductFromL2:
    """Query (q,0) against (v,sqrt(R²-||v||²)); convert distances back to IP."""
    def __init__(self, index, radius_squared):
        self.index, self.radius_squared = index, radius_squared
        self.ntotal, self.d = index.ntotal, index.d - 1

    def search(self, queries, k):
        lifted = np.ascontiguousarray(np.column_stack([queries, np.zeros(len(queries), dtype=np.float32)]), dtype=np.float32)
        distances, ids = self.index.search(lifted, k)
        qnorm = np.square(queries.astype(np.float64)).sum(1)
        scores = ((qnorm[:, None] + self.radius_squared - distances) / 2).astype(np.float32)
        return scores, ids


def filter_results(raw_ids, raw_scores, seen, k=10):
    """Faiss row 0 represents local catalog ID 1, not padding ID 0."""
    predictions = np.zeros((len(raw_ids), k), dtype=np.int64)
    scores = np.empty((len(raw_ids), k), dtype=np.float32)
    for row in range(len(raw_ids)):
        blocked = set(np.asarray(seen[row]).tolist())
        selected = set()
        position = 0
        for index_id, score in zip(raw_ids[row], raw_scores[row]):
            item = int(index_id) + 1
            if index_id < 0 or item in blocked or item in selected:
                continue
            predictions[row, position], scores[row, position] = item, score
            selected.add(item)
            position += 1
            if position == k:
                break
        if position < k:
            raise ValueError('Insufficient candidates after filtering. Increase overfetch or inspect the index.')
    return predictions, scores


def search_filtered(index, queries, seen, k=10):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    fetch = min(int(index.ntotal), k + int(np.count_nonzero(seen, axis=1).max()))
    scores, ids = index.search(queries, fetch)
    return filter_results(ids, scores, seen, k)


def ann_recall(predictions, reference):
    return float((predictions[:, :, None] == reference[:, None, :]).any(-1).mean())


def select_configuration(rows, minimum_recall):
    eligible = [row for row in rows if row['ANNRecall@10'] >= minimum_recall]
    if not eligible:
        raise ValueError('No HNSW configuration meets the predeclared validation recall threshold.')
    return min(eligible, key=lambda row: (row['single_query_p95_ms'], row['efSearch']))
