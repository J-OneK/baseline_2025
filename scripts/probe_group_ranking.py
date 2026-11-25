"""Probe a single position for highly matched candidates on development users."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from finish_ranking_study import paired
from group_recall_study import arrays, bucket_masks, summary, user_history
from taac_retrieval.data import PreparedData
from taac_retrieval.ranking_data import read, write_new
from taac_retrieval.ranking_model import digest


def promote(original, ranked, history, actions, attributes, content, cols, recent, threshold):
    candidates = ranked[~np.isin(ranked, original)]
    if not len(candidates):
        return ranked.copy(), None
    known = (history > 0) & np.isin(actions, (1, 2))
    items = history[known][-recent:]
    if not len(items):
        return ranked.copy(), None
    candidate_attrs = attributes[candidates][:, cols]
    history_attrs = attributes[items][:, cols]
    match = (candidate_attrs[:, None] == history_attrs[None, :]).all(2)
    match &= (candidate_attrs > 0).all(1)[:, None]
    candidate_content, history_content = content[candidates], content[items]
    norm = np.linalg.norm(candidate_content, axis=1)[:, None] * np.linalg.norm(history_content, axis=1)[None, :]
    cosine = (candidate_content @ history_content.T) / np.maximum(norm, 1e-8)
    cosine[~match | (norm == 0)] = -np.inf
    candidate_index, history_index = np.unravel_index(cosine.argmax(), cosine.shape)
    best = cosine[candidate_index, history_index]
    if best < threshold:
        return ranked.copy(), None
    item = int(candidates[candidate_index])
    position = int(np.flatnonzero(ranked == item)[0])
    if position < 10:
        return ranked.copy(), None
    promoted = np.insert(np.delete(ranked, position), 9, item)
    return promoted, dict(item=item, source_item=int(items[history_index]), cosine=float(best), from_rank=position+1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/group_recall.json")
    p.add_argument("--output", default="reports/group_recall/ranking_probe.json")
    args = p.parse_args()
    config = read(args.config)
    base = read(config["ranking_config"])
    data = PreparedData(base["cache"])
    features = np.load(data.root / "item_sparse.npy", mmap_mode="r")
    content = np.load(data.root / "item_mm81.npy", mmap_mode="r")
    cols = [data.metadata["item_fields"].index(name) for name in config["fields"]]
    root = Path(config["outputs"]) / "develop"
    result = dict(scope="Post-hoc development-only ranking probe; thresholds chosen after inspecting recovered targets",
                  threshold=.995, required_fields=config["fields"], position=10, runs={},
                  source_sha256=digest(Path(__file__)), test_evaluated=False)
    samples = {}
    for seed in base["seeds"]:
        source = root / f"predictions_seed{seed}.npz"
        runs = arrays(source)
        original, ranked, rows, target = (runs[key] for key in ("baseline", "match", "rows", "targets"))
        history, actions = user_history(data, rows)
        outputs = [promote(old, current, seen, acted, features, content, cols, config["recent_events"], result["threshold"])
                   for old, current, seen, acted in zip(original, ranked, history, actions)]
        reranked = np.stack([prediction for prediction, detail in outputs])
        np.testing.assert_array_equal(np.sort(reranked, 1), np.sort(ranked, 1))
        np.testing.assert_array_equal(reranked[:, :9], ranked[:, :9])
        groups, per = summary(reranked, target, rows, data)
        original_groups, original_per = summary(ranked, target, rows, data)
        samples[seed] = {k:per[k].astype(float)-original_per[k] for k in ("hits_at_10", "ndcg_at_10")}
        gains = (samples[seed]["hits_at_10"] == 1)
        losses = (samples[seed]["hits_at_10"] == -1)
        cases = []
        for index in np.flatnonzero(gains | losses):
            detail = outputs[index][1]
            cases.append(dict(user_rid=int(data.user_rids[rows[index]]), gained=bool(gains[index]),
                target_frequency=int(data.train_counts[target[index]]), target_oid=int(data.catalog_oids[target[index]]),
                **detail))
        result["runs"][str(seed)] = dict(triggered_users=sum(d is not None for _, d in outputs),
            baseline=original_groups, promoted=groups, hit_change_cases=cases, input_sha256=digest(source))
        np.save(root / f"promoted_seed{seed}.npy", reranked)
    result["paired"] = {
        name:{key:paired(np.stack([sample[key][mask] for sample in samples.values()]))
              for key in ("hits_at_10", "ndcg_at_10")}
        for name, mask in bucket_masks(data, target).items()
    }
    write_new(args.output, result)
    print(json.dumps(dict(
        triggered={seed:run["triggered_users"] for seed, run in result["runs"].items()},
        gains={seed:[c["target_frequency"] for c in run["hit_change_cases"] if c["gained"]]
               for seed, run in result["runs"].items()},
        losses={seed:[c["target_frequency"] for c in run["hit_change_cases"] if not c["gained"]]
                for seed, run in result["runs"].items()},
        diffs={name:group["hits_at_10"] for name, group in result["paired"].items()}
    ), indent=2))


if __name__ == "__main__":
    main()
