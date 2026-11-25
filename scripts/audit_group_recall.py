"""Independently verify coverage, fixed budget, routing and frozen checkpoints."""
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from group_recall_study import arrays, bucket_masks, user_history
from taac_retrieval.data import PreparedData
from taac_retrieval.ranking_data import read, write_new
from taac_retrieval.ranking_model import digest


def verify(config, base, data, stage):
    root = Path(config["outputs"]) / stage
    result = read(root / "result.json")
    for path, sha in result["plan"]["source_sha256"].items():
        assert digest(path) == sha, path
    for name, sha in result["plan"]["inputs_sha256"].items():
        assert digest(data.root / name) == sha, name
    if stage == "confirm":
        cohort = read(root / "cohort.json")
        assert digest(root / "rows.npy") == cohort["rows_sha256"]
        old = read("reports/ranking_study/confirmation.json")["cohort_identity"]
        confirmed = data.user_rids[np.load(root / "rows.npy")]
        excluded = np.concatenate([
            np.load(source["user_rids"])[np.load(source["indices"])] for source in old["exclusion_sources"]
        ] + [data.user_rids[np.load("artifacts/ranking_confirmation_v1/cohort.npy")]])
        assert not np.intersect1d(confirmed, excluded).size
    attributes = np.load(data.root / "item_sparse.npy", mmap_mode="r")
    content = np.load(data.root / "item_mm81.npy", mmap_mode="r")
    columns = list(map(data.metadata["item_fields"].index, config["fields"]))
    report = {}
    for seed, entries in result["runs"].items():
        path = Path(base["root"]) / f"features_seed{seed}"
        frozen = read(path / "metadata.json")
        assert digest(frozen["checkpoint"]) == frozen["checkpoint_sha256"]
        assert digest(path / "item_vectors.npy") == frozen["files"]["item_vectors.npy"]
        model = Path(f"artifacts/ranking_content_v1/training/mean_mlp_seed{seed}")
        score_sha = digest(model / "best.pt")
        assert score_sha == read(model / "result.json")["checkpoint_sha256"]
        run = arrays(root / f"predictions_seed{seed}.npz")
        rows, target, baseline = (run[key] for key in ("rows", "targets", "baseline"))
        expected_rows = np.load(root / "rows.npy") if stage == "confirm" else data.eval_indices
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(target, data.items[rows, data.val_pos[rows]])
        history, action = user_history(data, rows)
        masks = bucket_masks(data, target)
        report[seed] = dict(frozen_recall_sha256=frozen["checkpoint_sha256"], frozen_ranking_sha256=score_sha)
        for mode in entries:
            prediction = run[mode]
            assert prediction.shape == (len(rows), 500)
            assert (np.diff(np.sort(prediction, axis=1), axis=1) > 0).all()
            assert prediction.min() > 0 and prediction.max() <= data.num_items
            rank = np.where((prediction == target[:, None]).any(1),
                            (prediction == target[:, None]).argmax(1) + 1, 501)
            expected = entries[mode] if mode == "baseline" else entries[mode]["groups"]
            for group, mask in masks.items():
                for k in (10, 100, 500):
                    assert int(((rank <= k) & mask).sum()) == expected[group][f"hits_at_{k}"]
            new_targets = []
            for i, (old, current, past) in enumerate(zip(baseline, prediction, history)):
                assert not np.intersect1d(current, past).size
                added = current[~np.isin(current, old)]
                assert np.isin(old[:config["protected_top"]], current).all()
                assert np.isin(old[data.train_counts[old] <= 5], current).all()
                assert int((data.train_counts[added] == 0).sum()) <= config["quotas"]["cold"]
                assert int(((data.train_counts[added] > 0) & (data.train_counts[added] <= 5)).sum()) <= config["quotas"]["tail"]
                assert (data.train_counts[added] <= 5).all()
                recent = past[(past > 0) & np.isin(action[i], (1, 2))][-config["recent_events"]:]
                if len(added):
                    tail_attrs = attributes[added][:, columns]
                    matches = (tail_attrs[:, None] == attributes[recent][None, :, columns]) & (tail_attrs[:, None] > 0)
                    valid = (np.linalg.norm(content[recent], axis=1) > 0)[None, :]
                    assert (matches.any(2) & valid).any(1).all()
                    assert (np.linalg.norm(content[added], axis=1) > 0).all()
                if target[i] in added:
                    query = content[target[i]]
                    past_vectors = content[recent]
                    similarity = (past_vectors @ query) / np.maximum(np.linalg.norm(past_vectors, axis=1)*np.linalg.norm(query), 1e-8)
                    matched = matches[added == target[i]][0].any(1)
                    best = int(np.argmax(np.where(matched, similarity, -np.inf)))
                    new_targets.append(dict(user_rid=int(data.user_rids[rows[i]]), rank=int(rank[i]),
                        target_oid=int(data.catalog_oids[target[i]]), history_oid=int(data.catalog_oids[recent[best]]),
                        training_frequency=int(data.train_counts[target[i]]),
                        similarity=float(similarity[best])))
            report[seed][mode] = dict(new_targets=new_targets)
    return report


def main():
    config = read("configs/group_recall.json")
    base = read(config["ranking_config"])
    data = PreparedData(base["cache"])
    report = dict(scope="Independent verification of saved development and confirmation candidate pools",
                  stages={})
    for stage in ("develop", "confirm"):
        if (Path(config["outputs"]) / stage / "result.json").exists():
            report["stages"][stage] = verify(config, base, data, stage)
    write_new("reports/group_recall/verification.json", report)
    for stage, results in report["stages"].items():
        hits = [hit for seed in results.values() for hit in seed["match"]["new_targets"]]
        print(json.dumps(dict(stage=stage, gained_observations=len(hits),
            rank_range=[min(h["rank"] for h in hits), max(h["rank"] for h in hits)])))


if __name__ == "__main__":
    main()
