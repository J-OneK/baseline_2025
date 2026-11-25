"""Build listwise training features on actual retrieved and augmented candidates."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from group_recall_study import arrays, score_mlp
from prepare_channel_rank import selected_rows
from taac_retrieval.channel_features import (
    BASIC, FEATURES, channel_features, normalize_to_original, recent_context, vector_tables,
)
from taac_retrieval.content_ranker_train import load_content_ranker
from taac_retrieval.data import PreparedData
from taac_retrieval.group_recall import merge_candidates
from taac_retrieval.ranking_data import history_clicks, read, write_new
from taac_retrieval.ranking_model import digest


@torch.inference_mode()
def feature_array(data, candidates, rows, changed, retrieval_z, ctr_z, base, group, device):
    vector, attr, postings = vector_tables(data, group["fields"], device)
    hist, action, ages = recent_context(data, rows, group["recent_events"])
    early_clicks = np.load(Path(base["root"]) / "examples/stats_clicks.npy")
    present = np.load(data.root / "mm_present.npy")
    basic = np.stack((retrieval_z, ctr_z, np.log1p(data.train_counts[candidates]),
                      np.log1p(early_clicks[candidates]), present[candidates]), -1)
    side = []
    for start in range(0, len(rows), 96):
        s = slice(start, start + 96)
        inputs = [torch.as_tensor(value[s], device=device) for value in (candidates, hist, action, ages)]
        source = torch.as_tensor(changed[s], device=device)
        extra = channel_features(*inputs, vector, attr, postings, source, group["click_bonus"])
        side.append(extra.cpu().numpy())
    x = np.concatenate((basic, np.concatenate(side)), -1).astype(np.float32)
    assert x.shape == (*candidates.shape, len(FEATURES))
    assert np.isfinite(x).all()
    return x


def score_candidate_lists(arrays_input, proposals, base, group, data, seed, device):
    """Score the original and augmented lists using the same calibrated prior."""
    rows = arrays_input["rows"]
    frozen_path = Path("artifacts/ranking_content_v1/training") / f"mean_mlp_seed{seed}" / "best.pt"
    ranker, rank_config, _ = load_content_ranker(frozen_path, device)
    alpha = rank_config["blend"]["alpha"]
    raw, raw_scores = arrays_input["candidates"], arrays_input["scores"]
    history = np.stack([
        history_clicks(data.items[row], data.actions[row], int(data.val_cut[row]), base["data"]["click_history"])
        for row in rows
    ])
    ctr = score_mlp(ranker, raw, rows, history, device)
    empty = np.zeros(raw.shape, dtype=bool)
    original_r, _ = normalize_to_original(raw_scores, empty, [])
    original_c, _ = normalize_to_original(ctr, empty, [])
    original_score = (1-alpha)*original_r + alpha*original_c
    original_sort = np.argsort(-original_score, axis=1, kind="stable")
    original_order = np.take_along_axis(raw, original_sort, axis=1)
    quota = [group["quotas"][g] for g in ("cold", "tail")]
    merged = np.stack([
        merge_candidates(retrieved, np.isin(retrieved, ranked[:group["protected_top"]]),
                         data.train_counts, new, quota)
        for retrieved, ranked, new in zip(raw, original_order, proposals)
    ])
    changed = merged != raw
    added = np.zeros((len(rows), sum(quota)), dtype=raw.dtype)
    per_user, _ = np.where(changed)
    offset = np.cumsum(changed, 1)[changed] - 1
    added[per_user, offset] = merged[changed]
    vectors = np.load(Path(base["root"]) / f"features_seed{seed}" / "item_vectors.npy", mmap_mode="r")
    added_recall = np.einsum("bd,bd->b", arrays_input["queries"][per_user], vectors[merged[changed]])
    added_ctr = score_mlp(ranker, added, rows, history, device)[per_user, offset]
    _, recall_z = normalize_to_original(raw_scores, changed, added_recall)
    _, ctr_z = normalize_to_original(ctr, changed, added_ctr)
    del ranker
    fusion = (1-alpha)*recall_z + alpha*ctr_z
    return merged, changed, fusion, recall_z, ctr_z, original_order


def build(config, base, group, data, split, seed, device):
    root = Path(config["output"]) / split
    reference = root if split != "dev" else Path(group["outputs"]) / "confirm"
    assert not (root / f"features_seed{seed}.json").exists()
    source = reference / f"recall_seed{seed}.npz"
    arrays_input = arrays(source)
    cohort = selected_rows(config, split)
    np.testing.assert_array_equal(arrays_input["rows"], cohort)
    proposals = np.load(root / "proposals.npy" if split != "dev" else reference / "proposals.npy",
                        allow_pickle=False)
    if split == "dev":
        proposals = proposals[:, 1]
    assert proposals.shape[:2] == (len(cohort), 2)
    n = len(cohort)
    if split == "fit":
        # Users absent from the union cannot yield a supervised row. Their labels never alter a candidate.
        target = arrays_input["targets"]
        possible = (arrays_input["candidates"] == target[:, None]).any(1)
        possible |= (proposals == target[:, None, None]).any((1, 2))
        arrays_input = {key:value[possible] for key,value in arrays_input.items()}
        proposals = proposals[possible]
    rows, target = arrays_input["rows"], arrays_input["targets"]
    np.testing.assert_array_equal(target, data.items[rows, data.val_pos[rows]])
    merged, changed, fusion, recall_z, ctr_z, original_order = score_candidate_lists(
        arrays_input, proposals, base, group, data, seed, device)
    frozen_path = Path("artifacts/ranking_content_v1/training") / f"mean_mlp_seed{seed}" / "best.pt"
    label = merged == target[:, None]
    keep = label.any(1) if split == "fit" else np.ones(len(rows), dtype=bool)
    rows, target, merged, changed, fusion, recall_z, ctr_z, original_order = (
        array[keep] for array in (rows, target, merged, changed, fusion, recall_z, ctr_z, original_order)
    )
    features = feature_array(data, merged, rows, changed, recall_z, ctr_z, base, group, device)
    base_order = np.argsort(-fusion, kind="stable", axis=1)
    fixed = np.take_along_axis(merged, base_order, axis=1)
    if split == "dev":
        known = arrays(reference / f"predictions_seed{seed}.npz")
        np.testing.assert_array_equal(rows, known["rows"])
        np.testing.assert_array_equal(np.sort(fixed, 1), np.sort(known["match"], 1))
        np.testing.assert_array_equal(fixed[:, :10], known["match"][:, :10])
    np.save(root / f"x_seed{seed}.npy", features)
    np.savez(root / f"lists_seed{seed}.npz", rows=rows, targets=target, candidates=merged, prior=fusion,
             original=original_order, fixed=fixed, added=changed)
    positive = (merged == target[:, None]) & changed
    write_new(root / f"features_seed{seed}.json", dict(
        split=split, seed=seed, initially_available=n, n_rows=len(rows),
        ranker_checkpoint_sha256=digest(frozen_path),
        retrieval_candidates_sha256=digest(source),
        features=list(FEATURES), basic_features=len(BASIC),
        aux_positives=int(positive.sum()),
        cold_targets=int((data.train_counts[target] == 0).sum()),
        tail_targets=int(((data.train_counts[target] > 0)&(data.train_counts[target] <= 5)).sum()),
        test_evaluated=False
    ))
    print(f"{split} seed{seed} features ready. {len(rows)} users, {int(positive.sum())} auxiliary positives.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/channel_rank.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=("fit","dev","fresh"), required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    config = read(a.config)
    base, group = read(config["ranking_config"]), read(config["group_config"])
    data = PreparedData(base["cache"])
    build(config, base, group, data, a.split, a.seed, a.device)


if __name__ == "__main__":
    main()
