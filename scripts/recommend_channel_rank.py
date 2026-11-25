"""Run exact retrieval, grouped content recall and locked DCN V2 for one user."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from build_channel_features import feature_array, score_candidate_lists
from group_recall_study import proposals_for, user_history
from run_channel_rank import infer, load_model as load_ranker, ranks_for
from taac_retrieval.data import SequenceDataset
from taac_retrieval.evaluate_checkpoint import load_model
from taac_retrieval.metrics import exact_topk, to_device
from taac_retrieval.ranking_data import read
from taac_retrieval.ranking_model import digest


@torch.inference_mode()
def recommend(args):
    config = read(args.config)
    base, group = read(config["ranking_config"]), read(config["group_config"])
    root = Path(config["output"])
    lock = read(root / "selection.json")
    chosen = lock["selected"]
    seed = args.seed
    ranker = chosen["checkpoints"][str(seed)]
    if digest(ranker["checkpoint"]) != ranker["sha256"]:
        raise ValueError("Ranking weights differ from the locked checksum.")
    folder = Path(base["root"]) / f"features_seed{seed}"
    metadata = read(folder / "metadata.json")
    vectors_path = folder / "item_vectors.npy"
    if (digest(metadata["checkpoint"]) != metadata["checkpoint_sha256"]
            or digest(vectors_path) != metadata["files"][vectors_path.name]):
        raise ValueError("Retrieval artifacts differ from the frozen feature metadata.")
    retriever, data, _ = load_model(metadata["checkpoint"], base["cache"], args.device)
    if args.rid is None:
        rows = np.load(root / args.cohort / "rows.npy")
        if not 0 <= args.index < len(rows):
            raise IndexError("User index is outside the selected cohort.")
        row = int(rows[args.index])
    else:
        matches = np.flatnonzero(data.user_rids == args.rid)
        if not len(matches):
            raise ValueError("User RID is outside the prepared cache.")
        row = int(matches[0])
    rows = np.array([row])
    data.set_eval_indices(rows)
    sample = SequenceDataset(data, "val")[0]
    batch = to_device(default_collate([sample]), args.device)
    q = retriever.predict(batch)
    vectors = torch.as_tensor(np.load(vectors_path), device=args.device)
    candidates = exact_topk(q, vectors, batch["items"], k=500)
    scores = (q[:, None] * vectors[candidates]).sum(-1).cpu().numpy()
    recall = dict(rows=rows, candidates=candidates.cpu().numpy(), scores=scores, queries=q.cpu().numpy())
    del retriever, vectors
    history, actions = user_history(data, rows)
    proposals = proposals_for(group, data, history, actions)[:, 1]
    merged, changed, fusion, rz, cz, original = score_candidate_lists(
        recall, proposals, base, group, data, seed, args.device)
    x = feature_array(data, merged, rows, changed, rz, cz, base, group, args.device)
    model = load_ranker(ranker["checkpoint"], args.device)
    scores = fusion + chosen["scale"] * infer(model, x, args.device, batch_size=1)
    lists = {"candidates": merged}
    fixed, final = (ranks_for(lists, values)[0] for values in (fusion, scores))
    addition = set(merged[changed].tolist())
    result = dict(seed=seed, user_rid=int(data.user_rids[row]), budget=len(final),
                  auxiliary_added=len(addition), top10=[
                      dict(rank=rank, oid=int(data.catalog_oids[item]), from_auxiliary=int(item) in addition)
                      for rank, item in enumerate(final[:10], 1)
                  ])
    if args.show_target:
        target = sample["target"]
        ranks = {}
        for label, order in (("original", original[0]), ("augmented_prior", fixed), ("dcn_v2", final)):
            positions = np.flatnonzero(order == target)
            ranks[label] = int(positions[0]) + 1 if len(positions) else None
        result["target_audit"] = dict(oid=int(data.catalog_oids[target]), ranks=ranks,
                                     from_auxiliary=target in addition)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/channel_rank.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, choices=(2026, 2027), default=2026)
    p.add_argument("--cohort", choices=("dev", "fresh"), default="fresh")
    user = p.add_mutually_exclusive_group()
    user.add_argument("--index", type=int, default=0)
    user.add_argument("--rid", type=int)
    p.add_argument("--show-target", action="store_true")
    args = p.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(4)
    print(json.dumps(recommend(args), indent=2))


if __name__ == "__main__":
    main()
