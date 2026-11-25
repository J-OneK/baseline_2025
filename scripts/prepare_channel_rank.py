"""Prepare disjoint ranking cohorts and real frozen Top-500 candidate lists."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from group_recall_study import arrays, frozen_candidates, user_history
from taac_retrieval.data import PreparedData
from taac_retrieval.group_recall import RareAttributeIndex
from taac_retrieval.ranking_data import read, write_new
from taac_retrieval.ranking_model import digest


def partition(config, group, data):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    old_file = Path("reports/ranking_study/confirmation.json")
    prior = read(old_file)["cohort_identity"]
    excluded = [
        np.asarray(data.user_rids)[np.asarray(data.eval_indices)],
        data.user_rids[np.load("artifacts/ranking_confirmation_v1/cohort.npy")]
    ]
    for source in prior["exclusion_sources"]:
        assert digest(source["indices"]) == source["indices_sha256"]
        assert digest(source["user_rids"]) == source["user_rids_sha256"]
        excluded.append(np.load(source["user_rids"])[np.load(source["indices"])])
    previous = Path(group["outputs"]) / "confirm"
    approval = read(previous / "cohort.json")
    assert digest(previous / "rows.npy") == approval["rows_sha256"]
    dev = np.load(previous / "rows.npy")
    excluded.append(data.user_rids[dev])
    available = np.flatnonzero(~np.isin(data.user_rids, np.unique(np.concatenate(excluded))))
    rng = np.random.default_rng(config["cohort_seed"])
    chosen = rng.choice(available, config["fit_users"] + config["confirmation_users"], replace=False)
    splits = dict(fit=np.sort(chosen[:config["fit_users"]]), dev=dev, fresh=np.sort(chosen[config["fit_users"]:]))
    for name, rows in splits.items():
        (root / name).mkdir()
        np.save(root / name / "rows.npy", rows)
    write_new(root / "cohorts.json", dict(config=config, previous_cohorts_source_sha256=digest(old_file),
        fit_population=int(len(available)), cohorts={
            name:dict(users=len(rows), rows_sha256=digest(root / name / "rows.npy")) for name, rows in splits.items()
        }, excluded_overlap=0, timestamp_split="strict per-user upstream validation boundary",
        test_evaluated=False))
    print(json.dumps({m:len(r) for m,r in splits.items()}), flush=True)


def selected_rows(config, split):
    root = Path(config["output"])
    cohort = read(root / "cohorts.json")
    assert cohort["config"] == config
    source = root / split / "rows.npy"
    assert cohort["cohorts"][split]["rows_sha256"] == digest(source)
    if split == "fresh":
        selection = read(root / "selection.json")
        assert selection["fresh_rows_sha256"] == digest(source)
    return np.load(source)


def proposal_chunk(work):
    history, actions, group = work
    return np.stack([
        _INDEX.propose(items, acted, group["recent_events"], group["proposal_count_per_bucket"], group["click_bonus"])[1]
        for items, acted in zip(history, actions)
    ])


def init_index(index):
    global _INDEX
    _INDEX = index


def prepare_proposals(config, group, data, split, workers):
    rows = selected_rows(config, split)
    out = Path(config["output"]) / split / "proposals.npy"
    if out.exists():
        raise FileExistsError(out)
    if split == "dev":
        old = Path(group["outputs"]) / "confirm"
        np.testing.assert_array_equal(rows, np.load(old / "rows.npy"))
        values = np.load(old / "proposals.npy")[:, 1]
    else:
        attributes = np.load(data.root / "item_sparse.npy", mmap_mode="r")
        content = np.load(data.root / "item_mm81.npy")
        norm = np.linalg.norm(content, axis=1, keepdims=True)
        np.divide(content, norm, out=content, where=norm > 0)
        columns = [data.metadata["item_fields"].index(field) for field in group["fields"]]
        index = RareAttributeIndex(attributes, content, data.train_counts, columns, group["max_posting"])
        history, actions = user_history(data, rows)
        tasks = [(history[i:i+128], actions[i:i+128], group) for i in range(0,len(rows),128)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
                                 initializer=init_index, initargs=(index,)) as pool:
            values = np.concatenate(list(pool.map(proposal_chunk, tasks)))
    np.save(out, values)
    print(f"{split} content proposals ready for {len(rows)} users.", flush=True)


def prepare_retrieval(config, base, group, data, split, seed, device):
    rows = selected_rows(config, split)
    folder = Path(base["root"]) / f"features_seed{seed}"
    if split == "dev":
        source = Path(group["outputs"]) / "confirm" / f"recall_seed{seed}.npz"
        recall = arrays(source)
        np.testing.assert_array_equal(recall["rows"], rows)
        return source
    source = Path(config["output"]) / split / f"recall_seed{seed}.npz"
    frozen_candidates(folder, data, rows, source, device)
    print(f"{split} seed{seed} Top-500 ready for {len(rows)} users.", flush=True)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/channel_rank.json")
    parser.add_argument("--stage", choices=["cohorts","proposals","retrieval"], required=True)
    parser.add_argument("--split", choices=["fit","dev","fresh"], default="fit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read(args.config)
    base, group = read(config["ranking_config"]), read(config["group_config"])
    data = PreparedData(base["cache"])
    if args.stage == "cohorts":
        partition(config, group, data)
    elif args.stage == "proposals":
        prepare_proposals(config, group, data, args.split, args.workers)
    else:
        prepare_retrieval(config, base, group, data, args.split, args.seed, args.device)


if __name__ == "__main__":
    main()
