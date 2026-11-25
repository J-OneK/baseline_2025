"""Diagnose ranking-stage losses and evaluate a grouped content recall channel."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from finish_ranking_study import paired
from taac_retrieval.content_ranker import blend_scores
from taac_retrieval.content_ranker_train import load_content_ranker
from taac_retrieval.coverage import coverage_summary
from taac_retrieval.data import PreparedData, SequenceDataset
from taac_retrieval.evaluate_checkpoint import load_model
from taac_retrieval.group_recall import RareAttributeIndex, merge_candidates
from taac_retrieval.metrics import exact_topk, to_device
from taac_retrieval.ranking_data import history_clicks, read, write_new
from taac_retrieval.ranking_model import digest


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def bucket_masks(data, targets):
    frequency = data.train_counts[targets]
    return {
        "all": np.ones(len(frequency), dtype=bool),
        "cold_0": frequency == 0,
        "tail_1_5": (frequency >= 1) & (frequency <= 5),
        "mid_6_50": (frequency >= 6) & (frequency <= 50),
        "head_gt50": frequency > 50,
    }


def user_history(data, rows):
    observed = np.arange(data.max_history) < data.val_cut[rows, None]
    return np.where(observed, data.items[rows], 0), np.where(observed, data.actions[rows], 0)


def diagnose(base, data):
    rows = np.asarray(data.eval_indices)
    target = data.items[rows, data.val_pos[rows]]
    history, actions = user_history(data, rows)
    attrs = np.load(data.root / "item_sparse.npy", mmap_mode="r")
    present = np.load(data.root / "mm_present.npy")
    training = np.arange(data.max_history) < data.train_cut[:, None]
    clicks = np.bincount(data.items[training & (data.actions == 2)], minlength=data.num_items + 1)
    stats = {}
    by_seed = {}
    fields = data.metadata["item_fields"]
    for name, mask in bucket_masks(data, target).items():
        current = target[mask]
        stats[name] = dict(users=len(current), content_present=int(present[current].sum()),
                           zero_training_clicks=int((clicks[current] == 0).sum()), fields={})
        for field in ("102", "121", "122"):
            j = fields.index(field)
            values = attrs[current, j]
            matched = (attrs[history[mask], j] == values[:, None]) & (values[:, None] > 0)
            counts = np.bincount(attrs[1:, j])
            any_match = matched.any(1)
            stats[name]["fields"][field] = dict(
                history_matches=int(any_match.sum()),
                clicked_history_matches=int((matched & (actions[mask] == 2)).any(1).sum()),
                matched_full_catalog_posting_median=float(np.median(counts[values[any_match]]))
            )
    for seed in base["seeds"]:
        feature_root = Path(base["root"]) / f"features_seed{seed}"
        recall = arrays(feature_root / "validation_candidates.npz")
        rank_dir = Path("artifacts/ranking_content_v1/training") / f"mean_mlp_seed{seed}"
        epoch = read(rank_dir / "result.json")["best_epoch"]
        ranking = arrays(rank_dir / f"validation_epoch{epoch}.npz")
        generated = np.load(f"artifacts/slate_ranking_v1/seed{seed}/autoregressive_epoch5.npy")
        np.testing.assert_array_equal(recall["rows"], rows)
        np.testing.assert_array_equal(recall["targets"], target)
        masks = bucket_masks(data, target)
        score_dict = {
            "recall_top500": recall["candidates"],
            "mlp_top100": ranking["predictions"][:, :100],
            "mlp_top10": ranking["predictions"][:, :10],
            "generative_top10": generated,
        }
        by_seed[str(seed)] = {}
        for stage, predictions in score_dict.items():
            hits = (predictions == target[:, None]).any(1)
            by_seed[str(seed)][stage] = {name:int(hits[mask].sum()) for name, mask in masks.items()}
    return dict(development_users=len(rows), target_statistics=stats,
                hits_by_seed_and_stage=by_seed, train_click_counts_use_only_train_prefix=True)


@torch.inference_mode()
def score_mlp(ranker, pool, rows, histories, device):
    pieces = []
    limit = max(1, 8192 // pool.shape[1])
    for start in range(0, len(rows), limit):
        selection = slice(start, start + limit)
        ids = torch.as_tensor(pool[selection], device=device)
        user_rows = torch.as_tensor(rows[selection], device=device)
        clicks = torch.as_tensor(histories[selection], device=device)
        n, k = ids.shape
        scores = ranker(None, user_rows[:, None].expand(n, k).reshape(-1), ids.reshape(-1),
                        clicks[:, None].expand(n, k, -1).reshape(n * k, -1))
        pieces.append(scores.reshape(n, k).cpu().numpy())
    return np.concatenate(pieces)


@torch.inference_mode()
def frozen_candidates(feature_root, data, rows, save_path, device):
    if save_path.exists():
        cached = arrays(save_path)
        np.testing.assert_array_equal(cached["rows"], rows)
        return cached
    model, evaluation, _ = load_model(read(feature_root / "metadata.json")["checkpoint"], data.root, device)
    evaluation.set_eval_indices(rows)
    vectors = torch.as_tensor(np.load(feature_root / "item_vectors.npy"), device=device)
    outputs = {key:[] for key in ("candidates", "queries", "scores", "rows", "targets")}
    for batch in DataLoader(SequenceDataset(evaluation, "val"), batch_size=128, shuffle=False):
        batch = to_device(batch, device)
        query = model.predict(batch)
        found = exact_topk(query, vectors, batch["items"], 500)
        score = (query[:, None] * vectors[found]).sum(-1)
        for key, value in (("candidates", found), ("queries", query), ("scores", score),
                           ("rows", batch["row"]), ("targets", batch["target"])):
            outputs[key].append(value.cpu().numpy())
    cached = {key:np.concatenate(values) for key, values in outputs.items()}
    np.savez_compressed(save_path, **cached)
    return cached


def summary(predictions, target, rows, data):
    return coverage_summary(predictions, target, rows, data.train_counts, ks=(10, 100, 500))


def proposals_for(config, data, history, actions):
    features = np.load(data.root / "item_sparse.npy", mmap_mode="r")
    content = np.load(data.root / "item_mm81.npy")
    norms = np.linalg.norm(content, axis=1, keepdims=True)
    np.divide(content, norms, out=content, where=norms > 0)
    columns = [data.metadata["item_fields"].index(name) for name in config["fields"]]
    index = RareAttributeIndex(features, content, data.train_counts, columns, config["max_posting"])
    return np.stack([
        index.propose(seen, acted, config["recent_events"], config["proposal_count_per_bucket"], config["click_bonus"])
        for seen, acted in zip(history, actions)
    ])


def run(config, base, data, root, rows, device, confirm=False):
    target = data.items[rows, data.val_pos[rows]]
    history, action = user_history(data, rows)
    if (history == target[:, None]).any():
        raise ValueError("A held-out target is present in its own history.")
    proposed = proposals_for(config, data, history, action)
    np.save(root / "proposals.npy", proposed)
    print(f"Scored attribute-proposed candidates for {len(rows)} users.", flush=True)
    click_history = np.stack([
        history_clicks(data.items[row], data.actions[row], int(data.val_cut[row]), base["data"]["click_history"])
        for row in rows
    ])
    results = dict(users=len(rows), runs={}, means={}, differences={}, sources={})
    evidence = {}
    modes = ["match"] if confirm else config["modes"]
    quotas = [config["quotas"][group] for group in ("cold", "tail")]
    for seed in base["seeds"]:
        feature_root = Path(base["root"]) / f"features_seed{seed}"
        rank_dir = Path("artifacts/ranking_content_v1/training") / f"mean_mlp_seed{seed}"
        rank_checkpoint = rank_dir / "best.pt"
        ranker, ranking_config, checkpoint = load_content_ranker(rank_checkpoint, device)
        alpha = ranking_config["blend"]["alpha"]
        if confirm:
            recall = frozen_candidates(feature_root, data, rows, root / f"recall_seed{seed}.npz", device)
            ctr = score_mlp(ranker, recall["candidates"], rows, click_history, device)
            expected_rank = None
        else:
            recall = arrays(feature_root / "validation_candidates.npz")
            rank_file = rank_dir / f"validation_epoch{checkpoint['epoch']}.npz"
            rank_data = arrays(rank_file)
            ctr = rank_data["ctr_logits"]
            expected_rank = rank_data["predictions"]
            results["sources"][str(seed)] = dict(
                ranking_validation_sha256=digest(rank_file), ranking_checkpoint_sha256=digest(rank_checkpoint),
                retrieval_validation_sha256=digest(feature_root / "validation_candidates.npz")
            )
        np.testing.assert_array_equal(recall["rows"], rows)
        np.testing.assert_array_equal(recall["targets"], target)
        raw_order, raw_score = recall["candidates"], recall["scores"]
        base_scores = blend_scores(raw_score, ctr, alpha)
        indices = np.argsort(-base_scores, axis=1, kind="stable")
        base_ranked = np.take_along_axis(raw_order, indices, axis=1)
        if expected_rank is not None:
            np.testing.assert_array_equal(base_ranked, expected_rank)
        masks = bucket_masks(data, target)
        base_groups, base_individual = summary(base_ranked, target, rows, data)
        evidence[seed] = {"baseline": base_individual}
        results["runs"][str(seed)] = {"baseline": base_groups}
        output = {"baseline": base_ranked, "rows": rows, "targets": target}
        vectors = np.load(feature_root / "item_vectors.npy", mmap_mode="r")
        for mode in modes:
            pool = np.stack([
                merge_candidates(items, np.isin(items, ranked[:config["protected_top"]]),
                                 data.train_counts, proposal, quotas)
                for items, ranked, proposal in zip(raw_order, base_ranked, proposed[:, config["modes"].index(mode)])
            ])
            changed = pool != raw_order
            user_indices, slots = np.where(changed)
            padded = np.zeros((len(rows), sum(quotas)), dtype=pool.dtype)
            positions = np.cumsum(changed, axis=1)[user_indices, slots] - 1
            padded[user_indices, positions] = pool[user_indices, slots]
            added_ctr = score_mlp(ranker, padded, rows, click_history, device)
            score = base_scores.copy()
            for field, predictions, weight in (
                (raw_score, np.einsum("bd,bd->b", recall["queries"][user_indices], vectors[pool[changed]]), 1-alpha),
                (ctr, added_ctr[user_indices, positions], alpha),
            ):
                scale = field.std(1)[user_indices]
                normalized = (predictions - field.mean(1)[user_indices]) / np.where(scale < 1e-6, 1., scale)
                if field is raw_score:
                    score[changed] = weight * normalized
                else:
                    score[changed] += weight * normalized
            merged_ranked = np.take_along_axis(pool, np.argsort(-score, axis=1, kind="stable"), axis=1)
            for items, seen, previously in zip(merged_ranked, history, base_ranked):
                if np.intersect1d(items, seen).size or not np.isin(previously[:config["protected_top"]], items).all():
                    raise ValueError("Candidate merge violated history filtering or the protected prefix.")
            groups, individual = summary(merged_ranked, target, rows, data)
            changed_counts = {}
            for group, mask in masks.items():
                changed_counts[group] = {}
                for k in (10, 100, 500):
                    previous = base_individual[f"hits_at_{k}"].astype(bool)
                    current = individual[f"hits_at_{k}"].astype(bool)
                    changed_counts[group][str(k)] = dict(gained=int((current & ~previous & mask).sum()),
                                                          lost=int((previous & ~current & mask).sum()))
            result = dict(groups=groups, changes=changed_counts, candidates_replaced=int(changed.sum()),
                          users_with_top10_change=int((merged_ranked[:, :10] != base_ranked[:, :10]).any(1).sum()))
            results["runs"][str(seed)][mode] = result
            evidence[seed][mode] = individual
            output[mode] = merged_ranked
        del ranker
        np.savez_compressed(root / f"predictions_seed{seed}.npz", **output)
        print(f"Evaluated seed {seed}.", flush=True)
    for mode in ["baseline", *modes]:
        runs = [run[mode] if mode == "baseline" else run[mode]["groups"] for run in results["runs"].values()]
        results["means"][mode] = {
            group:{key:float(np.mean([run[group][key] for run in runs]))
                   for key in ("Recall@500", "Recall@100", "Recall@10", "NDCG@10")}
            for group, mask in bucket_masks(data, target).items() if mask.any()
        }
    for mode in modes:
        results["differences"][mode] = {}
        for group, mask in bucket_masks(data, target).items():
            if not mask.any():
                continue
            results["differences"][mode][group] = {}
            for key in ("hits_at_10", "ndcg_at_10", "hits_at_100", "hits_at_500"):
                delta = np.stack([r[mode][key][mask].astype(float) - r["baseline"][key][mask].astype(float) for r in evidence.values()])
                summary_delta = paired(delta)
                summary_delta["distinct_gained_users"] = int((delta > 0).any(0).sum())
                summary_delta["distinct_lost_users"] = int((delta < 0).any(0).sum())
                results["differences"][mode][group][key] = summary_delta
    return results


def fresh_rows(data, config_path, root):
    prior = read("reports/ranking_study/confirmation.json")["cohort_identity"]
    excluded = [np.asarray(data.user_rids)[np.asarray(data.eval_indices)]]
    sources = []
    for source in prior["exclusion_sources"]:
        assert digest(source["indices"]) == source["indices_sha256"]
        assert digest(source["user_rids"]) == source["user_rids_sha256"]
        excluded.append(np.load(source["user_rids"])[np.load(source["indices"])])
        sources.append(source["indices"])
    previous = Path("artifacts/ranking_confirmation_v1/cohort.npy")
    assert digest(previous) == prior["cohort_sha256"]
    excluded.append(data.user_rids[np.load(previous)])
    sources.append(str(previous))
    eligible = np.flatnonzero(~np.isin(data.user_rids, np.unique(np.concatenate(excluded))))
    rows = np.sort(np.random.default_rng(6239).choice(eligible, 5000, replace=False))
    np.save(root / "rows.npy", rows)
    write_new(root / "cohort.json", dict(seed=6239, users=len(rows), eligible_users=len(eligible),
        rows_sha256=digest(root / "rows.npy"), exclusion_sources=sources,
        overlap_previous_evaluations=0, protocol="Validation targets of users outside identified older cohorts",
        config_sha256=digest(config_path)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/group_recall.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("diagnose", "develop", "confirm"), default="develop")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read(args.config)
    base = read(config["ranking_config"])
    data = PreparedData(base["cache"])
    if args.stage == "diagnose":
        result = diagnose(base, data)
        print(json.dumps(result, indent=2))
        if args.report:
            write_new(args.report, result)
        return
    root = Path(config["outputs"]) / args.stage
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(config=config, registered_at=datetime.now(timezone.utc).isoformat(),
                inputs_sha256={name:digest(data.root / name) for name in
                    ["item_sparse.npy", "item_mm81.npy", "items.npy", "actions.npy", "train_counts.npy", "val_cut.npy"]},
                source_sha256={name:digest(name) for name in
                    ["scripts/group_recall_study.py", "taac_retrieval/group_recall.py", args.config]},
                test_evaluated=False)
    write_new(root / "plan.json", plan)
    if args.stage == "confirm":
        development = read(Path(config["outputs"]) / "develop/result.json")
        assert development["confirmation_gate"]["passed"]
        assert development["plan"]["source_sha256"] == plan["source_sha256"]
        rows = fresh_rows(data, args.config, root)
    else:
        rows = np.asarray(data.eval_indices)
    result = run(config, base, data, root, rows, args.device, confirm=args.stage == "confirm")
    result["plan"] = plan
    if args.stage == "confirm":
        result["cohort"] = read(root / "cohort.json")
    else:
        fixed = result["means"]["match"]
        baseline = result["means"]["baseline"]
        comparisons = [
            fixed["all"]["Recall@500"] >= baseline["all"]["Recall@500"],
            result["differences"]["match"]["cold_0"]["hits_at_500"]["distinct_gained_users"] >= 3,
            fixed["cold_0"]["Recall@10"] >= baseline["cold_0"]["Recall@10"],
            fixed["tail_1_5"]["Recall@10"] >= baseline["tail_1_5"]["Recall@10"],
        ]
        result["confirmation_gate"] = dict(passed=all(comparisons), overall_budget=comparisons[0],
            cold_gains=comparisons[1], cold_top10=comparisons[2], tail_top10=comparisons[3])
    write_new(root / "result.json", result)
    if args.report:
        write_new(args.report, result)
    print(json.dumps(result["means"], indent=2))


if __name__ == "__main__":
    main()
