"""Train, lock on development users, then evaluate one locked ranker."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from finish_ranking_study import paired
from group_recall_study import arrays
from prepare_channel_rank import selected_rows
from taac_retrieval.channel_features import BASIC, FEATURES
from taac_retrieval.channel_ranker import ChannelRanker, list_loss
from taac_retrieval.coverage import coverage_summary
from taac_retrieval.data import PreparedData
from taac_retrieval.ranking_data import read, write_new
from taac_retrieval.ranking_model import digest

VARIANTS = ("basic_mlp", "channel_mlp", "channel_dcn_v2")
CODE = ("scripts/prepare_channel_rank.py", "scripts/build_channel_features.py",
        "scripts/run_channel_rank.py", "taac_retrieval/channel_features.py",
        "taac_retrieval/channel_ranker.py")
RARE = ("cold_0", "tail_1_5")
PAIRED_GROUPS = ("all", *RARE, "mid_6_50", "head_gt50")
SELECTION_RULE = (
    "Compare mean development NDCG@10, then HR@10, then earlier epoch, then lower "
    "residual scale, then listed architecture order. Eligible arms strictly improve "
    "NDCG@10 and preserve or improve HR@10 and combined rare Top10 hits relative to "
    "the fixed augmented baseline. If none qualify, confirm the highest NDCG arm "
    "as a diagnostic without claiming development eligibility."
)


def ranks_for(lists, scores):
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(lists["candidates"], order, axis=1)


def evaluate(lists, prediction, counts):
    return coverage_summary(prediction, lists["targets"], lists["rows"], counts)


def snapshot(lists, baseline, learned, counts):
    result, per_user = evaluate(lists, learned, counts)
    _, before = evaluate(lists, baseline, counts)
    aux_positive = ((lists["candidates"] == lists["targets"][:, None]) & lists["added"]).any(1)
    return result, per_user, dict(
        new_channel_targets=int(aux_positive.sum()),
        new_channel_top10=int(per_user["hits_at_10"][aux_positive].sum()),
        new_channel_top100=int(per_user["hits_at_100"][aux_positive].sum()),
        gains_top10=int((per_user["hits_at_10"] > before["hits_at_10"]).sum()),
        losses_top10=int((per_user["hits_at_10"] < before["hits_at_10"]).sum()),
    )


@torch.inference_mode()
def infer(model, x, device, batch_size):
    model.eval()
    return np.concatenate([
        model(torch.as_tensor(x[i:i + batch_size], device=device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ])


def stage_fit(config, counts, seed, device):
    root = Path(config["output"])
    if (root / "selection.json").exists():
        raise FileExistsError("Protocol was locked. Use a new experiment for changes.")
    log_path = root / f"development_seed{seed}.json"
    if log_path.exists():
        raise FileExistsError(log_path)
    fit, dev = (arrays(root / split / f"lists_seed{seed}.npz") for split in ("fit", "dev"))
    xfit, xdev = (np.load(root / split / f"x_seed{seed}.npy") for split in ("fit", "dev"))
    np.testing.assert_array_equal(dev["rows"], selected_rows(config, "dev"))
    assert not np.intersect1d(fit["rows"], dev["rows"]).size
    matches = fit["candidates"] == fit["targets"][:, None]
    assert (matches.sum(1) == 1).all()
    target = torch.as_tensor(matches.argmax(1), device=device)
    weights = torch.as_tensor(np.where(counts[fit["targets"]] <= 5, config["rare_weight"], 1),
                              device=device, dtype=torch.float32)
    xfit = torch.as_tensor(xfit, device=device)
    prior = torch.as_tensor(fit["prior"], device=device, dtype=torch.float32)
    std, mean = torch.std_mean(xfit, dim=(0, 1), correction=0)
    std = std.clamp_min(0.05)
    baselines = {name:evaluate(dev, dev[name], counts)[0] for name in ("original", "fixed")}
    results = []
    for variant in VARIANTS:
        torch.manual_seed(seed)
        features = len(BASIC) if variant == "basic_mlp" else len(FEATURES)
        rank = config["cross_rank"] if variant == "channel_dcn_v2" else None
        model = ChannelRanker(mean[:features], std[:features], config["width"], rank).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        shuffle = torch.Generator(device=device).manual_seed(seed)
        ckpt = root / "checkpoints" / f"{variant}_seed{seed}"
        ckpt.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            losses = []
            for batch in torch.randperm(len(xfit), generator=shuffle, device=device).split(config["batch_size"]):
                optim.zero_grad(set_to_none=True)
                residual = model(xfit[batch])
                loss = list_loss(prior[batch], residual, target[batch], weights[batch], config["l2_residual"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                losses.append(loss.detach())
            loss = float(torch.stack(losses).mean().cpu())
            assert np.isfinite(loss)
            path = ckpt / f"epoch{epoch}.pt"
            torch.save(dict(state=model.state_dict(), variant=variant, epoch=epoch, seed=seed,
                            width=config["width"], cross_rank=rank), path)
            sha = digest(path)
            update = infer(model, xdev, device, config["batch_size"])
            assert np.isfinite(update).all()
            for scale in config["residual_scales"]:
                prediction = ranks_for(dev, dev["prior"] + scale * update)
                metrics, _, changes = snapshot(dev, dev["fixed"], prediction, counts)
                results.append(dict(variant=variant, epoch=epoch, scale=scale, loss=loss,
                                    checkpoint=str(path), sha256=sha, metrics=metrics, changes=changes))
            last = metrics["all"]
            print(f"{seed} {variant} epoch {epoch} loss {loss:.4f} HR {last['Recall@10']:.4%} "
                  f"NDCG {last['NDCG@10']:.4%}", flush=True)
    write_new(log_path, dict(seed=seed, config=config, selection_rule=SELECTION_RULE,
                             source_digests={path:digest(path) for path in CODE},
                             baselines=baselines, runs=results))


def means(metrics):
    return dict(
        hr10=float(np.mean([m["all"]["Recall@10"] for m in metrics])),
        ndcg10=float(np.mean([m["all"]["NDCG@10"] for m in metrics])),
        rare_top10_hits=sum(m[group]["hits_at_10"] for m in metrics for group in RARE),
        recall500=float(np.mean([m["all"]["Recall@500"] for m in metrics])),
    )


def stage_lock(config):
    root = Path(config["output"])
    output = root / "selection.json"
    if output.exists():
        raise FileExistsError(output)
    reports = [read(root / f"development_seed{s}.json") for s in config["seeds"]]
    source_digests = {path:digest(path) for path in CODE}
    for report in reports:
        assert report["config"] == config and report["selection_rule"] == SELECTION_RULE
        assert report["source_digests"] == source_digests
    baselines = {
        name:means([r["baselines"][name] for r in reports]) for name in ("original", "fixed")
    }
    fixed = baselines["fixed"]
    candidates = []
    for runs in zip(*(r["runs"] for r in reports), strict=True):
        first = runs[0]
        key = {name:first[name] for name in ("variant", "epoch", "scale")}
        assert all(all(run[name] == val for name, val in key.items()) for run in runs)
        aggregate = means([run["metrics"] for run in runs])
        eligible = (aggregate["ndcg10"] > fixed["ndcg10"] and aggregate["hr10"] >= fixed["hr10"]
                    and aggregate["rare_top10_hits"] >= fixed["rare_top10_hits"])
        candidates.append(dict(**key, **aggregate, eligible=eligible,
                               new_channel_top10=sum(run["changes"]["new_channel_top10"] for run in runs),
                               checkpoints={str(s):{k:run[k] for k in ("checkpoint", "sha256")}
                                            for s,run in zip(config["seeds"], runs)}))
    order = lambda arm: (
        arm["eligible"], arm["ndcg10"], arm["hr10"], -arm["epoch"], -arm["scale"],
        -VARIANTS.index(arm["variant"]),
    )
    chosen = max(candidates, key=order)
    cohorts = read(root / "cohorts.json")
    write_new(output, dict(selection_rule=SELECTION_RULE, selected=chosen,
                           best_by_variant={v:max((a for a in candidates if a["variant"] == v), key=order)
                                            for v in VARIANTS},
                           baselines=baselines, source_digests=source_digests,
                           total_configurations=len(candidates), all_configurations=candidates,
                           fresh_rows_sha256=cohorts["cohorts"]["fresh"]["rows_sha256"]))
    print(chosen, flush=True)


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    state = saved["state"]
    model = ChannelRanker(state["mean"], state["std"], saved["width"], saved["cross_rank"]).to(device)
    model.load_state_dict(state)
    return model


def stage_fresh(config, counts, device):
    root = Path(config["output"])
    path = root / "confirmation.json"
    if path.exists():
        raise FileExistsError(path)
    lock = read(root / "selection.json")
    for name, sha in lock["source_digests"].items():
        assert digest(name) == sha, name
    cohort = selected_rows(config, "fresh")
    chosen = lock["selected"]
    runs, observations = {}, {}
    for seed in config["seeds"]:
        lists = arrays(root / "fresh" / f"lists_seed{seed}.npz")
        np.testing.assert_array_equal(lists["rows"], cohort)
        x = np.load(root / "fresh" / f"x_seed{seed}.npy")
        ckpt = chosen["checkpoints"][str(seed)]
        assert digest(ckpt["checkpoint"]) == ckpt["sha256"]
        model = load_model(ckpt["checkpoint"], device)
        update = infer(model, x, device, config["batch_size"])
        prediction = ranks_for(lists, lists["prior"] + chosen["scale"] * update)
        np.testing.assert_array_equal(np.sort(lists["fixed"], axis=1), np.sort(prediction, axis=1))
        metrics, learned, changes = snapshot(lists, lists["fixed"], prediction, counts)
        prior_metrics, prior = evaluate(lists, lists["fixed"], counts)
        original_metrics, original = evaluate(lists, lists["original"], counts)
        del model
        runs[str(seed)] = dict(learned=metrics, fixed=prior_metrics, original=original_metrics, changes=changes)
        observations[seed] = dict(learned=learned, fixed=prior, original=original)
        np.savez(root / "fresh" / f"predictions_seed{seed}.npz", rows=cohort, targets=lists["targets"],
                 target_frequency=counts[lists["targets"]], learned_rank=learned["target_rank"],
                 fixed_rank=prior["target_rank"], original_rank=original["target_rank"],
                 added_target=((lists["candidates"] == lists["targets"][:, None]) & lists["added"]).any(1),
                 top10=prediction[:, :10])
    comparisons = {}
    frequency = observations[config["seeds"][0]]["learned"]["target_train_frequency"]
    masks = (frequency >= 0, frequency == 0, (frequency >= 1) & (frequency <= 5),
             (frequency >= 6) & (frequency <= 50), frequency > 50)
    for after, before in (("learned", "fixed"), ("learned", "original"), ("fixed", "original")):
        comparisons[f"{after}_vs_{before}"] = {}
        for name, mask in zip(PAIRED_GROUPS, masks):
            comparisons[f"{after}_vs_{before}"][name] = {
                metric:paired(np.stack([
                    obs[after][metric][mask].astype(float) - obs[before][metric][mask]
                    for obs in observations.values()
                ])) for metric in ("hits_at_10", "ndcg_at_10", "hits_at_500")
            }
    aggregate = {name:means([run[name] for run in runs.values()]) for name in ("original", "fixed", "learned")}
    write_new(path, dict(users=len(cohort), seeds=config["seeds"], lock_sha256=digest(root / "selection.json"),
                         row_sha256=digest(root / "fresh" / "rows.npy"), selected=chosen, means=aggregate,
                         comparisons=comparisons, runs=runs, test_evaluated=False))
    print(aggregate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/channel_rank.json")
    parser.add_argument("--stage", choices=("fit", "lock", "fresh"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read(args.config)
    if args.stage == "lock":
        stage_lock(config)
        return
    counts = PreparedData(read(config["ranking_config"])["cache"]).train_counts
    if args.stage == "fit":
        if args.seed not in config["seeds"]:
            parser.error("--stage fit requires a configured --seed")
        stage_fit(config, counts, args.seed, args.device)
    else:
        stage_fresh(config, counts, args.device)


if __name__ == "__main__":
    main()
