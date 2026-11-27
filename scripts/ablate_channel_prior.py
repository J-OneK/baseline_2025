"""Isolate the fixed fusion score added to DCN V2 on frozen candidate lists."""
import argparse
from pathlib import Path

import numpy as np
import torch

from run_channel_rank import (
    ChannelRanker, FEATURES, arrays, evaluate, infer, list_loss, load_model,
    means, paired, ranks_for, read, snapshot, write_new,
)
from taac_retrieval.ranking_model import digest


ARMS = {"fusion_plus_dcn": 1, "dcn_only": 0}
CONFIRM = {
    "channel_6k": ("configs/channel_rank.json", "fresh"),
    "sequence_10k": ("configs/sequence_rank.json", "fresh"),
}
RULE = "For each arm, maximize mean dev NDCG@10 across seeds, then HR@10, then prefer the earlier epoch. Score coefficients fixed at 1."


def inputs(root, split, seed):
    folder = root / split
    lists = arrays(folder / f"lists_seed{seed}.npz")
    features = np.load(folder / f"x_seed{seed}.npy", allow_pickle=False)
    metadata = read(folder / f"features_seed{seed}.json")
    assert metadata["features"] == list(FEATURES)
    assert features.shape == (len(lists["rows"]), 500, len(FEATURES))
    assert lists["candidates"].shape == features.shape[:2]
    assert np.isfinite(features).all() and np.isfinite(lists["prior"]).all()
    np.testing.assert_allclose(lists["prior"], 0.9 * features[..., 0] + 0.1 * features[..., 1], atol=2e-6)
    np.testing.assert_array_equal(ranks_for(lists, lists["prior"]), lists["fixed"])
    return lists, features


def artifacts(root, split, seed):
    folder = root / split
    return {str(file): digest(file) for file in (
        folder / "rows.npy", folder / f"lists_seed{seed}.npz",
        folder / f"x_seed{seed}.npy", folder / f"features_seed{seed}.json",
    )}


def fit(config, counts, output, device, epochs):
    root = Path(config["output"])
    cohorts = read(root / "cohorts.json")["cohorts"]
    for split in ("fit", "dev"):
        assert digest(root / split / "rows.npy") == cohorts[split]["rows_sha256"]
    runs, baselines, provenance = {}, {}, {}
    for seed in config["seeds"]:
        fit_lists, xfit = inputs(root, "fit", seed)
        dev, xdev = inputs(root, "dev", seed)
        assert not np.intersect1d(fit_lists["rows"], dev["rows"]).size
        assert np.isin(fit_lists["rows"], np.load(root / "fit/rows.npy")).all()
        np.testing.assert_array_equal(dev["rows"], np.load(root / "dev/rows.npy"))
        for split in ("fit", "dev"):
            provenance.update(artifacts(root, split, seed))
        matches = fit_lists["candidates"] == fit_lists["targets"][:, None]
        assert (matches.sum(1) == 1).all()
        target = torch.as_tensor(matches.argmax(1), device=device)
        weights = torch.as_tensor(np.where(counts[fit_lists["targets"]] <= 5, config["rare_weight"], 1),
                                  device=device, dtype=torch.float32)
        xfit = torch.as_tensor(xfit, device=device)
        prior = torch.as_tensor(fit_lists["prior"], device=device)
        std, mean = torch.std_mean(xfit, dim=(0, 1), correction=0)
        std = std.clamp_min(0.05)
        baselines[seed] = evaluate(dev, dev["fixed"], counts)[0]
        runs[seed] = {"fit_lists": len(xfit), "dev_users": len(xdev)}
        for arm, factor in ARMS.items():
            # Recreate the same starting weights and batch order for each arm.
            torch.manual_seed(seed)
            shuffle = torch.Generator(device=device).manual_seed(seed)
            model = ChannelRanker(mean, std, config["width"], config["cross_rank"]).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])
            anchor = prior if factor else torch.zeros_like(prior)
            folder = output / "checkpoints" / arm / str(seed)
            folder.mkdir(parents=True)
            records = runs[seed][arm] = []
            for epoch in range(1, epochs + 1):
                model.train()
                losses = []
                for batch in torch.randperm(len(xfit), generator=shuffle, device=device).split(config["batch_size"]):
                    optim.zero_grad(set_to_none=True)
                    learned = model(xfit[batch])
                    loss = list_loss(anchor[batch], learned, target[batch], weights[batch], config["l2_residual"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optim.step()
                    losses.append(loss.detach())
                update = infer(model, xdev, device, config["batch_size"])
                loss = float(torch.stack(losses).mean().cpu())
                assert np.isfinite(update).all() and np.isfinite(loss)
                scores = dev["prior"] + update if factor else update
                metrics = evaluate(dev, ranks_for(dev, scores), counts)[0]
                path = folder / f"epoch{epoch}.pt"
                torch.save(dict(state=model.state_dict(), epoch=epoch, seed=seed, arm=arm,
                                width=config["width"], cross_rank=config["cross_rank"]), path)
                records.append(dict(epoch=epoch, loss=loss, metrics=metrics,
                                    checkpoint=str(path), sha256=digest(path)))
                if epoch in (1, 5, 10, epochs):
                    print(seed, arm, epoch, means([metrics]), flush=True)
    selection = {}
    for arm in ARMS:
        candidates = []
        for records in zip(*(run[arm] for run in runs.values()), strict=True):
            candidates.append(dict(epoch=records[0]["epoch"], **means([r["metrics"] for r in records]),
                                   checkpoints={s: {k: r[k] for k in ("checkpoint", "sha256")}
                                                for s, r in zip(runs, records, strict=True)}))
        chosen = max(candidates, key=lambda a: (a["ndcg10"], a["hr10"], -a["epoch"]))
        selection[arm] = dict(best=chosen, epoch5=candidates[4], curve=candidates)
    write_new(output / "development.json", dict(rule=RULE, seeds=config["seeds"], epochs=epochs,
              baseline=means(list(baselines.values())), selection=selection, runs=runs, inputs_sha256=provenance))
    print("Locked", {arm: value["best"]["epoch"] for arm, value in selection.items()}, flush=True)


def confirm(config, counts, output, device):
    dev = read(output / "development.json")
    for path, sha in dev["inputs_sha256"].items():
        assert digest(path) == sha, path
    result, provenance = {}, {}
    source = Path(config["output"])
    used = np.union1d(np.load(source / "fit/rows.npy"), np.load(source / "dev/rows.npy"))
    for name, (config_path, split) in CONFIRM.items():
        root = Path(read(config_path)["output"])
        rows = np.load(root / split / "rows.npy")
        assert digest(root / split / "rows.npy") == read(root / "cohorts.json")["cohorts"][split]["rows_sha256"]
        assert len(np.unique(rows)) == len(rows) and not np.intersect1d(used, rows).size
        used = np.union1d(used, rows)
        runs, values = {}, {}
        for seed in config["seeds"]:
            lists, features = inputs(root, split, seed)
            np.testing.assert_array_equal(lists["rows"], rows)
            provenance.update(artifacts(root, split, seed))
            base = lists["fixed"]
            metric, local = evaluate(lists, base, counts)
            runs[seed], values[seed] = {"fusion": {"metrics": metric}}, {"fusion": local}
            for arm, factor in ARMS.items():
                for selected in ("best", "epoch5"):
                    ckpt = dev["selection"][arm][selected]["checkpoints"][str(seed)]
                    assert digest(ckpt["checkpoint"]) == ckpt["sha256"]
                    model = load_model(ckpt["checkpoint"], device)
                    update = infer(model, features, device, config["batch_size"])
                    assert np.isfinite(update).all()
                    scores = lists["prior"] + update if factor else update
                    pred = ranks_for(lists, scores)
                    np.testing.assert_array_equal(np.sort(pred, 1), np.sort(base, 1))
                    metric, local, changes = snapshot(lists, base, pred, counts)
                    tag = arm if selected == "best" else f"{arm}_epoch5"
                    runs[seed][tag] = dict(metrics=metric, changes=changes)
                    values[seed][tag] = local
            np.savez(output / f"{name}_ranks_seed{seed}.npz", rows=rows, targets=lists["targets"],
                     frequency=local["target_train_frequency"],
                     **{tag: per["target_rank"] for tag, per in values[seed].items()})
        frequency = local["target_train_frequency"]
        masks = dict(all=frequency >= 0, cold_0=frequency == 0,
                     tail_1_5=(frequency >= 1) & (frequency <= 5),
                     mid_6_50=(frequency >= 6) & (frequency <= 50), head_gt50=frequency > 50)
        comparisons = {}
        for challenger, control in (("dcn_only", "fusion_plus_dcn"), ("dcn_only_epoch5", "fusion_plus_dcn_epoch5"),
                                    ("dcn_only", "fusion"), ("fusion_plus_dcn", "fusion")):
            comparisons[f"{challenger}_vs_{control}"] = {
                group: {metric: paired(np.stack([
                    per[challenger][metric][mask].astype(float) - per[control][metric][mask]
                    for per in values.values()
                ])) for metric in ("hits_at_10", "ndcg_at_10")} for group, mask in masks.items()
            }
        summary = {arm: means([v[arm]["metrics"] for v in runs.values()]) for arm in next(iter(runs.values()))}
        result[name] = dict(users=len(rows), summary=summary, comparisons=comparisons, runs=runs)
        print(name, summary, flush=True)
    write_new(output / "confirmation.json", dict(development_sha256=digest(output / "development.json"),
              inputs_sha256=provenance, test_evaluated=False, cohorts=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/channel_prior_ablation_v1")
    parser.add_argument("--stage", choices=("fit", "confirm"), required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error("--epochs must include the original five-epoch budget")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = read("configs/channel_rank.json")
    cache = Path(read(config["ranking_config"])["cache"])
    counts = np.load(cache / "train_counts.npy", mmap_mode="r")
    output = Path(args.output)
    meta = dict(config=config, rule=RULE, seeds=config["seeds"], epochs=args.epochs,
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                torch=str(torch.__version__), source_sha256=digest(__file__),
                train_counts_sha256=digest(cache / "train_counts.npy"), confirm_cohorts=CONFIRM)
    if args.stage == "fit":
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "protocol.json", meta)
        fit(config, counts, output, device, args.epochs)
    else:
        locked = read(output / "protocol.json")
        for key in ("config", "rule", "source_sha256", "train_counts_sha256"):
            assert locked[key] == meta[key], key
        if (output / "confirmation.json").exists():
            raise FileExistsError(output / "confirmation.json")
        confirm(config, counts, output, device)


if __name__ == "__main__":
    main()
