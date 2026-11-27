"""Measure the incremental value of the joint history ranker on two frozen DCN bases."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from run_sequence_rank import (
    CandidateLists, PreparedData, content_lookup, digest, evaluate, infer,
    list_loss, load_history_ranker, load_model, make_model, means, paired,
    ranks_for, read, updates, write_new,
)


BASES = ("fusion_plus_dcn", "dcn_only")
VARIANT = "joint_transformer"
RULE = "Lock each base's previously dev-selected DCN. For each joint ranker, select epoch and scale by mean dev NDCG@10, then HR@10, earlier epoch, and smaller scale. Confirm on the same archived 10,000 users."


def baseline(lists, dcn, base, device):
    scores = infer(dcn, lists.x, device, batch_size=128)
    return lists.lists["prior"] + scores if base == "fusion_plus_dcn" else scores


def frozen_dcn(spec, seed, device):
    checkpoint = spec["checkpoints"][str(seed)]
    assert digest(checkpoint["checkpoint"]) == checkpoint["sha256"]
    return load_model(checkpoint["checkpoint"], device).eval()


def fit(config, parent, data, spec, root, device):
    lookup = content_lookup(data, device)
    runs = {}
    for seed in config["seeds"]:
        args = (seed, data, lookup, config["history_length"], device)
        train, dev = (CandidateLists(parent, split, *args) for split in ("fit", "dev"))
        np.testing.assert_array_equal(dev.lists["rows"], np.load(parent / "dev/rows.npy"))
        assert np.isin(train.lists["rows"], np.load(parent / "fit/rows.npy")).all()
        assert not np.intersect1d(train.lists["rows"], dev.lists["rows"]).size
        match = train.lists["candidates"] == train.lists["targets"][:, None]
        assert (match.sum(1) == 1).all()
        target = torch.as_tensor(match.argmax(1), device=device)
        weights = torch.as_tensor(np.where(data.train_counts[train.lists["targets"]] <= 5,
                                          config["rare_weight"], 1), device=device, dtype=torch.float32)
        runs[seed] = dict(train_lists=len(train), dev_users=len(dev), bases={})
        for base in BASES:
            dcn = frozen_dcn(spec[base], seed, device)
            prior = torch.as_tensor(baseline(train, dcn, base, device), device=device)
            dev_scores = baseline(dev, dcn, base, device)
            base_metrics = evaluate(dev.lists, ranks_for(dev.lists, dev_scores), data.train_counts)[0]
            torch.manual_seed(seed)
            model = make_model(config, VARIANT, dcn, lookup.shape[1], device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
            shuffle = torch.Generator(device=device).manual_seed(seed)
            folder = root / "checkpoints" / base / str(seed)
            folder.mkdir(parents=True)
            records = []
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                losses = []
                for batch in torch.randperm(len(train), device=device, generator=shuffle).split(config["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    residual = model(*train.inputs(batch))
                    loss = list_loss(prior[batch], residual, target[batch], weights[batch], config["residual_penalty"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    losses.append(loss.detach())
                loss = float(torch.stack(losses).mean().cpu())
                correction = updates(model, dev, config["batch_size"])
                assert np.isfinite(loss) and np.isfinite(correction).all()
                path = folder / f"epoch{epoch}.pt"
                torch.save(model.state_dict(), path)
                checkpoint = dict(path=str(path), sha256=digest(path))
                for scale in config["residual_scales"]:
                    pred = ranks_for(dev.lists, dev_scores + scale * correction)
                    metrics = evaluate(dev.lists, pred, data.train_counts)[0]
                    records.append(dict(epoch=epoch, scale=scale, loss=loss, metrics=metrics, checkpoint=checkpoint))
                print(seed, base, epoch, means([metrics]), flush=True)
            runs[seed]["bases"][base] = dict(baseline=base_metrics, records=records)
    locked = {}
    for base in BASES:
        results = [run["bases"][base] for run in runs.values()]
        arms = []
        for same in zip(*(r["records"] for r in results), strict=True):
            epoch, scale = same[0]["epoch"], same[0]["scale"]
            assert all((r["epoch"], r["scale"]) == (epoch, scale) for r in same)
            arms.append(dict(variant=VARIANT, epoch=epoch, scale=scale,
                             **means([r["metrics"] for r in same]),
                             checkpoints={str(s): r["checkpoint"] for s, r in zip(runs, same, strict=True)}))
        selected = max(arms, key=lambda a: (a["ndcg10"], a["hr10"], -a["epoch"], -a["scale"]))
        locked[base] = dict(selected=selected, arms=arms, baseline=means([r["baseline"] for r in results]))
    write_new(root / "development.json", dict(rule=RULE, runs=runs, locked=locked))
    print("Locked selection", {b: (v["selected"]["epoch"], v["selected"]["scale"]) for b, v in locked.items()}, flush=True)


@torch.inference_mode()
def latency(lists, dcn, joint, scale, base, device):
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    values = [lists.inputs(slice(i, i + 1)) for i in range(96)]
    timings = {"base": [], "with_joint": []}
    for i, features in enumerate(values):
        prior = torch.as_tensor(lists.lists["prior"][i:i+1], device=device)
        jobs = ("base", "with_joint") if i % 2 else ("with_joint", "base")
        for job in jobs:
            sync()
            start = time.perf_counter()
            s = dcn(features[0])
            if base == "fusion_plus_dcn":
                s = s + prior
            if job == "with_joint":
                s = s + scale * joint(*features)
            sync()
            if i >= 16:
                timings[job].append(1000 * (time.perf_counter() - start))
    return {name: {f"p{p}_ms": float(np.percentile(ms, p)) for p in (50, 95)} for name, ms in timings.items()}


def compare(values, after, before, groups):
    return {
        group: {metric: paired(np.stack([
            run[after][metric][mask].astype(float) - run[before][metric][mask] for run in values.values()
        ])) for metric in ("hits_at_10", "ndcg_at_10")} for group, mask in groups.items()
    }


def confirm(config, parent, data, spec, root, anchor_root, device):
    locked = read(root / "development.json")["locked"]
    eval_root = Path(config["output"])
    rows_path = eval_root / "fresh/rows.npy"
    assert digest(rows_path) == read(eval_root / "cohorts.json")["cohorts"]["fresh"]["rows_sha256"]
    rows = np.load(rows_path)
    assert len(np.unique(rows)) == config["confirmation_users"] == len(rows)
    for split in ("fit", "dev"):
        assert not np.intersect1d(rows, np.load(parent / split / "rows.npy")).size
    lookup = content_lookup(data, device)
    runs, values = {}, {}
    for seed in config["seeds"]:
        lists = CandidateLists(eval_root, "fresh", seed, data, lookup, config["history_length"], device)
        np.testing.assert_array_equal(rows, lists.lists["rows"])
        with np.load(anchor_root / f"sequence_10k_ranks_seed{seed}.npz") as known:
            np.testing.assert_array_equal(rows, known["rows"])
            historical = {b: known[b] for b in BASES}
        runs[seed], values[seed] = {}, {}
        for base in BASES:
            dcn = frozen_dcn(spec[base], seed, device)
            initial = baseline(lists, dcn, base, device)
            arm = locked[base]["selected"]
            joint = load_history_ranker(config, arm, seed, dcn, lookup.shape[1], device)
            residual = updates(joint, lists, config["batch_size"])
            assert np.isfinite(initial).all() and np.isfinite(residual).all()
            for name, scores in ((base, initial), (f"{base}_with_joint", initial + arm["scale"] * residual)):
                pred = ranks_for(lists.lists, scores)
                metrics, per_user = evaluate(lists.lists, pred, data.train_counts)
                runs[seed][name], values[seed][name] = metrics, per_user
            np.testing.assert_array_equal(values[seed][base]["target_rank"], historical[base])
            first, second = (values[seed][k] for k in (base, f"{base}_with_joint"))
            np.testing.assert_array_equal(first["hits_at_500"], second["hits_at_500"])
            runs[seed][f"{base}_timing"] = latency(lists, dcn, joint, arm["scale"], base, device)
        np.savez(root / f"ranks_seed{seed}.npz", rows=rows, targets=lists.lists["targets"],
                 frequency=values[seed][BASES[0]]["target_train_frequency"],
                 **{name: value["target_rank"] for name, value in values[seed].items()})
        print(seed, {name: means([runs[seed][name]]) for name in values[seed]}, flush=True)
    freq = next(iter(values.values()))[BASES[0]]["target_train_frequency"]
    groups = dict(all=freq >= 0, cold_0=freq == 0, tail_1_5=(freq >= 1) & (freq <= 5),
                  mid_6_50=(freq >= 6) & (freq <= 50), head_gt50=freq > 50)
    arms = [(f"{base}_with_joint", base) for base in BASES] + [
        ("dcn_only_with_joint", "fusion_plus_dcn_with_joint"), ("dcn_only_with_joint", "fusion_plus_dcn"),
    ]
    comparisons = {f"{a}_vs_{b}": compare(values, a, b, groups) for a, b in arms}
    summary = {name: means([run[name] for run in runs.values()]) for name in next(iter(values.values()))}
    write_new(root / "confirmation.json", dict(users=len(rows), summary=summary, comparisons=comparisons, runs=runs,
              protocol_sha256=digest(root / "protocol.json"),
              development_sha256=digest(root / "development.json"), rows_sha256=digest(rows_path),
              checks=dict(baselines_match_archived_target_ranks=True, candidate_coverage_identical=True),
              timing_scope="One user, 500 on-device candidates. 16 warmups + 80 calls per arm, synchronized, no feature construction/search/sort.",
              test_evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "confirm"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/sequence_base_ablation_v1")
    parser.add_argument("--dcn-ablation", default="artifacts/channel_prior_ablation_v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = read("configs/sequence_rank.json")
    base = read(config["base_config"])
    parent, root, anchor_root = Path(base["output"]), Path(args.output), Path(args.dcn_ablation)
    anchor_manifest = anchor_root / "development.json"
    anchor_dev = read(anchor_manifest)
    assert anchor_dev["seeds"] == config["seeds"]
    for file, sha in anchor_dev["inputs_sha256"].items():
        assert digest(file) == sha
    spec = {name: anchor_dev["selection"][name]["best"] for name in BASES}
    protocol = dict(rule=RULE, sequence_config=config, dcn_epochs={k: s["epoch"] for k, s in spec.items()},
                    dcn_dev_sha256=digest(anchor_manifest), torch=str(torch.__version__),
                    sources={str(path): digest(path) for path in (Path(__file__).relative_to(Path.cwd()),
                        Path("scripts/run_sequence_rank.py"), Path("taac_retrieval/sequence_ranker.py"))})
    data = PreparedData(read(base["ranking_config"])["cache"])
    if args.stage == "fit":
        root.mkdir(parents=True, exist_ok=False)
        write_new(root / "protocol.json", {**protocol, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
        fit(config, parent, data, spec, root, device)
    else:
        if (root / "confirmation.json").exists():
            raise FileExistsError(root / "confirmation.json")
        saved = read(root / "protocol.json")
        assert all(saved[k] == v for k, v in protocol.items())
        confirm(config, parent, data, spec, root, anchor_root, device)


if __name__ == "__main__":
    main()
