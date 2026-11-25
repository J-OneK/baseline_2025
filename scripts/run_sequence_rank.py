"""Compare history-aware rankers on frozen Top-500 lists."""
import argparse
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from finish_ranking_study import paired
from group_recall_study import arrays
from prepare_channel_rank import selected_rows
from run_channel_rank import evaluate, infer, load_model, means, ranks_for, snapshot
from taac_retrieval.channel_features import recent_context
from taac_retrieval.channel_ranker import list_loss
from taac_retrieval.data import PreparedData
from taac_retrieval.ranking_data import read, write_new
from taac_retrieval.ranking_model import digest
from taac_retrieval.sequence_ranker import HistoryRanker

VARIANTS = ("history_attention", "joint_transformer")
CODE = ("scripts/run_sequence_rank.py", "taac_retrieval/sequence_ranker.py",
        "scripts/build_channel_features.py", "scripts/prepare_channel_rank.py")
RULE = (
    "Same two seeds, parent fit/dev lists, frozen DCN anchor, same candidate data, list loss and "
    "epochs. Per arm lock epoch and residual scale using development mean NDCG@10, then HR@10, "
    "then earlier epoch and lower scale. Prefer configurations preserving dev HR@10 and cold+tail "
    "hit count. Recommend a challenger only if it also improves dev NDCG. Confirm both locked arms "
    "once on 10000 users outside parent fit/dev/fresh and known older evaluation cohorts."
)


def register(config, base, data):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    parent = Path(base["output"])
    cohort = read(parent / "cohorts.json")
    previous = Path("reports/ranking_study/confirmation.json")
    sources = read(previous)["cohort_identity"]["exclusion_sources"]
    excluded = [data.user_rids[data.eval_indices],
                data.user_rids[np.load("artifacts/ranking_confirmation_v1/cohort.npy")]]
    for source in sources:
        assert digest(source["indices"]) == source["indices_sha256"]
        assert digest(source["user_rids"]) == source["user_rids_sha256"]
        excluded.append(np.load(source["user_rids"])[np.load(source["indices"])])
    parent_cohorts = {}
    for split in ("fit", "dev", "fresh"):
        file = parent / split / "rows.npy"
        sha = digest(file)
        assert cohort["cohorts"][split]["rows_sha256"] == sha
        parent_cohorts[split] = sha
        excluded.append(data.user_rids[np.load(file)])
    blocked = np.unique(np.concatenate(excluded))
    available = np.flatnonzero(~np.isin(data.user_rids, blocked))
    selected = np.sort(np.random.default_rng(config["cohort_seed"]).choice(
        available, config["confirmation_users"], replace=False))
    assert not np.isin(data.user_rids[selected], blocked).any()
    (root / "fresh").mkdir()
    file = root / "fresh/rows.npy"
    np.save(file, selected)
    pipeline = dict(base, output=config["output"], confirmation_users=config["confirmation_users"],
                    cohort_seed=config["cohort_seed"])
    write_new(root / "candidate_config.json", pipeline)
    write_new(root / "cohorts.json", dict(config=pipeline, parent_cohorts=parent_cohorts,
        prior_manifest_sha256=digest(previous), available_outside_known_cohorts=int(len(available)),
        test_evaluated=False,
        cohorts={"fresh": {"users": len(selected), "rows_sha256": digest(file)}}))
    write_new(root / "plan.json", dict(config=config, selection_rule=RULE,
        parent_selection_sha256=digest(parent / "selection.json"),
        sources={path:digest(path) for path in CODE}))
    print(f"Reserved {len(selected)} fresh users from {len(available)} eligible; "
          "no fresh labels used.", flush=True)


def check_plan(config, parent):
    root = Path(config["output"])
    plan = read(root / "plan.json")
    assert plan["config"] == config and plan["selection_rule"] == RULE
    assert digest(parent / "selection.json") == plan["parent_selection_sha256"]
    for path, sha in plan["sources"].items():
        assert digest(path) == sha, path


def anchor_model(parent, seed, device):
    chosen = read(parent / "selection.json")["selected"]
    assert chosen["variant"] == "channel_dcn_v2"
    checkpoint = chosen["checkpoints"][str(seed)]
    assert digest(checkpoint["checkpoint"]) == checkpoint["sha256"]
    return load_model(checkpoint["checkpoint"], device).eval(), chosen["scale"]


class CandidateLists:
    def __init__(self, root, split, seed, data, lookups, window, device):
        folder = root / split
        self.lists = arrays(folder / f"lists_seed{seed}.npz")
        rows = self.lists["rows"]
        np.testing.assert_array_equal(self.lists["targets"], data.items[rows, data.val_pos[rows]])
        self.x = np.load(folder / f"x_seed{seed}.npy")
        assert self.x.shape == (len(rows), 500, 15)
        self.features = torch.as_tensor(self.x, device=device)
        self.items = torch.as_tensor(self.lists["candidates"], device=device)
        history, actions, ages = recent_context(data, rows, window)
        self.history = lookups[torch.as_tensor(history, device=device)]
        self.actions = torch.as_tensor(actions, device=device, dtype=torch.long)
        self.ages = torch.as_tensor(ages, device=device)
        self.lookup = lookups

    def __len__(self):
        return len(self.lists["rows"])

    def inputs(self, index):
        return (self.features[index], self.lookup[self.items[index]], self.history[index],
                self.actions[index], self.ages[index])


def content_lookup(data, device):
    vector = torch.as_tensor(np.load(data.root / "item_mm81.npy"), device=device)
    return F.normalize(vector, dim=-1)


@torch.inference_mode()
def updates(model, lists, step, keep_history=True):
    model.eval()
    scores = []
    for start in range(0, len(lists), step):
        inputs = lists.inputs(slice(start, start + step))
        keep = None if keep_history else torch.zeros_like(inputs[3], dtype=torch.bool)
        scores.append(model(*inputs, keep=keep).cpu().numpy())
    return np.concatenate(scores)


def make_model(config, variant, anchor, dim, device):
    return HistoryRanker(variant, config["width"], config["heads"], config["history_length"],
                         anchor.mean, anchor.std, dim).to(device)


def fit(config, base, data, seed, device):
    root, parent = Path(config["output"]), Path(base["output"])
    out = root / f"development_seed{seed}.json"
    if out.exists() or (root / "selection.json").exists():
        raise FileExistsError(out)
    check_plan(config, parent)
    lookup = content_lookup(data, device)
    args = (seed, data, lookup, config["history_length"], device)
    train = CandidateLists(parent, "fit", *args)
    dev = CandidateLists(parent, "dev", *args)
    for split, lists in (("fit", train), ("dev", dev)):
        declared = np.load(parent / split / "rows.npy")
        if split == "dev":
            np.testing.assert_array_equal(lists.lists["rows"], declared)
        else:
            assert np.isin(lists.lists["rows"], declared).all()
    assert not np.intersect1d(train.lists["rows"], dev.lists["rows"]).size
    anchor, scale = anchor_model(parent, seed, device)
    batch_size = config["batch_size"]
    train_base, dev_base = [
        lists.lists["prior"] + scale * infer(anchor, lists.x, device, batch_size)
        for lists in (train, dev)]
    match = train.lists["candidates"] == train.lists["targets"][:, None]
    assert (match.sum(1) == 1).all()
    target = torch.as_tensor(match.argmax(1), device=device)
    weights = torch.as_tensor(np.where(data.train_counts[train.lists["targets"]] <= 5,
                                      config["rare_weight"], 1), device=device, dtype=torch.float32)
    prior = torch.as_tensor(train_base, device=device)
    baseline = ranks_for(dev.lists, dev_base)
    reference = evaluate(dev.lists, baseline, data.train_counts)[0]
    runs = []
    for variant in VARIANTS:
        torch.manual_seed(seed)
        model = make_model(config, variant, anchor, lookup.shape[1], device)
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        shuffle = torch.Generator(device=device).manual_seed(seed)
        folder = root / f"{variant}_seed{seed}"
        folder.mkdir(exist_ok=True)
        params = sum(parameter.numel() for parameter in model.parameters())
        for epoch in range(1, config["epochs"] + 1):
            started = time.perf_counter()
            model.train()
            losses = []
            for batch in torch.randperm(len(train), generator=shuffle, device=device).split(batch_size):
                opt.zero_grad(set_to_none=True)
                residual = model(*train.inputs(batch))
                loss = list_loss(prior[batch], residual, target[batch], weights[batch],
                                 config["residual_penalty"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(loss.detach())
            loss = float(torch.stack(losses).mean().cpu())
            assert np.isfinite(loss)
            file = folder / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), file)
            checkpoint = dict(path=str(file), sha256=digest(file))
            correction = updates(model, dev, batch_size)
            assert np.isfinite(correction).all()
            for residual_scale in config["residual_scales"]:
                prediction = ranks_for(dev.lists, dev_base + residual_scale * correction)
                metric, _, changes = snapshot(dev.lists, baseline, prediction, data.train_counts)
                runs.append(dict(variant=variant, epoch=epoch, scale=residual_scale, loss=loss,
                                 params=params, checkpoint=checkpoint, metrics=metric, changes=changes))
            current = metric["all"]
            print(f"{seed} {variant} ep{epoch} loss {loss:.5f} HR@10 {current['Recall@10']:.4%} "
                  f"NDCG {current['NDCG@10']:.6f} seconds {time.perf_counter() - started:.2f}", flush=True)
    write_new(out, dict(seed=seed, train_lists=len(train), dev_users=len(dev), anchor=reference, runs=runs))


def lock(config, base):
    root = Path(config["output"])
    check_plan(config, Path(base["output"]))
    reports = [read(root / f"development_seed{seed}.json") for seed in config["seeds"]]
    anchor = means([report["anchor"] for report in reports])
    candidates = []
    for same in zip(*(report["runs"] for report in reports), strict=True):
        key = {p:same[0][p] for p in ("variant", "epoch", "scale", "params")}
        assert all(all(run[p] == value for p,value in key.items()) for run in same)
        aggregate = means([run["metrics"] for run in same])
        guarded = aggregate["hr10"] >= anchor["hr10"] and aggregate["rare_top10_hits"] >= anchor["rare_top10_hits"]
        candidates.append(dict(**key, **aggregate, guarded=guarded,
            new_channel_top10=sum(run["changes"]["new_channel_top10"] for run in same),
            checkpoints={str(seed):run["checkpoint"] for seed,run in zip(config["seeds"], same)}))
    order = lambda arm: (arm["guarded"], arm["ndcg10"], arm["hr10"], -arm["epoch"], -arm["scale"])
    arms = {v:max((arm for arm in candidates if arm["variant"] == v), key=order) for v in VARIANTS}
    best = max(arms.values(), key=order)
    diagnostic = best["variant"] if best["guarded"] and best["ndcg10"] > anchor["ndcg10"] else "dcn_v2"
    write_new(root / "selection.json", dict(rule=RULE, anchor=anchor, development_choice=diagnostic,
        arms=arms, all_configurations=candidates, plan_sha256=digest(root / "plan.json"),
        fresh_rows_sha256=read(root / "cohorts.json")["cohorts"]["fresh"]["rows_sha256"]))
    print(f"Development choice {diagnostic}; arms {arms}", flush=True)


def load_history_ranker(config, arm, seed, anchor, dim, device):
    ckpt = arm["checkpoints"][str(seed)]
    assert digest(ckpt["path"]) == ckpt["sha256"]
    model = make_model(config, arm["variant"], anchor, dim, device)
    model.load_state_dict(torch.load(ckpt["path"], map_location=device, weights_only=True))
    return model.eval()


def timing(anchor, scale, models, values, device):
    """Warm one-user model scoring on 500 already materialized candidates."""
    synchronize = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
    records = [values.inputs(slice(i, i + 1)) for i in range(96)]
    timings = {"dcn_v2": [], **{key:[] for key in models}}
    jobs = {"dcn_v2": None, **models}
    for index, inputs in enumerate(records):
        for name, model in jobs.items():
            synchronize()
            started = time.perf_counter()
            scores = scale * anchor(inputs[0])
            if model is not None:
                scores = scores + model(*inputs)
            synchronize()
            if index >= 16:
                timings[name].append(1000 * (time.perf_counter() - started))
    return {name:{f"p{p}_ms":float(np.percentile(ms, p)) for p in (50, 95)}
            for name,ms in timings.items()}


@torch.inference_mode()
def confirm(config, base, data, device):
    parent, root = Path(base["output"]), Path(config["output"])
    out = root / "confirmation.json"
    if out.exists():
        raise FileExistsError(out)
    check_plan(config, parent)
    selection = read(root / "selection.json")
    assert digest(root / "plan.json") == selection["plan_sha256"]
    rows = selected_rows(read(root / "candidate_config.json"), "fresh")
    lookup = content_lookup(data, device)
    reports, values = {}, {}
    for seed in config["seeds"]:
        anchor, scale = anchor_model(parent, seed, device)
        args = (seed, data, lookup, config["history_length"], device)
        fresh = CandidateLists(root, "fresh", *args)
        np.testing.assert_array_equal(fresh.lists["rows"], rows)
        dev = CandidateLists(parent, "dev", *args)
        current_prior, dev_prior = [
            lists.lists["prior"] + scale * infer(anchor, lists.x, device, config["batch_size"])
            for lists in (fresh, dev)]
        baseline = ranks_for(fresh.lists, current_prior)
        metrics, local = evaluate(fresh.lists, baseline, data.train_counts)
        reports[seed] = {"dcn_v2": dict(metrics=metrics)}
        values[seed] = {"dcn_v2": local}
        models = {}
        for name, arm in selection["arms"].items():
            model = load_history_ranker(config, arm, seed, anchor, lookup.shape[1], device)
            models[name] = model
            correction = updates(model, fresh, config["batch_size"])
            prediction = ranks_for(fresh.lists, current_prior + arm["scale"] * correction)
            metric, local, changes = snapshot(fresh.lists, baseline, prediction, data.train_counts)
            masked_scores = updates(model, dev, config["batch_size"], keep_history=False)
            masked_prediction = ranks_for(dev.lists, dev_prior + arm["scale"] * masked_scores)
            masked = evaluate(dev.lists, masked_prediction, data.train_counts)[0]
            reports[seed][name] = dict(metrics=metric, changes=changes, dev_no_history=masked["all"])
            values[seed][name] = local
        reports[seed]["timing"] = timing(anchor, scale, models, fresh, device)
        np.savez(root / "fresh" / f"target_ranks_seed{seed}.npz", rows=rows,
                 frequency=values[seed]["dcn_v2"]["target_train_frequency"],
                 **{name:result["target_rank"] for name,result in values[seed].items()})
        print(seed, {k:means([r["metrics"]]) for k,r in reports[seed].items() if k != "timing"}, flush=True)
    variants = ("dcn_v2", *VARIANTS)
    summary = {k:means([r[k]["metrics"] for r in reports.values()]) for k in variants}
    frequency = values[config["seeds"][0]]["dcn_v2"]["target_train_frequency"]
    buckets = dict(all=frequency >= 0, cold_0=frequency == 0, tail_1_5=(frequency >= 1) & (frequency <= 5))
    comparisons = {}
    for challenger, control in [(v, "dcn_v2") for v in VARIANTS] + [(VARIANTS[1], VARIANTS[0])]:
        comparisons[f"{challenger}_vs_{control}"] = {
            name:{metric:paired(np.stack([
                v[challenger][metric][mask].astype(float) - v[control][metric][mask] for v in values.values()
            ])) for metric in ("hits_at_10", "ndcg_at_10")}
            for name,mask in buckets.items()}
    environment = dict(device=device, device_name=torch.cuda.get_device_name(device)
                       if str(device).startswith("cuda") else platform.processor(),
                       torch=str(torch.__version__), torch_threads=torch.get_num_threads(),
                       timing_scope="80 warmed per-user model calls x 500, synchronized, GPU shared if CUDA")
    write_new(out, dict(users=len(rows), selection_sha256=digest(root / "selection.json"),
        summary=summary, comparisons=comparisons, runs=reports, environment=environment, test_evaluated=False))
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/sequence_rank.json")
    p.add_argument("--stage", choices=("register", "fit", "lock", "confirm"), required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    config = read(args.config)
    base = read(config["base_config"])
    if args.stage == "lock":
        lock(config, base)
        return
    data = PreparedData(read(base["ranking_config"])["cache"])
    if args.stage == "register":
        register(config, base, data)
    elif args.stage == "fit":
        if args.seed not in config["seeds"]:
            p.error("Supply a configured seed for fit.")
        fit(config, base, data, args.seed, args.device)
    else:
        confirm(config, base, data, args.device)


if __name__ == "__main__":
    main()
