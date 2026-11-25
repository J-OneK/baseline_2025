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
from prepare_channel_rank import selected_rows
from run_sequence_rank import load_history_ranker, check_plan
from taac_retrieval.channel_features import FEATURES, recent_context
from run_channel_rank import infer, load_model as load_ranker, ranks_for
from taac_retrieval.data import SequenceDataset
from taac_retrieval.evaluate_checkpoint import load_model
from taac_retrieval.metrics import exact_topk, to_device
from taac_retrieval.ranking_data import read
from taac_retrieval.ranking_model import digest


@torch.inference_mode()
def recommend(args, sequence_config=None):
    config = read(args.config if sequence_config is None else sequence_config["base_config"])
    base, group = read(config["ranking_config"]), read(config["group_config"])
    root = Path(config["output"])
    lock = read(root / "selection.json")
    chosen = lock["selected"]
    seed = args.seed
    sequence_selection = None
    if sequence_config is not None:
        sequence_root = Path(sequence_config["output"])
        check_plan(sequence_config, root)
        sequence_selection = read(sequence_root / "selection.json")
        if digest(sequence_root / "plan.json") != sequence_selection["plan_sha256"]:
            raise ValueError("Sequence selection does not match the frozen plan.")
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
        if sequence_selection is not None and args.cohort == "fresh":
            rows = selected_rows(read(sequence_root / "candidate_config.json"), "fresh")
        else:
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
    delta_dcn = chosen["scale"] * infer(model, x, args.device, batch_size=1)
    base_scores = fusion + delta_dcn
    lists = {"candidates": merged}
    fixed, dcn = (ranks_for(lists, values)[0] for values in (fusion, base_scores))
    orders = {"original": original[0], "augmented_prior": fixed, "dcn_v2": dcn}
    final = dcn
    if sequence_selection is not None:
        arm = sequence_selection["arms"][args.variant]
        inputs, context = sequence_inputs(sequence_config, data, rows, merged, x, args.device)
        sequence = load_history_ranker(sequence_config, arm, seed, model, inputs[1].shape[-1], args.device)
        delta_sequence = arm["scale"] * sequence(*inputs).cpu().numpy()
        final_scores = base_scores + delta_sequence
        final = ranks_for(lists, final_scores)[0]
        orders[args.variant] = final
    addition = set(merged[changed].tolist())
    result = dict(seed=seed, user_rid=int(data.user_rids[row]), budget=len(final),
                  auxiliary_added=len(addition), top10=[
                      dict(rank=rank, oid=int(data.catalog_oids[item]), from_auxiliary=int(item) in addition)
                      for rank, item in enumerate(final[:10], 1)
                  ])
    if sequence_selection is not None:
        result["ranker"] = args.variant
    if args.show_target:
        target = sample["target"]
        ranks = {}
        for label, order in orders.items():
            ranks[label] = rank_of(order, target)
        result["target_audit"] = dict(oid=int(data.catalog_oids[target]), ranks=ranks,
                                     from_auxiliary=target in addition)
    if sequence_selection is not None and args.trace:
        focus = int(sample["target"]) if args.show_target else int(final[0])
        result["data_trace"] = data_trace(
            data, row, sample, recall, proposals, merged, changed, inputs, context, focus,
            fusion, delta_dcn, delta_sequence, final_scores, sequence, orders)
    return result


def sequence_inputs(config, data, rows, merged, features, device):
    items, actions, ages = recent_context(data, rows, config["history_length"])
    content = np.load(data.root / "item_mm81.npy", mmap_mode="r")

    def vectors(ids):
        return torch.nn.functional.normalize(torch.as_tensor(content[ids], device=device), dim=-1)

    inputs = (torch.as_tensor(features, device=device), vectors(merged), vectors(items),
              torch.as_tensor(actions, dtype=torch.long, device=device), torch.as_tensor(ages, device=device))
    return inputs, (items, actions, ages)


def rank_of(order, item):
    positions = np.flatnonzero(order == item)
    return int(positions[0]) + 1 if len(positions) else None


def data_trace(data, row, sample, recall, proposals, merged, changed, inputs, context, focus,
               fusion, delta_dcn, delta_sequence, final_scores, model, orders):
    def identifier(item):
        item = int(item)
        return dict(local_id=item, source_rid=int(data.catalog_rids[item]), oid=int(data.catalog_oids[item]))

    def first_values(value, count=4):
        return [round(float(item), 6) for item in value[:count]]

    items, actions, ages = context
    indices = np.flatnonzero(items[0])[-4:]
    tail = [dict(**identifier(items[0, i]), action=int(actions[0, i]), log_age_minutes=float(ages[0, i]))
            for i in indices]
    cut = int(data.val_cut[row])
    arrays = dict(
        recall_input_ids=[1, *sample["items"].shape],
        query_vector=list(recall["queries"].shape),
        main_candidates=list(recall["candidates"].shape),
        auxiliary_proposals=list(proposals.shape),
        merged_candidates=list(merged.shape),
        static_features=list(inputs[0].shape),
        candidate_content=list(inputs[1].shape),
        history_content=list(inputs[2].shape),
        history_actions=list(inputs[3].shape),
        history_ages=list(inputs[4].shape),
        history_tokens=[*inputs[3].shape, model.history_content.out_features],
        ranking_scores=list(final_scores.shape),
    )
    if model.variant == "joint_transformer":
        arrays["candidate_tokens"] = [*merged.shape, 2, model.item.out_features]
    trace = dict(
        cache=dict(user_row=row, model_user_id=int(sample["user"]), max_history=int(data.max_history),
                   cached_events=int(data.lengths[row]), validation_boundary=cut,
                   catalog_items=data.num_items),
        shapes=arrays,
        sequence_dataset_tail=[
            dict(**identifier(item), action=int(action), timestamp=int(timestamp))
            for item, action, timestamp in zip(data.items[row, :cut][-3:], data.actions[row, :cut][-3:],
                                               data.timestamps[row, :cut][-3:])
        ],
        sequence_rank_history=dict(valid_events=int((items[0] > 0).sum()), tail=tail),
        query_head=first_values(recall["queries"][0]),
        main_top3=[dict(**identifier(item), inner_product=float(score))
                   for item, score in zip(recall["candidates"][0, :3], recall["scores"][0, :3])],
        auxiliary_added_preview=[identifier(item) for item in merged[changed][:3]],
        focus=dict(**identifier(focus), train_events=int(data.train_counts[focus]),
                   ranks=dict(raw_recall=rank_of(recall["candidates"][0], focus),
                              **{name:rank_of(order, focus) for name, order in orders.items()})),
    )
    columns = np.flatnonzero(merged[0] == focus)
    if len(columns):
        i = int(columns[0])
        trace["focus"].update(
            candidate_column=i, from_auxiliary=bool(changed[0, i]),
            static_features={name:float(value) for name, value in zip(FEATURES, inputs[0][0, i].tolist())},
            content_unit_head=first_values(inputs[1][0, i].tolist()),
            fixed_fusion=float(fusion[0, i]), dcn_residual=float(delta_dcn[0, i]),
            dcn_score=float(fusion[0, i] + delta_dcn[0, i]),
            sequence_residual=float(delta_sequence[0, i]), final_score=float(final_scores[0, i]),
        )
    return trace


def main(use_sequence=False):
    description = "Run the locked sequence ranker on two-channel Top-500 candidates." if use_sequence else __doc__
    p = argparse.ArgumentParser(description=description)
    if not use_sequence:
        p.add_argument("--config", default="configs/channel_rank.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, choices=(2026, 2027), default=2026)
    p.add_argument("--cohort", choices=("dev", "fresh"), default="fresh")
    user = p.add_mutually_exclusive_group()
    user.add_argument("--index", type=int, default=0)
    user.add_argument("--rid", type=int)
    p.add_argument("--show-target", action="store_true")
    if use_sequence:
        p.add_argument("--sequence-config", default="configs/sequence_rank.json")
        p.add_argument("--variant", choices=("joint_transformer", "history_attention"),
                       default="joint_transformer")
        p.add_argument("--trace", action="store_true", help="Show intermediate shapes and sample data.")
    args = p.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(4)
    sequence_config = read(args.sequence_config) if use_sequence else None
    print(json.dumps(recommend(args, sequence_config), indent=2))


if __name__ == "__main__":
    main()
