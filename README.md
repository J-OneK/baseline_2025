# Tencent Ads Retrieval and Listwise Ranking

[中文说明](README_CN.md) · [Data flow and recorded trace](docs/data_flow.md) · [Code walkthrough](docs/code_walkthrough.md) · [Release scope](docs/release.md)

A recommendation retrieval project built on the TAAC 2025 SASRec-style baseline and the public TencentGR-1M data. The main branch contains a complete path from Parquet preparation to exact evaluation, vector export, persisted Faiss indexes, and a local Top-10 recommendation demo.

The retriever uses one causal Transformer block, 32-dimensional representations, click-only supervision, and sampled InfoNCE with 64 negatives. The full path adds grouped content candidates for cold and tail ads. It reranks the fixed 500 candidates with a channel-aware DCN V2 base score and a OneTrans-style residual that reads individual history events.

```text
Public Parquet → per-user chronological holdouts → arrays
  ├─ click InfoNCE-64 → exact Top-500
  └─ historical attribute and content matches → cold/tail proposals
                       ↓
Merge 500 candidates → 15 features → DCN V2 base score
                       ↓
Behavior tokens + two tokens per candidate → joint Transformer residual
                       ↓
Sort combined scores → Top-10 ad OIDs
```

## Confirmed ranking results

On the same Top-500 lists from 10,000 held-out validation users, the joint ranker increased **NDCG@10 from 0.013950 to 0.014800** across two seeds. The paired 95% interval for the difference was +0.000069 to +0.001694. **HR@10 increased from 2.660% to 2.825%** with an interval spanning zero. The earlier DCN V2 comparison used a separate 6,000-user cohort. [Ranking protocol and results](reports/sequence_rank/README.md)

With the ranking artifacts available, run the full path and inspect a real example:

```bash
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 311428 --show-target --trace
```

Grouped content recall recovered the cold ad. The DCN V2 base placed it 11th and the joint ranker moved it to 8th. [Raw data, tensor shapes, and score trace](docs/data_flow.md)

## Historical retrieval and index results

Historical local confirmation used 280,985 eligible users, 2,708,122 fixed candidate ads, 10,000 evaluation users per split, and three seeds (2026–2028).

| Model | Test HR@10 | Test NDCG@10 |
|---|---:|---:|
| Training-frequency popularity | 0.7600% | 0.004230 |
| Original-objective BCE control | 2.1133% | 0.010528 |
| Selected click InfoNCE-64 | 2.9733% | 0.015391 |

These are three-seed means, not the score of the single deployed checkpoint. Model selection used validation only. The BCE-to-InfoNCE comparison changes both supervision and sampled loss settings; individual ablations are retained in the [original study report](reports/objective_study/README.md).

The separate index benchmark used the selected **seed 2026** checkpoint. MIPS-to-L2 HNSW achieved 98.81% overlap with exact Top-10 results. Warmed single-thread CPU **index search plus history filtering** p95 was 40.20 ms for FlatIP and 4.14 ms for HNSW on 512 test queries. This excludes encoding, startup, loading and network traffic. See [measurement boundaries](docs/release.md).

## Reproduce the retrieval and Faiss pipeline

Run from the repository root. The project requirements pin the recorded environment; choose a PyTorch build appropriate for your CUDA/CPU hardware if necessary.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-project.txt
python scripts/run_main.py --help
python scripts/run_main.py --stage all --dry-run
```

Place the already downloaded dataset at `TencentGR-1M/`, or pass `--data-dir /path/to/TencentGR-1M`. Data, checkpoints and generated indexes are local artifacts and are not distributed by Git. The Python workflow does not need the legacy C++ Faiss demo to be built.

```bash
python scripts/run_main.py --stage prepare --data-dir TencentGR-1M
python scripts/run_main.py --stage train --device cuda:0
python scripts/run_main.py --stage evaluate --device cuda:0
python scripts/run_main.py --stage export --device cuda:0
python scripts/run_main.py --stage index
python scripts/run_main.py --stage recommend --split val --index 0 --device cuda:0
```

The default configuration is [configs/main.json](configs/main.json); outputs go to `artifacts/showcase/`. Use the same `--root` value for every stage when choosing another output directory. `--stage all` runs the full pipeline, including test evaluation. Preparation scans the public snapshot and creates both the historical reference cache and the larger cache needed to reproduce the cohort exclusion rule. Full training/evaluation is a workstation workload; `--device cpu` is supported but may be slow. The older `requirements.txt` plus `requirements-project-extra.txt` remain available for an existing legacy environment.

Training selects a checkpoint with validation and skips test labels. The explicit evaluation stage evaluates the frozen checkpoint. Recommendation defaults to validation user 0 in the fixed cohort; `--index` is a cohort position, not an arbitrary public user ID. The output contains ad OIDs, raw inner-product scores, and local timing components. The content channel and listwise ranker have a separate [offline entry point](docs/ranking_extension.md).

## Read the project

- [Documentation entry](docs/index.md): commands and repository map.
- [Code walkthrough](docs/code_walkthrough.md): dataset schema, splits, tensors, model, loss, evaluation and retrieval.
- [Release and evidence](docs/release.md): historical metrics, artifact identity and remaining limitations.
- [Branches](docs/branches.md): supervision, cold-start and depth experiments.

`taac_retrieval/` contains the current pipeline. Root-level `model.py`, `dataset.py`, `main.py`, `infer.py` and related files retain the supplied baseline lineage; the current model reuses its attention and feed-forward blocks. The [upstream English documentation](docs/upstream_baseline_en.md) and [Chinese documentation](docs/upstream_baseline_zh.md) remain available. The original objective study and its reproduction runner remain on main because they support the selected model. Generative and other exploratory comparisons live on research branches.

## Scope

This is an offline, per-user holdout study using a public static catalog, not an official private-test result or an online experiment. The ranking confirmation and retrieval test use different cohorts and catalog sizes. Cold and tail Top-10 performance remains weak. The project uses the supplied 32-dimensional embedding family **81**, not a newly trained text encoder. See the [release notes](docs/release.md) before quoting metrics.
