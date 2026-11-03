# Tencent Ads Sequential Retrieval

[中文项目说明](README_CN.md)

A reproducible retrieval project built on TAAC2025 / TencentGR-1M: chronological holdout, vectorized feature preparation, a causal Transformer, controlled objective studies, exact full-catalog evaluation, and persisted Faiss retrieval. Cold-start and long-tail coverage are the next research focus; an independent ranking stage is added only when candidate coverage supports it.

- [Project overview and commands](PROJECT_OVERVIEW.md)
- [Complete Chinese code walkthrough](PROJECT_FULL_WALKTHROUGH.md)
- [Documentation index](docs/index.md) and [directory layout](docs/directory_layout.md)
- [Baseline guide](BASELINE_GUIDE.md)
- [Objective and ANN study](reports/objective_study/README.md)
- [InfoNCE supervision ablation](reports/infonce_supervision/README.md)
- [Cold-start and long-tail study](reports/cold_start/README.md)
- [Transformer depth study](reports/depth_study/README.md) and [matched-initialization protocol](docs/depth_study.md)

The active modules are in `taac_retrieval/`. Configurations and entry points use semantic names such as `configs/baseline.json`, `configs/objective_study.json`, `scripts/run_baseline.py`, and `scripts/run_objective_study.py`. Artifacts and reports use matching `baseline/`, `objective_study/`, `infonce_supervision/`, `cold_start/`, and `depth_study/` directories.

The depth study compares 1, 2, and 4 Transformer layers while fixing width, heads, FFN, loss, and sampling. Shared initial parameters are checked individually; all selection uses validation data, and deeper models do not automatically replace the existing model or index.

Historical date-based paths remain symlink aliases so existing artifacts stay readable. Frozen plans and source snapshots retain their original provenance; use a fresh output directory when reproducing an experiment with current code.

The original open-source baseline documentation is preserved in [English](docs/upstream_baseline_en.md) and [Chinese](docs/upstream_baseline_zh.md). This project reports a local offline task, not official leaderboard or online business results.
