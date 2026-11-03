# 腾讯广告序列召回与向量检索

[English](README.md) · [完整代码导读](docs/code_walkthrough.md) · [版本与指标口径](docs/release.md)

基于 TAAC 2025 提供的 SASRec 风格基线和公开 TencentGR-1M 数据，完成“Parquet 数据处理 → 序列模型训练 → 全候选精确评估 → 向量导出 → Faiss 持久化索引 → 推荐演示”的完整召回项目。

主分支固定为 **1 层因果 Transformer、32 维表示、仅点击监督、InfoNCE-64**。项目重点是数据与评估协议、训练目标对照，以及保持原始内积语义的检索工程。它没有复现整篇生成式推荐论文，也没有独立的精排模型。

```text
公开 Parquet → 清洗与逐用户时间留出 → 数组缓存
                         ↓
广告 ID + 属性 + 81 号向量 → 广告表示
用户 token + 因果历史序列 → 用户表示
                         ↓
       点击 InfoNCE-64 / 全候选精确 Top-10 评估
                         ↓
向量导出 → MIPS→L2 HNSW → 历史广告过滤 → Top-10 广告原始 ID
```

## 已有实验结果

历史确认实验包含 280,985 名有效用户、2,708,122 个固定候选广告，验证和测试各 10,000 名用户，种子为 2026、2027、2028。

| 模型 | 测试 HR@10 | 测试 NDCG@10 |
|---|---:|---:|
| 按训练频次的热门召回 | 0.7600% | 0.004230 |
| 原目标 BCE 对照 | 2.1133% | 0.010528 |
| 选定的点击 InfoNCE-64 | 2.9733% | 0.015391 |

以上为**三个种子的均值**。平均 NDCG@10 相对提高 46.19%，HR@10 相对提高 40.69%。开发时只用验证集选方案，训练时也只用验证集选 epoch。原目标 BCE 到选定方案同时改变了监督位置和负样本/损失设置，不能把总收益全部归因于“仅点击监督”。控制变量和配对区间见[原始实验报告](reports/objective_study/README.md)。

索引展示使用单独固定的 **seed=2026** 模型。HNSW 经 MIPS→L2 变换后，与精确 Top-10 的 ID 重合率为 98.81%。512 个测试查询上的单线程预热 CPU **索引检索与历史过滤 p95** 从 40.20 ms 降到 4.14 ms，约快 9.70 倍。该数字不含编码器、网络、进程启动及加载，不能写成端到端推荐接口延迟。

## 安装与运行

在仓库根目录运行。项目依赖文件固定已记录环境的版本；如本机 CUDA/CPU 不兼容，请相应选用合适的 PyTorch 构建。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-project.txt
python scripts/run_main.py --help
python scripts/run_main.py --stage all --dry-run
```

将已下载的数据放到 `TencentGR-1M/`，也可用 `--data-dir /实际数据路径`。Git 只保存代码、配置和小型报告；原始数据、权重、缓存和索引保留在本地。当前 Python 主流程无需编译旧版 C++ 检索示例。旧环境也可按原 `requirements.txt` 加 `requirements-project-extra.txt` 补充依赖。

```bash
python scripts/run_main.py --stage prepare --data-dir TencentGR-1M
python scripts/run_main.py --stage train --device cuda:0
python scripts/run_main.py --stage evaluate --device cuda:0
python scripts/run_main.py --stage export --device cuda:0
python scripts/run_main.py --stage index
python scripts/run_main.py --stage recommend --split val --index 0 --device cuda:0
```

默认配置为 [configs/main.json](configs/main.json)，默认输出目录为 `artifacts/showcase/`。如需其他目录，每个阶段使用相同的 `--root`。`--stage all` 会完成全部阶段，包含固定模型的测试评估。准备阶段会生成小规模参考缓存和较大训练缓存，用于重现“排除之前查看过测试的用户”的规则。完整实验需要工作站资源；可设置 `--device cpu`，耗时会增加。

训练阶段不读取测试标签，只按验证集选最优 checkpoint；显式 `evaluate` 阶段评估固定模型。推荐演示默认读取验证队列，`--index 0` 表示固定队列的第一个用户，不是任意用户 ID。返回十个广告原始 OID、内积分数及本地耗时。新的单种子运行不等于重新完成历史三种子研究。

## 阅读路线

- [文档入口](docs/index.md)：文件地图、运行与展示方式。
- [代码导读](docs/code_walkthrough.md)：原始数据、样本构造、模型、损失、评估和推荐流程。
- [版本与证据边界](docs/release.md)：主模型、历史结果、可复现性及局限。
- [分支说明](docs/branches.md)：监督方式、冷启动与模型深度的后续尝试。

当前项目代码位于 `taac_retrieval/`。根目录的 `model.py`、`dataset.py`、`main.py`、`infer.py` 等保留原始基线来源；当前模型复用其中的注意力与前馈模块。[上游中文说明](docs/upstream_baseline_zh.md)和[英文说明](docs/upstream_baseline_en.md)保留作参考。主分支保留支持主方案选择的原始目标消融，之后的探索放在 `research/*` 分支。

## 使用边界

这是公开静态数据上的逐用户时间留出实验，不是官方私有测试成绩或线上 A/B。冷启动、长尾的 Top-10 仍然较弱。项目只接入现成的 **81 号 32 维向量**，没有训练新的文本编码器或实现语义 ID 自回归生成。简历及展示引用指标前，请先阅读[版本说明](docs/release.md)。
