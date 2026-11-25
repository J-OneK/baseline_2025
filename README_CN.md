# 腾讯广告序列召回与列表排序

[English](README.md) · [数据链路实录](docs/data_flow.md) · [完整代码导读](docs/code_walkthrough.md) · [版本与指标口径](docs/release.md)

基于 TAAC 2025 提供的 SASRec 风格基线和公开 TencentGR-1M 数据，实现从原始行为处理、序列召回，到固定候选的联合排序与本地演示。召回留有单独的全目录评估和 Faiss 索引复现。

主召回使用 **1 层因果 Transformer、32 维表示、仅点击监督、InfoNCE-64**。完整推荐在同组辅助召回和 DCN V2 基础分上叠加 **OneTrans 风格联合 Transformer 残差**，让每条候选直接读取逐条历史。

```text
公开 Parquet → 清洗与逐用户时间留出 → 数组缓存
  ├─ 点击 InfoNCE-64 → 查询向量 → 主召回 Top-500
  └─ 102/122 同组历史 + 81 内容向量 → 冷尾提议
                         ↓
合并为 500 条 → 15 项候选特征 → DCN V2 基础分
                         ↓
40 条历史 + 每个候选的两个 token → 联合 Transformer 残差
                         ↓
总分排序 → Top-10 广告 OID
```

## 完整推荐结果

同一份 Top-500 候选、10,000 人留出确认集、两个种子平均。DCN V2 加入联合排序后，**NDCG@10 由 0.013950 到 0.014800，相对提升约 6.1%**，差值的 95% 配对区间为 +0.000069～+0.001694。HR@10 从 **2.660% 到 2.825%**，HR 差值区间跨零。独立的 6,000 人确认集用于比较更早的内容融合与 DCN V2。 [排序对照与计时](reports/sequence_rank/README.md) · [版本与队列](docs/release.md)

本地已有冻结权重和向量时，运行完整推荐并查看中间数据：

```bash
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 311428 --show-target --trace
```

该用户的冷启动点击由辅助通道补回，DCN V2 排第 11，联合排序排第 8。原始 Parquet、候选和特征的形状及分数见[数据链路实录](docs/data_flow.md)。

## 主召回与索引的历史实验

历史确认实验包含 280,985 名有效用户、2,708,122 个固定候选广告，验证和测试各 10,000 名用户，种子为 2026、2027、2028。

| 模型 | 测试 HR@10 | 测试 NDCG@10 |
|---|---:|---:|
| 按训练频次的热门召回 | 0.7600% | 0.004230 |
| 原目标 BCE 对照 | 2.1133% | 0.010528 |
| 选定的点击 InfoNCE-64 | 2.9733% | 0.015391 |

以上为**三个种子的均值**。平均 NDCG@10 相对提高 46.19%，HR@10 相对提高 40.69%。开发时只用验证集选方案，训练时也只用验证集选 epoch。原目标 BCE 到选定方案同时改变了监督位置和负样本/损失设置，不能把总收益全部归因于“仅点击监督”。控制变量和配对区间见[原始实验报告](reports/objective_study/README.md)。

索引展示使用单独固定的 **seed=2026** 模型。HNSW 经 MIPS→L2 变换后，与精确 Top-10 的 ID 重合率为 98.81%。512 个测试查询上的单线程预热 CPU **索引检索与历史过滤 p95** 从 40.20 ms 降到 4.14 ms，约快 9.70 倍。该数字不含编码器、网络、进程启动及加载，不能写成端到端推荐接口延迟。

## 安装与召回索引复现

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

当前项目代码位于 `taac_retrieval/`。根目录的 `model.py`、`dataset.py`、`main.py`、`infer.py` 等保留原始基线来源；当前模型复用其中的注意力与前馈模块。[上游中文说明](docs/upstream_baseline_zh.md)和[英文说明](docs/upstream_baseline_en.md)保留作参考。主分支也保留列表排序消融与分桶报告。生成式召回等方法的对照见[研究分支](docs/branches.md)。

## 使用边界

这是公开静态数据上的逐用户时间留出实验，不是官方私有测试成绩或线上 A/B。排序确认与三种子召回测试使用不同缓存和用户队列。冷启动、长尾的 Top-10 仍然较弱。现成的 **81 号 32 维向量**同时用于广告编码和辅助候选匹配。简历及展示引用指标前，请先阅读[版本说明](docs/release.md)。
