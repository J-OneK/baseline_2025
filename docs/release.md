# 主分支版本与实验依据

本次冻结将主分支固定为一层点击 InfoNCE-64 的完整召回链路，并将后续探索保留到独立分支。冻结主要整理代码入口、默认配置和文档；历史报告没有被改写成新版本重新训练所得的结果。

## 固定的主方案

| 项目 | 设置 |
|---|---|
| 编码器 | 一层因果 Transformer，hidden=32，heads=1，dropout=0.2 |
| 广告输入 | 训练可见广告 ID、13 个离散属性、81 号 32 维向量 |
| 用户输入 | 用户 ID/静态属性组成的 token，加按时间排列的历史广告 |
| 主模型的动作输入 | 不额外加历史动作 embedding；事件仍参与历史上下文 |
| 监督与负样本 | 仅点击目标，64 个均匀采样的训练可见广告，排除该用户训练前缀 |
| 损失 | InfoNCE，temperature=1.0 |
| 优化 | Adam，学习率 0.001，batch=128，5 轮；按验证 NDCG@10、再 HR@10 选 epoch |
| 展示模型 | 固定 seed=2026；完整研究确认种子为 2026、2027、2028 |
| 索引 | 原始内积；HNSW 使用 MIPS→L2 变换，M=32，efConstruction=100，efSearch=2048 |

配置见 [main.json](../configs/main.json)。历史确认 seed=2026 的最优 epoch 为 4，测试 HR@10 为 2.8400%、NDCG@10 为 0.014665。这与 README 中的三种子均值不同。

## 历史证据

[原始报告目录](../reports/objective_study/README.md)保留当时生成的内容，包含开发阶段选型、六次确认训练、分桶、环境与索引测量。

| 记录 | 用途 |
|---|---|
| [model_selection.json](../reports/objective_study/model_selection.json) | 只根据开发验证指标锁定点击 InfoNCE-64 |
| [metrics.json](../reports/objective_study/metrics.json) | 三种子测试、配对 bootstrap、频次与内容分桶 |
| [scale_data_audit.json](../reports/objective_study/scale_data_audit.json) | 逐用户截断、目标未进入历史、训练频次与 OOV 检查 |
| [ann_initial_failure.json](../reports/objective_study/ann_initial_failure.json) | 原始内积 HNSW 未达到验证门槛的配置 |
| [ann_amendment.json](../reports/objective_study/ann_amendment.json) | 根据验证查询调整索引几何的记录 |
| [environment.json](../reports/objective_study/environment.json) | 原测量环境 |
| [code_manifest.json](../reports/objective_study/code_manifest.json) | 原实验源码摘要，非本次整理后的摘要 |
| [verification.json](../reports/objective_study/verification.json) | 原实验完成时的检查结果，非本次发布的测试计数 |

历史文件包含旧目录名、绝对路径和生成时的源码摘要，应当作为原始记录阅读。当前命令以 `scripts/run_main.py` 和语义化 runner 为准。完整旧路径与探索代码保存在 `research/full-study`，分支范围见[分支说明](branches.md)。

## 本次冻结的验证

主分支整理后已运行 36 项 CPU 单元测试，全部通过；从现有主模型 checkpoint 加载后，使用已有完整 Faiss 索引为验证队列第 0 个用户成功返回十个广告。此检查验证整理后的代码可以加载并使用原产物，没有重新训练模型或重跑完整三种子确认实验。历史报告中的测试计数仍保留其生成时的值。

## 指标怎么解释

准确率评估使用完整固定候选库，过滤已见广告，每个用户只有一个目标点击，因此 HR@10 等于这个协议下的 Recall@10。目标是用户截断序列中最后两个首次出现的广告点击；训练仅使用验证目标时间之前的事件。测试历史可以包含较早的验证点击。

候选库取自抽样公开序列中所有出现过的广告，独立于点击标签和模型固定。它包含训练时未见的广告；广告 ID 与广告属性词表只在训练可见广告上拟合，未见 ID 使用 OOV 表示。公共静态特征与完整广告目录属于已知快照条件，不能解释为跨所有用户的全局时间隔离。

三种子测试 NDCG@10 为 0.010528→0.015391，相对增加 46.19%；种子与用户配对 bootstrap 的绝对差 95% 区间为 [0.002961, 0.006804]。原始 BCE 与选定方案的监督位置和负样本数不同，非等 FLOPs 对照。原始实验还保留 click BCE、click BCE-64、加权 BCE、动作输入等对照，支持更具体地分析训练目标。

ANN Recall@10 是近似检索与**同一模型精确检索**的 ID 重合率，不是真实点击召回率。历史 ANN 测量只用固定 seed=2026 的 512 个测试查询，不应与全体 10,000 用户的三种子模型均值混用。

索引延迟是单线程、预热后的 CPU 搜索与 Python 历史过滤 p95；不含编码、传输、进程启动及加载。8 线程批量 QPS 是另一个负载。索引文件大小不是进程峰值内存。共享工作站测量也不是生产服务 SLA。

## 如何展示已有产物

`run_main.py` 采用新的 `artifacts/showcase/` 布局，供新运行使用；它不会假装已有历史产物是新训练输出。若本机已有原研究产物，可直接演示历史主模型，无须重训：

```bash
python -m taac_retrieval.recommend \
  --checkpoint artifacts/objective_study/confirmation/selected_seed2026/best.pt \
  --cache artifacts/objective_study/scale_data \
  --export-dir artifacts/objective_study/export \
  --index-dir artifacts/objective_study/ann \
  --device cuda:0 --split val --index 0
```

以上路径是本地文件位置，不是仓库内分发的大文件；使用前应核对已有目录。推荐代码会检查 checkpoint SHA 与导出/索引身份，避免悄悄组合不同模型的产物。演示默认使用验证用户；单次预热请求的耗时输出不等于重新测得的 p95。

新运行使用 README 中的逐阶段命令。`run_baseline.py` 与 `run_objective_study.py` 用于完整对照研究，参数请看各自 `--help`；`run_main.py` 只生成一个固定主方案，不生成三种子均值，也不再次开展选型。

## 已知局限与可陈述范围

冷启动组历史 HR@10 仅 0.0425%，长尾 1–5 次组为 0%；总体收益主要集中在头部广告。计数是全部用户训练前缀中的广告事件数，包含曝光、点击和未知事件，不是点击次数。通用 `coverage.py` 可以在验证集上检查 Top-10/100/500，后续冷尾策略结果保留在研究分支。

本版本没有独立排序模型、常驻推荐 API 或线上流量验证；没有复现语义 ID 生成，也没有重新训练文本编码器。适合描述为“公开广告数据上的序列召回与向量检索项目”，可以引用本地协议下的对照结果，不应称为比赛获奖、官方成绩或线上 CTR/CVR 收益。
