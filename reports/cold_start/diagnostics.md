# 冷启动与长尾：固定召回模型的候选覆盖诊断

本次只使用**验证集**，固定原有点击 InfoNCE-64 的三个 checkpoint，不训练参数、不调索引、不查看测试目标。目标是回答：冷启动和长尾广告只是排不到前十，还是连前 500 都进不去？

## 核心结果

下表为 seed 2026、2027、2028 的平均 Recall。三次评估使用相同的 10,000 名冻结验证用户、同一目标广告和 2,708,122 个本地候选广告。

| 目标广告分组 | 验证用户数 | Recall@10 | Recall@100 | Recall@500 |
|---|---:|---:|---:|---:|
| 全体 | 10,000 | 2.9733% | 11.9467% | 23.0600% |
| 冷启动：0 次 | 659 | 0.1012% | 1.4669% | 8.3966% |
| 长尾：1–5 次 | 1,326 | 0.0000% | 0.0503% | 0.7290% |
| 中频：6–50 次 | 2,966 | 0.1911% | 2.2140% | 6.8555% |
| 头部：大于 50 次 | 5,049 | 5.7635% | 22.1562% | 40.3578% |

**长尾首先存在明显的候选覆盖问题。** 即使扩大到前 500，平均仍有约 99.27% 的长尾目标未进入候选。只在这 500 个候选上接一个排序模型，无法找回没有入池的目标。

冷启动比长尾的 Top-500 覆盖稍高，但仍有约 91.60% 的目标在池外；中频组也只有约 6.86% 的目标入池。后续值得优先检验内容表示、内容召回通道与候选融合，而不是只依据总体 Top-10 结果判断排序阶段的价值。

这不等于“排序没有价值”：全体用户约 23.06% 的目标已经进入前 500，但只有约 2.97% 进入前十。候选内部存在进一步排序的空间；不过，这个空间不能替代对池外冷启动、长尾目标的召回改善。Recall@500 也不是接上排序后必然达到的 Top-10 指标。

## 三个种子的覆盖

| 种子 | 全体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 | 中频 Recall@500 | 头部 Recall@500 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 23.1200% | 8.4977% | 0.5279% | 6.5408% | 40.7011% |
| 2027 | 23.2100% | 8.0425% | 0.9050% | 7.7883% | 40.1070% |
| 2028 | 22.8500% | 8.6495% | 0.7541% | 6.2374% | 40.2654% |

完整逐种子值和 NDCG@10/100/500 均保存在[汇总 JSON](../../artifacts/cold_start/diagnostics/click_summary.json)，本表不根据测试分数挑选种子。

## 指标与分桶口径

- 每名用户只有一个留出相关点击，所以当前定义下 Recall@K 与 HR@K 数值相同。NDCG@K 则在命中后额外考虑目标名次。
- 广告频次为所有用户**训练前缀**内的事件计数，包含曝光、点击和未知行为，不是点击次数；同一广告在某个用户历史中没出现，不代表它是全局冷启动。
- 分桶依据**目标广告**的训练频次，不是用户自己的事件数。验证组的 659/1326/2966/5049 人与旧测试报告的 785/1413/2984/4818 人不同，因为它们是不同的目标时点与广告。
- 目录固定为本地缓存广告目录，所有验证目标均在目录内；先屏蔽该用户可见历史，再做完整目录原始内积精确 Top-500。没有使用 ANN 近似结果。
- 每个 checkpoint 使用原来按验证 NDCG@10 选定的 epoch；本次没有按 Recall@500 重新挑 checkpoint。

还保存了 `conditional_Recall@10_given_Top500`：只看已经进入前 500 的目标，它们有多少进入前十。这是候选内部的描述统计，不是独立排序模型的效果。

## 排名与精确参照核对

每个用户的 `target_rank` 只记录前 500 内的真实名次；0 表示“目标位于前 500 之外”，不表示真实名次为 0，也不把未知名次伪记为 501。

三种子新结果的逐用户 Top-10 命中与 NDCG 和原验证文件**完全相同**。Top-500 与原 Top-10 的候选 ID 在极少数数值平分附近可能不同：三种子的集合重合率分别为 99.999%、99.994%、99.996%。本次 query batch 为 128，旧评估为 256，TopK 预算也不同，因此不声称候选 ID 逐位相同。融合对照应统一使用本次保存的候选结果，避免混用不同平分处理。

## 文件与复现

代码入口：[coverage.py](../../taac_retrieval/coverage.py)，纯统计接口是 `coverage_summary`；测试见 [test_coverage.py](../../tests/test_coverage.py)。五项测试覆盖已知名次与频次边界、空分组、无效 ID/重复候选、禁止静默缩小 K、分块精确检索以及拒绝覆盖已有输出。

原模型与缓存使用按功能命名的真实目录：

- checkpoint：`artifacts/objective_study/confirmation/selected_seed{seed}/best.pt`。
- 数据缓存：[scale_data/metadata.json](../../artifacts/objective_study/scale_data/metadata.json)。
- 冻结用户：[confirm_eval_indices.npy](../../artifacts/objective_study/confirm_eval_indices.npy)，实际加载时以 checkpoint 保存的队列为准。
- 原阶段的选择与结果：[objective_study 报告](../objective_study/README.md)。

三个诊断结果分别为：

- [seed 2026 JSON](../../artifacts/cold_start/diagnostics/click_seed2026.json) / [NPZ](../../artifacts/cold_start/diagnostics/click_seed2026.npz)。
- [seed 2027 JSON](../../artifacts/cold_start/diagnostics/click_seed2027.json) / [NPZ](../../artifacts/cold_start/diagnostics/click_seed2027.npz)。
- [seed 2028 JSON](../../artifacts/cold_start/diagnostics/click_seed2028.json) / [NPZ](../../artifacts/cold_start/diagnostics/click_seed2028.npz)。

NPZ 保存前 500 个本地广告 ID、内积分数、用户向量、可见历史、用户行号、目标广告、目标训练频次和逐用户各 K 指标，便于后续复用同一批用户做多路候选融合。JSON 保存 checkpoint SHA、缓存签名、评估队列 SHA、候选口径与代码摘要。

使用新输出前缀复现一个种子：

```bash
.project_venv/bin/python -m taac_retrieval.coverage \
  --checkpoint artifacts/objective_study/confirmation/selected_seed2026/best.pt \
  --cache artifacts/objective_study/scale_data \
  --output artifacts/cold_start/diagnostics_reproduction/click_seed2026 \
  --device cuda:6 --ks 10 100 500
```

结果文件已存在时程序拒绝覆盖。这里的 K 不会自动缩小；若使用微型测试目录，应显式传入较小的 K。
