# 主分支完整推荐与联合排序

```text
可见历史 → InfoNCE 主召回 Top-500 + 102/122 同组近邻提议
        → 去重和等量替换，固定 500 条
        → 15 项特征 → 纯 DCN V2 基础分
        → 逐条历史和候选特征联合注意力 → OneTrans 风格残差
        → 按 DCN 分 + 联合残差排序 → Top-10 OID
```

每个冷尾桶最多补 5 条。原融合分仍用于保护旧 Top-100 的**候选成员**，也保留在 trace 中作参照。召回分和内容 MLP 分仍各自作为输入特征；只去掉了**最终排序分上固定的 `0.9 / 0.1` 加法**。两种排序器给同一份 500 条候选打分，中途没有按 DCN 名次截断。

## 方案怎么定的

旧融合把大部分新补入的目标留在榜尾；把高余弦候选硬塞进前十也降低了总体命中。于是先用真实候选列表训练 DCN V2，学习候选来源、历史相似度、曝光或点击、时间和组大小的组合。然后让候选直接读最近 40 条已知行为，检验逐条历史能否补充 DCN 的汇总信息。 [早期失败与特征交叉](../reports/channel_rank/README.md#遇到的问题与选择依据)

最终在同一批 10,000 人留出确认队列、同样的 500 条候选和两个种子上比较：

| 排序方法 | HR@10 | NDCG@10 |
|---|---:|---:|
| 原融合 + DCN V2 | 2.715% | 0.014622 |
| 原融合 + DCN V2 + 联合排序 | 2.890% | 0.015648 |
| 纯 DCN V2 | 2.725% | 0.014490 |
| **纯 DCN V2 + 联合排序（默认）** | **2.890%** | **0.015674** |

在纯 DCN 上重训联合模型后，NDCG 相对提高约 **8.2%**，差值的 95% 配对区间为 **+0.000387～+0.001932**；HR 差值区间包含零。两个完整组合的差值区间也包含零，因此默认采用更直接的纯 DCN 基础分。整体增量主要来自头部。RTX 2080 Ti 上，已准备好的 500 条候选的模型打分 p95 为 **0.34 ms → 2.86 ms**。 [四组消融与计时范围](../reports/sequence_base_ablation/README.md) · [版本口径](release.md)

## 本机演示与复现

```bash
# 历史辅助通道补入冷启动广告，纯 DCN 和最终排序均为第 3 名
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 311428 --show-target --trace

# 主召回已经找到了目标，联合排序把它从第 14 提到第 8
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 957879 --show-target --trace

# 只用纯 DCN 打分
python scripts/recommend_channel_rank.py --device cpu --seed 2026 --rid 957879 --show-target
```

`--show-target` 在打分后打印留出点击名次，`--trace` 打印中间数组和打分。省略 `--rid`，默认使用 10,000 人确认队列的首位用户；`--cohort dev --index 0` 选择开发队列。`--legacy` 复查原融合加 DCN 的旧流程；`--variant history_attention` 复查基于旧融合的轻量对照。 [真实数据链路](data_flow.md)

默认版本由 [ranking_release.json](../configs/ranking_release.json) 固定两种子的 DCN 第 11 轮和联合模型第 5 轮（残差系数 1）。推荐入口读取本机 `artifacts/baseline/data`、`artifacts/ranking_v1`、`artifacts/ranking_content_v1`、`artifacts/channel_rank_v1`、`artifacts/sequence_rank_v1`、`artifacts/channel_prior_ablation_v1` 和 `artifacts/sequence_base_ablation_v1` 中的相关资产。权重、目录、向量和候选缓存不随 Git 分发。需重训时，先准备[原排序候选](../reports/sequence_rank/README.md#本机复现)，然后依次跑[纯 DCN 消融](../reports/channel_prior_ablation/README.md#复跑和核验)和[联合排序消融](../reports/sequence_base_ablation/README.md#复跑与核验)；若换成本地新权重，用 `--release-config` 指向记录了对应路径与 SHA 的配置。

| 入口或源码 | 职责 |
|---|---|
| [recommend_sequence_rank.py](../scripts/recommend_sequence_rank.py) | 完整双路召回、联合排序和 trace |
| [recommend_channel_rank.py](../scripts/recommend_channel_rank.py) | 共享的候选与打分链路，也可单独运行纯 DCN |
| [group_recall.py](../taac_retrieval/group_recall.py) | 同组查找、内容相似度和固定候选预算 |
| [build_channel_features.py](../scripts/build_channel_features.py) | 旧分保护、候选合并、15 项特征 |
| [channel_ranker.py](../taac_retrieval/channel_ranker.py) | DCN V2 交叉层、直接打分与列表损失 |
| [sequence_ranker.py](../taac_retrieval/sequence_ranker.py) | 历史 token、候选 token、联合注意力与残差 |
| [ablate_sequence_rank.py](../scripts/ablate_sequence_rank.py) | 在两种基础分上分别训练、锁定与确认 |

`run_main.py` 保留主召回和 Faiss 的独立复现。
