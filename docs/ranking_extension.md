# 主分支排序链路

```text
同一可见历史 → InfoNCE 内积 Top-500
             → 102/122 同组近邻＋81 号向量补冷启动与长尾
             → 合并为 500 条
             → 0.9 z(内积)＋0.1 z(内容 MLP logit)＋DCN V2 残差 → Top-10
```

每组最多补 5 条。合并时保护原融合 Top-100 和已有的冷启动、长尾广告。最终排序可以重排全部 500 条。DCN V2 用 15 个数值特征，包括候选来源、与历史点击及曝光的相似度、时间差和匹配组大小。训练时每个样本是一份已召回目标的真实候选列表，留出的点击在列表内做交叉熵监督。

| 6,000 人确认集，两种子均值 | HR@10 | NDCG@10 |
|---|---:|---:|
| 召回＋冻结内容 MLP | 2.233% | 0.01225 |
| 加历史查组候选，沿用原排序分 | 2.233% | 0.01225 |
| 加历史查组与 DCN V2 | **2.917%** | **0.01602** |

HR@10 绝对提升 0.683 个百分点，95% 配对区间为 +0.383～+0.992 个百分点。两位用户的长尾目标由辅助通道补进候选后进入前十。冷启动收益主要在候选覆盖；固定预算下整体 Recall@500 从 16.975% 到 16.867%。整体排名提升主要来自头部样本。[完整消融、分桶与复现协议](../reports/channel_rank/README.md)

## 本机演示

从仓库根目录执行。使用已经准备的排序缓存、冻结召回和排序权重，现场重算候选与排序：

```bash
python scripts/recommend_channel_rank.py --device cpu --seed 2026 --rid 134009 --show-target
```

这名用户的目标 OID 为 20004004871。原召回漏掉它，补入后冻结融合分排名第 492，DCN V2 排名第 6。省略 `--show-target` 时只打印推荐列表及辅助来源。省略 `--rid` 时，默认用确认队列第一个用户，也可指定 `--cohort dev --index 0`。运行时使用全目录精确 Top-500。

| 源码 | 职责 |
|---|---|
| [group_recall.py](../taac_retrieval/group_recall.py) | 历史同组检索、相似度与固定候选预算 |
| [build_channel_features.py](../scripts/build_channel_features.py) | 冻结分数融合、合并、特征组装 |
| [channel_features.py](../taac_retrieval/channel_features.py) | 点击与曝光匹配、时间差等特征 |
| [channel_ranker.py](../taac_retrieval/channel_ranker.py) | 低秩交叉层与列表损失 |
| [run_channel_rank.py](../scripts/run_channel_rank.py) | 训练、开发集锁定、独立确认 |
| [recommend_channel_rank.py](../scripts/recommend_channel_rank.py) | 重新跑单用户完整推荐并展示目标路径 |

从 Git 获取的是源码、配置与可审读的结果。数据、权重和中间数组需按[复现步骤](../reports/channel_rank/README.md#本机复现)准备。`run_main.py` 是原始召回与 Faiss 的独立入口，其历史测试与延迟采用[另一套评估口径](release.md)。
