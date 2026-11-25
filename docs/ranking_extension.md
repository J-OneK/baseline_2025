# 主分支完整推荐与联合排序

```text
可见历史 → InfoNCE 内积 Top-500
        → 102/122 同组近邻 + 81 号向量补冷启动、长尾
        → 去重和等量替换，固定 500 条
        → 原融合分 + DCN V2 残差
        → 逐条历史和候选特征联合注意力，加入第二个残差
        → Top-10 OID
```

每桶最多补 5 条。合并保留原融合 Top-100 的成员和已有冷尾广告。DCN V2 在 15 项汇总特征上重新打分，联合 Transformer 用 40 条已知行为的内容、动作、位置和时间提供第二次调整。验证点击仅用于指标，`--show-target` 在打分后打印目标名次。 [真实用户数据链路](data_flow.md)

## 为什么采用联合排序

原来的高余弦硬提位会对近乎所有用户触发，HR@10 下滑，促使我们让新增候选按分数同主候选竞争。第一轮先在真实列表上训练 15 特征的 DCN V2。第二轮让排序模型逐条读取点击、曝光历史，保留同一个候选池和基础分。联合模型为每个广告构造两个特征 token，与用户历史交互，适合检验持续兴趣是否比单次最大相似更有用。 [提位失败与 DCN V2 选型](../reports/channel_rank/README.md#遇到的问题与选择依据) · [联合实验设计](../reports/sequence_rank/README.md)

| 两种子平均，10,000 人留出确认 | HR@10 | NDCG@10 |
|---|---:|---:|
| DCN V2 | 2.660% | 0.013950 |
| 加候选历史注意力 | 2.725% | 0.014211 |
| 加联合 Transformer | **2.825%** | **0.014800** |

NDCG@10 相对提高约 6.1%，差值的 95% 配对区间为 +0.000069～+0.001694。HR@10 的差值区间为 −0.020～+0.360 个百分点。共享 RTX 2080 Ti 上，已准备的 500 条候选的**模型打分** p95 从约 2.42 ms 到约 5.18 ms。更早的 DCN V2 相对原融合 HR@10 为 2.233% → 2.917%，来自另一批 6,000 人。 [对照与计时口径](../reports/sequence_rank/README.md#10000-人确认集) · [两轮评估边界](release.md)

## 本机演示

从仓库根目录，在安装项目依赖的 Python 环境中执行：

```bash
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 311428 --show-target --trace
```

这个用户的实际留出目标 OID 是 20004114747。它在原 500 条以外，由辅助通道补入。旧融合分排第 494，DCN V2 排第 11，联合排序排第 8。省略 `--rid` 后，默认选择 10,000 人确认队列的首位用户。指定 `--cohort dev --index 0` 可以选原 5,000 人开发队列。用 `--variant history_attention` 可运行对照模型。常规输出给出 OID、名次与辅助通道标记，`--trace` 另外给出各阶段样式和分数。

本机环境可以用 `.project_venv/bin/python` 替代 `python`。权重、目录和向量需在本地生成或准备。主要目录是 `artifacts/baseline/data`、`artifacts/ranking_v1`、`artifacts/ranking_content_v1`、`artifacts/channel_rank_v1` 与 `artifacts/sequence_rank_v1`。 [训练与复现步骤](../reports/sequence_rank/README.md#本机复现)

| 入口或源码 | 职责 |
|---|---|
| [recommend_sequence_rank.py](../scripts/recommend_sequence_rank.py) | 双路召回、联合排序和真实 trace 的主入口 |
| [recommend_channel_rank.py](../scripts/recommend_channel_rank.py) | 共享数据准备、打分与 trace，也可单独复查 DCN V2 |
| [group_recall.py](../taac_retrieval/group_recall.py) | 同组查询、内容相似度、固定候选预算 |
| [build_channel_features.py](../scripts/build_channel_features.py) | 冻结内容融合分、候选合并、15 项特征 |
| [channel_ranker.py](../taac_retrieval/channel_ranker.py) | DCN V2 交叉层、旧排序基础分与列表损失 |
| [sequence_ranker.py](../taac_retrieval/sequence_ranker.py) | 历史 token、候选 token、联合注意力与残差 |
| [run_sequence_rank.py](../scripts/run_sequence_rank.py) | 训练、锁定开发配置与 10,000 人确认 |

`run_main.py` 提供原召回和 Faiss 的独立复现。两种入口的计时范围见[版本说明](release.md)。
