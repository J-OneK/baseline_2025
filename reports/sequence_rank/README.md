# 候选感知历史排序与 OneTrans 风格联合建模

现有 DCN V2 用最近历史算出最大余弦、匹配动作、时间差等 15 个数值特征。本轮把点击与曝光逐条交给排序器，检验候选能否根据自身内容读取不同的历史，以及把行为和特征放在同一注意力里是否有额外收益。实验沿用两路召回、固定的 500 条合并候选和留出的下一点击目标。

## 模型与对照

```text
同一份 Top-500 和旧融合分
  ├─ 已锁定的 DCN V2 → 基准排序分
  ├─ 独立编码历史 → 候选查询历史 → 学习分数修正
  └─ 两种候选 token 与历史联合注意力 → 学习分数修正
```

两个新模型都保留原 DCN V2 分数，并将新增网络的末层初始化为零。候选输入相同，包括原 15 个数值特征和单位长度的 81 号内容向量。历史取原验证边界前最后 40 条已知点击或曝光，每条包含 81 号内容向量、动作、位置和相对最后可见事件的时间差。每请求的历史只编码一次，候选打分可以有不同的注意力权重。原 15 项输入保持原训练统计的归一化。

| 方案 | 候选如何使用历史 | 新增可训练参数 |
|---|---|---:|
| DCN V2 | 使用原历史汇总特征 | 0 |
| DCN V2 + 候选历史注意力 | 一层因果编码历史，每个候选用自己的查询做交叉注意力 | 12,097 |
| DCN V2 + 联合 Transformer | 每候选有广告内容与基础分 token、通道与匹配 token。两层联合注意力使 token 同时读取历史和同候选中允许读取的 token | 27,385 |

联合模型借鉴 OneTrans 的行为前缀、混合参数和候选读取共享前缀。两组候选 token 分别有自己的 Q/K/V 与 FFN，历史用共享参数。第一层候选读取未更新的历史投影，第二层读取更新后的因果历史。该实现按请求复用历史 K/V，候选之间互不读取，可以在相同候选列表下逐项打分。测试覆盖全空历史、padding 隔离、候选重排等变和历史因果掩码。 [模型](../../taac_retrieval/sequence_ranker.py) · [测试](../../tests/test_sequence_ranker.py) · [OneTrans §3](https://arxiv.org/html/2510.26104v3#S3)

## 训练与冻结

- 沿用原通道排序的有效训练列表，两个种子分别为 3,943 和 4,270 份。每份有一个出现在 500 条中的留出点击。仍用列表交叉熵、冷尾正例权重 2、残差平方惩罚 0.01。
- 两种新模型都用宽度 24、3 个注意力头、batch 32、AdamW 学习率 0.001、5 轮。把旧 DCN V2 分数作为已冻结的基线。两个新模型有不同参数量，因此它们的差异衡量两套实现的效果。
- 在之前的 5,000 人开发集上，每个新模型选择轮次和残差系数。在达到基准 HR@10 与冷尾命中数的组合里，优先按 NDCG@10 选，系数候选是 0.25、0.5、1。选定两组配置后才为新确认用户准备候选。
- 确认独立抽取 10,000 名用户，排除通道排序训练来源、开发集、此前的 6,000 人确认集以及已登记的历史评估名单。按 RID 取交集并核验。留出的验证点击用于当前评估，用户更早的历史仍可参加原召回和内容 MLP 训练。项目原测试点击另有口径。

[配置](../../configs/sequence_rank.json) · [训练、锁定与确认](../../scripts/run_sequence_rank.py)

## 开发集结果

| 两种子平均，5,000 人 | HR@10 | NDCG@10 | 冷尾命中次数合计 | 残差系数 |
|---|---:|---:|---:|---:|
| DCN V2 | 2.610% | 0.013311 | 5 | — |
| + 候选历史注意力 | 2.680% | 0.013769 | 5 | 0.25 |
| + 联合 Transformer | 2.770% | 0.014166 | 7 | 1.00 |

两种新模型均锁定第 5 轮。联合模型在开发集优先。候选历史注意力和联合模型的候选均使用相同的额外 81 号向量，所以相对 DCN V2 的结果衡量新增输入和模型一起使用的效果。两种新增模型之间的对照使用相同的输入。屏蔽历史的诊断用于观察模型对历史的敏感程度。

## 10,000 人确认集

| 两种子平均 | HR@10 | NDCG@10 | 冷启动命中次数合计 | 长尾命中次数合计 |
|---|---:|---:|---:|---:|
| DCN V2 | 2.660% | 0.013950 | 2 | 4 |
| + 候选历史注意力 | 2.725% | 0.014211 | 3 | 4 |
| + 联合 Transformer | **2.825%** | **0.014800** | **6** | **5** |

- 联合模型的 HR@10 差为 **+0.165 个百分点**，95% 配对区间 **−0.020～+0.360 个百分点**。NDCG@10 差为 **+0.000850**，区间 **+0.000069～+0.001694**。两个种子各自的 HR@10 变化为 2.350% → 2.580% 和 2.970% → 3.070%。
- 轻量历史注意力相对 DCN V2 的 HR@10 差为 +0.065 个百分点、NDCG 差为 +0.000261，两项区间都跨零。联合模型相对历史注意力也方向向上，两项区间仍跨零。
- 所有方案使用完全相同的 500 条候选。总体 Recall@500 都为 16.565%。联合模型两个种子共净增 33 次 Top-10 命中，其中头部 24 次、中频 4 次、冷启动 4 次、长尾 1 次。冷尾单桶的区间下界为 0，新增样本集中在 seed 2026。
- 联合模型在 40 次辅助通道独家找到目标的记录中命中前十 6 次，DCN V2 为 5 次。实际用户 RID 311428 的训练未见广告由辅助通道补入，DCN V2 排第 11，联合模型排第 8。其余这批新增冷尾命中来自原主召回候选。

开发集屏蔽直接供新模型注意力读取的逐条历史，让旧 15 项特征保持原值。联合模型的两种子平均 HR@10 由 2.770% 到 2.480%。这说明冻结模型对直接历史输入敏感。屏蔽会改变训练时的输入分布，而且相对 DCN V2 同时增加了候选内容输入与参数量，收益分解需要同输入和同参数量的训练对照。按最近 40 条中点击次数做事后分组，联合模型对 0 次、1–4 次、至少 5 次点击的组分别净增 2、14、17 次命中，分别约 +0.125、+0.127、+0.231 个百分点。记录用于提出假设，各组需要配对验证。 [结果明细](confirmation.json) · [分组、样例与重放核验](audit.json)

### 计算代价

同一张共享 RTX 2080 Ti 上，对两种子各预热 16 次后单用户逐次测 80 次、每次 500 条候选，只计模型打分并在前后进行 CUDA 同步。范围是两个种子的 p95，特征已经准备在设备上。

| 模型 | 打分 p95 |
|---|---:|
| DCN V2 | 2.42 ms |
| + 候选历史注意力 | 2.57–3.23 ms |
| + 联合 Transformer | 5.18–5.19 ms |

这是新实验的模型打分计时。之前的 Faiss 搜索、历史过滤和运行入口另有测量口径。完整候选提取、特征构建到出结果的计时需要单独测量。

联合模型已作为主分支完整推荐的最终打分阶段。它在同一批候选上改善了 NDCG，HR 的增量和冷尾收益仍有较大不确定性。主分支同时提供轻量配置，并分别保存打分计时。 [主分支入口](../../docs/ranking_extension.md) · [数据链路](../../docs/data_flow.md)

## 本机复现

已有权重和目录时可重算单用户链路：

```bash
python scripts/recommend_sequence_rank.py --device cpu --seed 2026 --rid 311428 --show-target --trace
```

[生成的真实 trace](demo_rid311428.json) 与[逐阶段解释](../../docs/data_flow.md)记录了该条广告从补入候选到第 8 名的过程。

从仓库根目录运行，使用项目 Python 环境和已有的 `artifacts/baseline`、`artifacts/channel_rank_v1`、`artifacts/ranking_v1`、`artifacts/ranking_content_v1`、`artifacts/group_recall_v1`、`artifacts/ranking_confirmation_v1` 本地资产。登记使用已核对的历史排除队列。现有完整实验产物位于 `artifacts/sequence_rank_v1`。重跑时复制配置到新文件，并给 `output` 指定新目录。

```bash
PY=/home/jyk/project/baseline_2025/.project_venv/bin/python
CONF=configs/sequence_rank_replay.json
OUT=artifacts/sequence_rank_replay
cp configs/sequence_rank.json "$CONF"
"$PY" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()); d["output"] = sys.argv[2]; p.write_text(json.dumps(d, indent=2) + "\n")' "$CONF" "$OUT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

"$PY" scripts/run_sequence_rank.py --config "$CONF" --stage register
for seed in 2026 2027; do
  "$PY" scripts/run_sequence_rank.py --config "$CONF" --stage fit --seed "$seed" --device cuda:5
done
"$PY" scripts/run_sequence_rank.py --config "$CONF" --stage lock
PIPE="$OUT/candidate_config.json"
"$PY" scripts/prepare_channel_rank.py --config "$PIPE" --stage proposals --split fresh --workers 8
for seed in 2026 2027; do
  "$PY" scripts/prepare_channel_rank.py --config "$PIPE" --stage retrieval --split fresh --seed "$seed" --device cuda:5
  "$PY" scripts/build_channel_features.py --config "$PIPE" --split fresh --seed "$seed" --device cuda:5
done
"$PY" scripts/run_sequence_rank.py --config "$CONF" --stage confirm --device cuda:5
```

实际运行核验了用户集合不重叠、500 候选去重、历史过滤与候选成员一致，并用 CPU 重放两个种子共 133 份确认列表的目标名次。仓库 57 项测试通过。 [预登记](plan.json) · [开发配置锁定](development.json)
