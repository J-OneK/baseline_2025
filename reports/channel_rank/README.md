# 在同一批候选上训练通道感知排序

## 问题与实验设计

原链路先用 InfoNCE 序列模型从完整目录里取 500 条，再把召回内积与内容 MLP 的点击得分融合。历史查组通道利用用户近期见过的广告，从共享 102 或 122 属性的组内补入冷启动和长尾候选。上一轮发现这些新增目标能进 Top-500，但几乎都待在榜尾。属性组生成器在相同预算下也没有获得更高的候选覆盖。

这次固定候选池，把问题改成「从系统实际能召回的 500 条里，学会把下一次点击往前排」。原内容 MLP 学习的是日志里的曝光和点击。新排序器每个训练样本拿到的是一整张实际候选列表，对齐了推断时的候选分布和 Top-K 排序任务。实验用同配额对照检验新增特征和交叉层。

```text
逐用户可见历史
  ├─ 冻结 InfoNCE → 全目录精确 Top-500
  └─ 历史属性查组 → 用 81 向量找相似冷启动、长尾广告
                         ↓
       固定 500 条，保护原排序 Top-100 和原有冷尾候选
                         ↓
        原召回和 MLP 融合分 + 新模型学到的残差分 → Top-10
```

- 冷启动是训练前缀全局出现 0 次，长尾是出现 1–5 次。计数包含曝光、点击和未知事件。
- 辅助通道每组最多补 5 条，去重，优先替换召回最低的未保护候选，总预算始终为 500。这里的 Top-100 保护在合并候选时生效。新模型对合并后的全部 500 条重新打分。
- 归一化参数来自原候选列表。设召回内积分与原 MLP 的点击 logit 分别为 `s` 和 `c`，原排名分为 `0.9z(s) + 0.1z(c)`。新增候选用相同用户原列表的均值与标准差做换算，原列表未被替换的广告得分不受新增候选影响。
- 仅在真实目标已被至少一个通道召回且合并后仍留在列表时，这名用户能贡献一个训练样本。目标身份只用于列表损失和之后的指标。500 条以外的点击也纳入开发及确认集分母。

[候选划分与生成](../../scripts/prepare_channel_rank.py) · [固定候选预算](../../taac_retrieval/group_recall.py) · [候选特征构建](../../scripts/build_channel_features.py)

## 模型与监督

| 对照 | 特征 | 得分方式 |
|---|---|---|
| 冻结 MLP | 原内容及属性输入 | 原召回内积与原内容 MLP 固定融合 |
| 列表训练 MLP | 召回和 MLP 分、训练事件数、早期点击数、81 向量存在性 | 固定融合分加 MLP 残差 |
| 通道特征 MLP | 基础 5 项与下述新增 10 项 | 固定融合分加 MLP 残差 |
| 通道特征 DCN V2 | 与通道特征 MLP 相同的 15 项 | 两层低秩交叉与 MLP 并行，拼接预测残差 |

十项新增特征是辅助通道补入标记、点击同组最大余弦及匹配标记、曝光同组最大余弦及匹配标记、最强同组历史是否点击、其相对最后一条可见事件的时间差、匹配字段数、最小匹配组大小、候选与近期历史的最大正余弦。历史窗口为最近 40 条可见的已知曝光、点击行为。同组比较的字段为 102、122。最强同组历史在选取时沿用 0.04 的点击加分。

DCN V2 每一层执行 `x[l+1] = x[l] + x[0] * (U(V(x[l])) + b)`，其中 `*` 为逐元素乘法，投影秩为 8。交叉层可以表达「某候选来自辅助通道，同时与近期点击高度相似」这类组合。深层分支用两层宽度为 32 的 GELU 网络，末层零初始化，使模型从原排序分开始训练。方差与均值来自训练列表。DeepFM 的因子分解项同样适用于显式交互，这轮已固定使用连续的召回和匹配信号，选择低秩 DCN V2 作为交互结构对照。 [DCN V2 原论文](https://arxiv.org/abs/2008.13535) · [DeepFM 原论文](https://arxiv.org/abs/1703.04247)

设某名用户的第 `p` 列为留出点击，`L = -log(exp(score[p]) / sum(exp(score)))`。分母包含这名用户的全部 500 个候选，其他候选作为竞争项。这个分数衡量同一候选列表内的相对顺序，日志曝光上的 CTR 仍由原 MLP 对照提供。训练把冷启动和长尾正例权重设为 2，并为残差添加权重 0.01 的均方惩罚。验证阶段用 `原融合分 + scale × 残差` 排序。

[特征实现](../../taac_retrieval/channel_features.py) · [模型和损失](../../taac_retrieval/channel_ranker.py) · [训练与锁定](../../scripts/run_channel_rank.py)

## 数据与冻结协议

沿用 TAAC2025/TencentGR-1M 静态广告目录和逐用户留出验证点击。召回全量对 1,538,730 个广告做精确内积检索并过滤历史。逐用户评估历史止于原协议的验证边界。辅助特征所需的广告内容和属性在静态目录中可查，频率来自训练前缀。用户历史已参与上游模型训练，本文验证的是留出点击的排序表现。

- 在排除之前用于主模型和排序验证的用户之后，从可用用户按固定种子抽取 24,000 人作为新排序训练来源，并预留互不重叠的 6,000 人做确认。
- 新排序器实际监督样本数为 3,943 与 4,270，辅助通道独家带入的正例为 57 与 56。
- 用上一轮历史查组使用过的 5,000 人做开发集。在两个种子上各训练 5 轮，AdamW 学习率为 0.001、batch size 为 128。只在开发集比较三个残差系数 0.25、0.5、1。
- 开发选择要求 NDCG@10 提高、HR@10 和冷启动加长尾的命中合计均不下降。在通过的组合中选 NDCG@10 最高者。模型、轮次、系数及确认用户文件哈希锁定后，才为 6,000 人生成候选并计算指标。
- 两个种子均报告结果，并对逐用户差值做配对 bootstrap。原协议测试点击未用于本实验。

## 开发集消融

下表均采用相同 500 条合并候选。各项为两种子平均，冷尾命中数是两种子合计。

| 5,000 人开发集 | HR@10 | NDCG@10 | 冷尾进前十 | 其中辅助通道独家召回 |
|---|---:|---:|---:|---:|
| 冻结 MLP | 1.840% | 0.009842 | 0 | 0 |
| 列表训练 MLP | 2.370% | 0.012407 | 0 | 0 |
| 通道特征 MLP | 2.430% | 0.012433 | 2 | 1 |
| 通道特征 DCN V2 | **2.610%** | **0.013311** | **5** | **4** |

三类新排序器的最优配置均为第 5 轮、残差系数 1。列表监督结合 5 个基础特征已改善开发集指标。这一组同时增加了学习目标和基础信号，结果衡量两者一起使用的收益。后两组的通道信息与交叉层各自进一步增加了开发集命中。这批对照用于选模，确认集只复核锁定后的完整方案。

## 6,000 人确认集

| 两种子平均 | HR@10 | NDCG@10 | 整体 Recall@500 | 冷启动 Recall@500 | 长尾 Recall@500 |
|---|---:|---:|---:|---:|---:|
| 原召回＋冻结内容 MLP | 2.233% | 0.01225 | 16.975% | 8.735% | 0.704% |
| 加历史查组，沿用冻结内容 MLP | 2.233% | 0.01225 | 16.867% | 9.835% | 0.870% |
| 加历史查组和锁定的 DCN V2 | **2.917%** | **0.01602** | 16.867% | 9.835% | 0.870% |

- 同候选排序对照的 HR@10 增加 **0.683 个百分点**，95% 配对 bootstrap 区间为 **+0.383～+0.992 个百分点**。NDCG@10 增加 0.00377，区间为 +0.00219～+0.00548。
- 种子 2026 有 64 个用户新增进前十、27 个用户掉出前十，命中净增 37。种子 2027 分别是 81 与 36，命中净增 45。两种子合计净增 82 次，其中 76 次来自头部广告。
- 添加辅助候选后，冷启动 Recall@500 增加 1.100 个百分点。长尾 Recall@500 增加 0.166 个百分点，区间下界为 0。固定预算也会挤掉一部分原召回结果，整体 Recall@500 低 0.108 个百分点，区间为 −0.267～+0.050 个百分点。

| 确认集目标分组 | 每种子人数 | 原 Top-10 命中合计 | 新 Top-10 命中合计 | 新 HR@10 |
|---|---:|---:|---:|---:|
| 冷启动 | 727 | 0 | 1 | 0.069% |
| 长尾 1–5 次 | 1,207 | 0 | 3 | 0.124% |
| 中频 6–50 次 | 2,071 | 16 | 18 | 0.435% |
| 头部超过 50 次 | 1,995 | 252 | 328 | 8.221% |

辅助通道在两个种子中各补到了 9 和 11 个原候选缺失的真实点击，其中各有 8 个被排序器送进 Top-100。最终有 **2 位不同用户的长尾目标**进了 Top-10，分种子累计 3 次。两组冷尾 Top-10 改善的区间下界均为 0，因此这部分作为可追溯的改善案例。确认集里唯一进入前十的冷启动目标来自原召回，排序把它从第 14 提至第 4。

例如用户 RID 134009，目标广告 OID 20004004871 在训练前缀里出现 4 次。两个种子的主召回 Top-500 都漏了这条广告。历史查组把它补进候选，冻结融合分将它排第 492，加入残差后分别来到第 6 和第 9。

对项目展示，完整方案可以讲清三条证据。先用同预算候选对照解释冷启动覆盖增加与总体覆盖代价，再用同候选排序验证端到端收益，最后用实际长尾样例追溯哪一个模块改变了结果。前期生成通道在同预算下未超过历史查组，适合作为设计对照。整体排序的改善已有独立离线确认，单桶 Top-10 的样本量仍小。

[开发集锁定与全部运行](development.json) · [确认集逐种子指标和差值区间](confirmation.json) · [实际冷尾命中明细](examples.json) · [候选和协议核验](verification.json)

## 本机复现

从仓库根目录执行。复跑以历史排序实验产物为前提，包括 `artifacts/baseline/data` 的小规模目录、`artifacts/ranking_v1` 的召回向量和早期统计、`artifacts/ranking_content_v1` 的内容 MLP、`artifacts/group_recall_v1/confirm` 的 5,000 人候选，以及 `reports/ranking_study/confirmation.json` 登记的排除队列数组。后者保留实验机器的文件路径和哈希。迁移到其他机器时，需准备相同文件并让登记路径指向对应副本。历史研究训练与候选生成见[原排序](../ranking_study/README.md)和[历史查组](../group_recall/README.md)。

已有产物时，用[单用户演示](../../docs/ranking_extension.md)现场重算完整路径。重跑训练时复制 `configs/channel_rank.json` 并修改 `output` 到新目录。历史锁定报告保留了研究版本 `5b52dbb` 的源码哈希。主分支整理了共享打分函数并新增演示 CLI，新的训练会生成自己的锁定摘要。

```bash
PY=python
CONF=configs/channel_rank_replay.json
cp configs/channel_rank.json "$CONF"
"$PY" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()); d["output"] = "artifacts/channel_rank_replay"; p.write_text(json.dumps(d, indent=2) + "\n")' "$CONF"

"$PY" scripts/prepare_channel_rank.py --config "$CONF" --stage cohorts
"$PY" scripts/prepare_channel_rank.py --config "$CONF" --stage proposals --split fit --workers 8

for seed in 2026 2027; do
  "$PY" scripts/prepare_channel_rank.py --config "$CONF" --stage retrieval --split fit --seed "$seed" --device cuda:5
  "$PY" scripts/build_channel_features.py --config "$CONF" --split fit --seed "$seed" --device cuda:5
  "$PY" scripts/build_channel_features.py --config "$CONF" --split dev --seed "$seed" --device cuda:5
  "$PY" scripts/run_channel_rank.py --config "$CONF" --stage fit --seed "$seed" --device cuda:5
done

"$PY" scripts/run_channel_rank.py --config "$CONF" --stage lock
"$PY" scripts/prepare_channel_rank.py --config "$CONF" --stage proposals --split fresh --workers 8

for seed in 2026 2027; do
  "$PY" scripts/prepare_channel_rank.py --config "$CONF" --stage retrieval --split fresh --seed "$seed" --device cuda:5
  "$PY" scripts/build_channel_features.py --config "$CONF" --split fresh --seed "$seed" --device cuda:5
done

"$PY" scripts/run_channel_rank.py --config "$CONF" --stage fresh --device cuda:5
```

本机验证覆盖每人 500 条候选、历史过滤、去重、候选来源、原 Top-10 保护和排序后候选集一致性。61 项单元测试通过。运行产生的完整特征、候选、日志和模型位于 `artifacts/channel_rank_v1`。
