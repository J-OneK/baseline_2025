# 从 Parquet 到 Top-10：核心代码导读

这个项目训练一个用户历史编码器和一个共享广告编码器。用户历史输出查询向量 `q`，广告输出向量 `v`，召回分数是 `q·v`。训练让历史后的真实点击广告分数高于采样负广告，推断时在固定候选库中检索分数最高的十个未见广告。

主分支固定一层模型，目标是把数据、训练、评估和检索做成可以追踪和复现的链路。运行入口见 [README](../README_CN.md)，结果口径见[版本说明](release.md)。

## 1. 代码来源与整体结构

根目录保留 TAAC 2025 提供的 SASRec 风格基线。当前 [taac_retrieval/model.py](../taac_retrieval/model.py) 从根目录 [model.py](../model.py) 复用 `FlashMultiHeadAttention` 与 `PointWiseFeedForward`，重新实现向量化特征拼接、序列掩码和统一训练接口。`main.py`、`dataset.py`、`infer.py` 等旧入口继续保留，用法见[上游文档](upstream_baseline_zh.md)。

| 模块 | 当前职责 |
|---|---|
| [prepare.py](../taac_retrieval/prepare.py) | 扫描 Parquet、清洗、时间留出、ID 对齐和特征数组 |
| [data.py](../taac_retrieval/data.py) | 内存映射缓存、构造训练/验证/测试样本、负采样 |
| [model.py](../taac_retrieval/model.py) | 广告与用户编码、因果历史编码 |
| [objectives.py](../taac_retrieval/objectives.py) | 监督位置、BCE 与 InfoNCE |
| [train.py](../taac_retrieval/train.py) | 确定性种子、优化、验证选 epoch、checkpoint |
| [metrics.py](../taac_retrieval/metrics.py) | 全库精确检索、HR/NDCG、热门基线 |
| [cohorts.py](../taac_retrieval/cohorts.py)、[audit.py](../taac_retrieval/audit.py) | 固定评估用户、排除旧测试用户、协议检查 |
| [export.py](../taac_retrieval/export.py) | 固定模型广告向量、验证/测试查询与身份信息 |
| [benchmark_ann.py](../taac_retrieval/benchmark_ann.py)、[ann.py](../taac_retrieval/ann.py) | 索引构建、几何变换、过滤、召回与延迟比较 |
| [recommend.py](../taac_retrieval/recommend.py)、[predict.py](../taac_retrieval/predict.py) | 持久化 ANN / 精确检索的单用户演示 |
| [coverage.py](../taac_retrieval/coverage.py) | 固定验证集的多 K 覆盖和频次分桶诊断 |
| [study.py](../taac_retrieval/study.py)、[objective_report.py](../taac_retrieval/objective_report.py) | 原始目标消融、三种子确认、统计报告生成 |

最后一组保留历史证据的复现能力，不是推荐请求的依赖。报告生成器统一命名为 `objective_report.py`，历史源码摘要仍保留整理前的名字。

## 2. 原始数据长什么样

准备代码读取下载目录中的这些内容：

| 路径 | 代码读取的结构 | 用途 |
|---|---|---|
| `seq/*.parquet` | `user_id`、`seq`；事件含 `item_id`、`timestamp`、`action_type` | 用户行为历史 |
| `item_feat/*.parquet` | `item_id` 与匿名字段 100、117 等 | 广告离散属性 |
| `user_feat/*.parquet` | `user_id`、单值与多值匿名字段 | 用户静态特征 |
| `mm_emb/emb_81_32_parquet/*.parquet` | `anonymous_cid`、`emb` | 32 维预提取内容向量 |
| `candidate/*.parquet` | `item_id` | 检查官方候选对本地留出目标的覆盖 |
| `indexer.pkl` | 原始 ID 与重编号 ID 的映射 | 还原 OID、跨表关联 |

下面只是说明结构的虚构样例，不是实际用户记录：

```json
{"user_id": 42, "seq": [
  {"item_id": 101, "timestamp": 1000, "action_type": 0},
  {"item_id": 202, "timestamp": 1010, "action_type": 1}
]}
```

原始 `action_type=0` 是曝光，`1` 是点击。缓存内统一编码为 `0=填充/用户 token`、`1=曝光`、`2=点击`、`3=未知`。未知事件保留上下文，但不作为监督目标。字段是匿名类别，不能从字段编号推断职业、品类或用户兴趣的自然语言含义。

“81 号向量”的 **81 是向量家族编号，32 才是本项目使用的维度**。代码读取现成 `emb`，按原始广告 OID 关联，经 `Linear(32, hidden)` 接入模型；不处理原始文本。缺失向量填零，额外保存 `mm_present.npy` 供诊断。对长度和有限值的检查避免错误向量悄悄进入训练。

## 3. 清洗、截断与留出

核心函数为 `prepare.clean_and_split`：

1. 去掉空事件、无效广告 ID 和缺失时间戳，按时间稳定排序，只保留最后 100 个事件。
2. 在这个保留序列内找“广告此前没有出现过”的点击，且其严格更早的历史至少有 5 个事件。
3. 最后一个合格点击作测试目标；更早且时间严格小于测试目标的最后一个合格点击作验证目标。
4. 训练和验证历史都停在验证目标时间之前；测试历史停在测试目标时间之前。

同一时间戳事件整批排除在目标历史之外，避免同秒事件顺序被误当成先后因果关系。首次出现规则同时排除“前面已经曝光过同一广告”的留出目标；数据没有曝光点击归因 ID，因此采用这个保守规则。这里的首次出现仅相对于保留的最多 100 个事件，不代表用户一生首次接触广告。

例如，验证目标是 `t=20` 的广告 B、测试目标是 `t=30` 的广告 C：训练只看 `<20` 的事件，验证输入同样只看 `<20`；测试可以看 `<30` 的事件，因此可能包含 B。测试输入不用于更新模型。

有效用户不足两个这样的点击就被剔除。历史确认抽样 300,000 名用户，保留 280,985 名；这不是训练整个百万用户数据。`cohorts.freeze_cohort` 排除小规模开发阶段已经看过测试结果的用户，再用固定 seed=3107 抽出 10,000 名评估用户，验证与测试分别使用这些用户的不同留出点击。

## 4. 三层 ID 与候选库

跨表关联要区分三个 ID：

```text
原始广告 OID → 公开数据重编号 RID → 本地候选数组 local ID
```

`indexer.pkl` 负责前两者映射；`catalog_rids.npy`、`catalog_oids.npy` 将本地数组索引映射回去。局部 ID=0 永远是填充，不能作为推荐。推荐最终返回 `item_oid`，不是模型 embedding 表的行号。

候选库是抽样用户保留序列中所有广告的并集，作为公共静态目录固定给所有模型，不按目标是否点击筛选。正式确认中有 2,708,122 个广告；没有直接用 660,000 个官方候选，因为其对本地逐用户留出目标覆盖不完整。该目录包含训练未见广告，因此属于已知公共目录的传导式评估，不能声称生产级全局时间回测。

`train_counts` 只统计所有用户的**训练前缀**事件：

- 大于 0 的广告进入训练 ID 词表和负采样池。
- 训练未见广告的独立 ID embedding 使用 0；广告属性类别词表也只用训练可见广告拟合，OOV=0。
- 内容向量和已知属性仍可提供共享表示，但这不保证冷启动召回有效。

广告事件计数包含曝光、点击和未知，不等同于点击次数。`cold_0`、`tail_1_5`、`mid_6_50`、`head_gt50` 也是按这份计数分桶。

## 5. 缓存与样本张量

`prepare.py` 将数据保存为 `.npy` 数组与 `metadata.json`；`PreparedData` 以内存映射方式载入。元数据记录源文件签名、准备参数、词表策略与审计统计；缓存设置或源文件签名改变会拒绝复用，防止悄悄混用不同实验。

| 数组/张量 | 形状或含义 |
|---|---|
| `items/actions/timestamps` | `[有效用户数, 100]`，清洗后的序列 |
| `train_cut/val_cut/test_cut` | 每用户各阶段历史截止位置 |
| `val_pos/test_pos` | 目标在保留序列中的位置 |
| `item_sparse` | `[候选数+1, 13]` |
| `item_mm81` | `[候选数+1, 32]` |
| 模型输入 `items/kinds/actions` | `[batch, 101]`，左填充，含一个显式用户 token |
| 训练 `positive/target_actions` | 与序列位置对齐的下一个广告及动作 |

`make_input` 保留一个用户 token 和末尾历史事件。用户 token 放在其历史之前，`kinds=2`；广告 token 为 `kinds=1`。训练用前缀的前 `n-1` 个广告预测前缀内下一个广告，第一个广告可以由用户 token 预测。验证/测试把完整合法历史放入输入，用最后一个有效位置输出预测目标。

## 6. 广告编码与用户编码

`SequenceRetriever.encode_items` 对候选广告使用同一套参数：

```text
ID embedding (32)
13 个广告属性 embedding (13×32)
81 号内容向量 → Linear(32,32)
                   ↓ 拼接
             Linear → ReLU → v (32)
```

这是共享特征编码器，既编码历史中的广告，也编码正负目标和全库候选。`encode_users` 将用户 ID、4 个单值字段与4个多值字段的 embedding 拼接后线性投影；多值字段按元素 embedding 求和。用户表示作为序列中的用户 token，不是每个位置再拼一次用户 ID。

`encode_history` 叠加位置 embedding，经过 dropout，再进入一层因果自注意力与前馈网络，使用残差、LayerNorm 和填充掩码。因果掩码禁止位置看见后面的广告。`predict` 取最后位置的 32 维表示作为 `q`。

一层指一个 Transformer block，不是整个模型只有一个线性操作；ID/属性/内容编码、注意力、FFN 和输出归一化仍共同参与训练。主方案没有额外加历史动作 embedding，`actions` 变体只用于原始消融。主模型也没有把曝光从历史序列中删掉：**仅点击监督只限制目标位置，不限制历史输入**。

## 7. 点击监督与 InfoNCE-64

`supervision_weights` 在 `target_actions==2` 的位置提供监督，填充、曝光和未知目标不产生主方案损失。一次训练批次的全部历史仍通过因果编码，取被监督位置的状态为查询。

`sample_negatives` 从训练可见广告均匀采样，每个正目标采 64 个负样本。拒绝该用户训练前缀内的广告，因而也排除当前正例；不查询验证/测试目标来“保护”它们。允许负样本之间重复，没有 hard-negative 或流行度采样。

对一个监督位置，令 `s+=q·v+`、`sj-=q·vj-`：

```text
logits = [s+, s1-, ..., s64-] / temperature
loss = CrossEntropy(logits, target=0)
```

`sampled_info_nce` 把正例放在第 0 列，主配置 temperature=1。训练函数按有效点击位置取均值，执行反向传播、梯度裁剪与 Adam 更新。固定种子控制初始化、用户打乱与负采样；每轮验证后仅按 NDCG@10、再 HR@10 比较并保存 `best.pt`。

原始研究保留 `all_bce`、`weighted_bce`、`click_bce` 与 `click_infonce`。`variant=baseline` 在代码中表示“不加历史动作 embedding”，**不是**自动选择原 BCE 目标；损失由 `objective` 决定。

## 8. 全候选评估

`metrics.evaluate` 先编码全部候选广告，再逐批编码用户。`exact_topk` 按块计算 `q @ V.T`，过滤该用户输入历史中的广告与填充项，合并每块 Top-K 得到全库精确 Top-10。没有仅采一百个负例的评估捷径。

每用户只有一个目标广告，HR@10 为是否命中；若目标在第 `r` 名（从 1 开始），NDCG@10 为 `1/log2(r+1)`，否则为 0。平均值按用户计算。热门基线只用训练前缀频次，同样过滤各自历史。

`evaluate_checkpoint.load_model` 检查缓存准备签名，恢复 checkpoint 中的固定评估用户。`coverage.py` 在验证集补 Top-10/100/500，并按训练频次、内容是否存在等维度诊断。它回答“目标有没有进入候选”，与独立排序模型能否把目标排前十是不同问题。

## 9. 向量导出与 Faiss

`export.py` 导出同一个固定 checkpoint 的广告向量、用户查询、历史与 OID 映射，并保存 checkpoint SHA。内积保留向量长度信息，主流程没有归一化为余弦。

历史实验中直接构建原始内积 HNSW 未达到验证集 98% 的 Top-10 重合门槛，因此仅根据验证查询尝试等价的 MIPS→L2 构图。在 `ann.lift_database` 中：

```text
广告：v' = [v, sqrt(R² - ||v||²)]
查询：q' = [q, 0]
R² = max(所有广告向量的平方范数)

||q' - v'||² = ||q||² + R² - 2 q·v
```

对固定查询，前两项是常数，距离越小等价于原内积越大。逻辑表示仍为 32 维，Faiss 内部构图为 33 维。`InnerProductFromL2` 将距离恢复成内积分数。变换保证精确排序等价，HNSW 本身仍是近似检索，故需要对精确 FlatIP 测量重合率。

`search_filtered` 多取候选后过滤无效 ID、重复结果和历史广告，必要时扩大搜索深度补足十个。最后直接按召回分数输出，没有额外学习的精排。`benchmark_ann.py` 将索引持久化，记录构图参数、导出身份、验证门槛与测量条件；详细数值见[历史报告](../reports/objective_study/README.md)。

## 10. 一次推荐请求

`recommend.py` 的执行链路是：

```text
载入 checkpoint + 缓存 + 导出身份 + 持久化索引
→ 校验 checkpoint SHA / 索引身份 / 维度与候选数
→ 读取固定验证队列中 index 指定的用户历史
→ 因果编码得到 q
→ HNSW 搜索并过滤历史
→ local ID 映射到 item_oid，返回十个广告和内积分数
```

`--index` 是固定队列位置，不是可以在线新建的用户 ID。`offline_target_oid` 只作为演示后的离线对照，不参与搜索。输出中的单次预热耗时单独列出编码与索引过滤，不包含进程启动和加载，也不是性能压测的 p95。

## 11. 读懂实验结论

主方案提升主要来自头部广告，冷启动和长尾 Top-10 仍弱。对项目的准确描述是：在固定公共目录、严格逐用户留出协议下，完成了一层序列模型的训练目标对照和完整检索链路，并保留了失败分析。

更深编码器、全事件 InfoNCE、ID 遮挡和内容融合在[研究分支](branches.md)中。这些尝试有助于讨论假设与取舍，但尚不能替代主方案已有的三种子测试和索引证据。主分支冻结范围、历史指标和使用边界以[版本说明](release.md)为准。
