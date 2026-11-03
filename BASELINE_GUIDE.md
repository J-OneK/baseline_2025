# 腾讯广告序列召回：数据与基线入门

后续目标函数与检索研究已扩展为 [完整项目](PROJECT_OVERVIEW.md)，包含训练目标消融、扩大复核、Faiss 基准和面试材料；本文保留基线阶段的代码入门与实验口径。

这份项目从 TAAC2025 开源代码出发，完成数据适配、可复现本地评估、SASRec 风格序列召回基线，以及历史行为类型的对照实验。

**先读 [实验结果](reports/baseline/README.md)。** 所有指标来自实际训练，负面和不确定的结果也保留。它是本地代理任务，不是官方排行榜成绩。

## 当前完成到哪里

| 阶段 | 已完成内容 |
|---|---|
| 数据管线 | 修复 ID/候选字段适配，处理缺失行为和内容，构建时间留出及数组缓存，检查全部缓存 |
| 训练与评估 | 跑通训练、验证选择、精确召回、测试指标、热门广告对照，保存模型和逐用户预测 |
| 历史行为对照 | 加入已观察历史行为的 Embedding，与基线保持其他设置一致，使用两个随机种子对比 |

本次从 100,000 名随机用户保留 93,586 名符合留出要求的用户，训练前缀有 6,455,291 个事件；候选池包含 1,538,730 个广告。验证、测试各评估固定的 2,000 名用户。只使用 `81` 号预提取文本向量，未接入原始图像或语义 ID 生成。

原始数据仍在 `TencentGR-1M/`，没有改写。准备好的数组缓存约 429 MiB，位于 `artifacts/baseline/data/`。

## 从哪里读代码

建议按下面的顺序阅读 `taac_retrieval/`。这个入口使用数组直接查表，避免原代码在每次 forward 中循环拼装 Python 特征字典。

1. `prepare.py`：流式读取 Parquet，清理、排序和切分序列，连接 ID、属性、内容向量，保存数组及数据报告。
2. `data.py`：生成错开一位的训练输入/正样本，以及完整保留最后已观察事件的验证/测试输入；负采样仅查看训练前缀。
3. `model.py`：广告 ID、属性、文本向量融合；用户画像融合；因果 Transformer；可开关的历史行为 Embedding。
4. `train.py`：固定种子，训练 BCE，按验证指标选模型，再运行一次测试。
5. `metrics.py`：分块精确内积 Top-10、历史过滤、HR/NDCG、热门广告对照，以及训练已见/未见目标分桶。
6. `audit.py`、`report.py`：验证缓存不变量，核对两种模型的初值和设置，汇总逐种子结果与不确定性。
7. `predict.py`：加载 checkpoint，对一个留出用户返回原始广告 ID 和匹配分数。

注意力与前馈模块直接复用根目录 `model.py` 的 `FlashMultiHeadAttention` 和 `PointWiseFeedForward`。这里采用同一套修正后的用户/物品掩码、有效位置编码和数组查表，实现的是适配后的 SASRec 风格基线；不声称与官方原脚本逐位等价。

两种实验模型只在一处不同：`use_actions=True` 时，将**已经观察到的历史事件**的行为向量加到广告 token 上。模型不会读取下一个目标的行为类型作为输入。两种模型都分配同一张行为 Embedding 表，从而保证其余参数的随机初始化完全一致；基线不使用这张表。

## 时间留出协议

- 各用户的有效事件先按时间稳定排序，保留最近至多 100 个事件。
- 目标只选“该广告此前在该用户序列中没有出现过”的点击。这样不会把目标广告的先前曝光作为输入透露给模型。
- 预留最后两次符合条件、时间不同的点击，分别作为验证、测试目标。每个目标的上下文时间必须严格早于目标时间，同时间事件也不进入上下文。
- 所有训练仅使用验证目标时间之前的前缀。后来测试时，验证点击已经发生，因此可以出现在测试上下文里。
- 点击不足或验证前缀少于 5 个事件的用户被排除；本次排除 6,414 名。
- 曝光、点击、未知行为分别编码为 1、2、3，0 为填充/用户 token。空行为保留为未知上下文，但不会充当监督目标。

这是**逐用户的新广告点击代理任务**。它主动排除了重复广告点击，不等价于官方未来点击标签，也不保证不同用户之间的全局时间隔离。用户画像、广告属性和内容来自公开静态快照，不能声称是线上按时可用特征回放。

## 候选、词表和负样本

候选池为所选公开序列中全部不同广告。它没有根据模型得分或目标点击逐个补样，也不会随实验变动；在这个公开目录可见的假设下，目标覆盖率为 100%。它不是“1 个目标加 99 个随机负样本”的评估。

官方 `candidate/` 的 66 万广告覆盖本地验证目标约 73.5%、测试目标约 80.9%，因此本项目没有直接拿官方池评价全部本地目标。

广告 ID 和属性词表仅从训练前缀可见广告拟合，未见值映射为 0。内容向量按原始广告 ID 连接，缺失向量补 0；候选内容的有效覆盖率约 94.7%。同一广告的内容编码器用于训练正负样本和推理候选。

训练保留原基线的基本目标：每个已知行为的下一广告为正样本，随机抽取一个训练可见广告作负样本，并排除该用户训练前缀中出现过的广告。曝光也是下一事件预测的正样本；它不直接等于负样本。负采样不读取未来验证或测试标签。

评估对所有候选做精确内积检索，并过滤用户历史广告。这里没有引入 ANN 近似误差；Faiss 的速度/精度权衡留给后续阶段。

## 已修复的原入口问题

- `dataset.py`：统一内容字典的字符串 OID 查找；用户 token 不再误查同数值广告 RID 的内容；真实用户 ID 列表替代 `uid+1` 假设；空行为保留为 -1；正确处理 Arrow 切片偏移；测试输入不再无条件丢掉最后事件。
- `infer.py`：适配 `item_id`、`cold_start` 和空候选特征；统一内容 OID；用本地路径和受检查的子进程调用代替作者机器路径。
- `main.py`、`eval.py`：移除生效代码中的作者数据路径，尊重环境变量；原训练入口支持 `--data_dir`、`--seed`、`--num_workers`。

**本项目指标使用 `taac_retrieval` 新入口。** 原 `main.py` 仍保留随机用户划分与验证 BCE，原 `eval.py` 仍是旧模型的预测导出器。新旧 checkpoint 结构不同，不能混用；旧 Faiss C++ 工具也没有作为本次验收的一部分编译验证。

## 复现

在当前机器使用已有 `taac` 环境：

```bash
cd /home/jyk/project/baseline_2025
conda activate taac
python -m unittest discover -s tests -v

# 新目录不会覆盖本次实验；默认每个 GPU 同时运行一个模型。
python scripts/run_baseline.py \
  --root artifacts/baseline_reproduction \
  --gpus 0,1 \
  --config configs/baseline.json
```

脚本依次执行测试、准备数据、全量缓存审计、两种模型各两个种子的训练、结果汇总。只用一张卡可传 `--gpus 0`；四张空闲卡可传 `--gpus 0,1,2,3`。模型输出已经存在时会停止，避免覆盖实验。

也可以分步运行：

```bash
python -m taac_retrieval.prepare \
  --data-dir TencentGR-1M --output artifacts/my_run/data \
  --max-users 100000 --eval-users 2000 --seed 2026

python -m taac_retrieval.audit \
  --cache artifacts/my_run/data --output artifacts/my_run/data_audit.json

python -m taac_retrieval.train \
  --cache artifacts/my_run/data --output artifacts/my_run/baseline_seed2026 \
  --variant baseline --seed 2026 --epochs 5 --batch-size 128 --device cuda:0

python -m taac_retrieval.train \
  --cache artifacts/my_run/data --output artifacts/my_run/actions_seed2026 \
  --variant actions --seed 2026 --epochs 5 --batch-size 128 --device cuda:0

python -m taac_retrieval.report \
  --root artifacts/my_run --output artifacts/my_run/report --seeds 2026
```

单用户召回示例，模型保持训练时所用配置和目录：

```bash
python -m taac_retrieval.predict \
  --cache artifacts/baseline/data \
  --checkpoint artifacts/baseline/actions_seed2026/best.pt \
  --split test --index 0 --device cuda:0
```

输出包括用户 RID、真实留出广告 OID、Top-10 原始广告 OID 和内积得分。匿名数据没有原始标题、图片，因此这里展示的是 ID 和分数。

## 保存的实验材料

- `configs/baseline.json`：数据和训练超参数。
- `reports/baseline/README.md`、`metrics.json`：逐种子结果、均值和配对用户置信区间。
- `reports/baseline/data_audit.json`：全量缓存的 35 项不变量检查。
- `reports/baseline/environment.json`、`code_manifest.json`：实际环境和源码摘要。
- `artifacts/baseline/*_seed*/config.json`：每次运行的实际配置、共同初值 SHA-256。
- `artifacts/baseline/*_seed*/events.jsonl`、`history.json`：训练与每轮验证日志。
- `artifacts/baseline/*_seed*/best.pt`：仅按验证集选出的模型。
- `artifacts/baseline/*_seed*/test_predictions.npz`：用户级预测及指标，可重新核算。

首轮结果中，行为 Embedding 的平均 NDCG 高于基线，HR 均值相同，但两个种子的变化方向不一致，置信区间跨过 0；当前证据不足以声称稳定改善。两个模型对本次 294 个训练未见测试目标均未命中，说明“能够读取内容”还不等于“解决了冷启动”。

后续优先依据验证集研究点击目标加权和训练目标对齐，再考虑负样本数量/InfoNCE；每项改动单独做对照。扩大评估样本、使用更多种子后再判断收益，不以测试集挑选超参数。
