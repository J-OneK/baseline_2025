# 生成式召回探索入口

已在独立分支 `research/generative-retrieval` 完成一版小型生成式检索及真实对照，代码和产物位于：

```text
/home/jyk/project/baseline_2025_generative
```

本机打开 [完整实现、流程讲解、实验结果与失败分析](/home/jyk/project/baseline_2025_generative/reports/generative_study/README.md)。

后续已完成[正确前缀诊断与后缀编码改进](/home/jyk/project/baseline_2025_generative/reports/generative_improvement/README.md)：只用训练点击次数重排桶内编号，与原冻结编码器方案作两个种子配对对照。

链路为：

```text
广告81号向量 → RQ-VAE → 三位内容码 + 三位去重码
用户历史 → 原因果编码器 → 自回归解码 → 目录约束Beam Search → 广告OID
```

这是借鉴 TIGER 的小型生成式探索，不是完整论文复现，也未使用预训练 LLM。

相同小规模验证集（93,586训练用户、153.9万广告、2,000评估用户）上，两个种子均值：

| 方案 | HR@10 | NDCG@10 | Recall@100 |
|---|---:|---:|---:|
| 原点击InfoNCE-64 | **2.1750%** | **0.011616** | **8.4500%** |
| 从零训练语义ID生成式 | 0.0500% | 0.000375 | 0.0500% |
| 固定原编码器，只训练生成头 | 0.1000% | 0.000433 | 0.2000% |
| 训练点击对齐后缀，固定原编码器 | 0.5500% | 0.001987 | 2.4000% |
| 简单训练点击热度基线 | 0.7000% | 0.005155 | 3.1000% |

还完成了保持前缀树相同的随机映射对照、碰撞与后缀诊断、独立概率重算，以及53项测试。未进行测试集评估，未融合或替换主模型。这些数字不可与不同规模的主项目三种子结果混用。

这条分支还把问题分成内容桶预测与桶内广告区分。诊断时直接给模型真实内容桶，原冻结编码器生成器的条件 HR@10 为 **17.425%**，接近同桶均匀随机期望 **17.652%**。提供了正确前缀的条件指标用于检查瓶颈。常规评估仍靠模型自由解码。把局部去重编号按训练点击频次重排后，常规 HR@10 从 **0.10% 到 0.55%**，收益来自头部。这个对照说明广告编号的学习难度与行为先验都影响了结果，也为后续只生成属性组、用目录找广告的实验提供了方向。

当前适合用来学习 Semantic ID、RQ-VAE、teacher forcing、Beam Search 和实验排查。后缀改法改善了生成式原型，但收益集中于头部，仍弱于简单热度和原向量模型；不能称为主系统效果提升，也未解决冷启动和长尾。

在主目录查看研究报告，不必切换已有工作文件：

```bash
git show research/generative-retrieval:reports/generative_study/README.md
```

本机直接演示：

```bash
cd /home/jyk/project/baseline_2025_generative
/home/jyk/project/baseline_2025/.project_venv/bin/python \
  -m taac_retrieval.generative_recommend --device cuda:0 --index 0 --show-target
```

本机的权重、码表和逐用户结果由Git忽略；研究分支提交源码、配置与小型报告。重新克隆后需要另行准备原数据及实验产物。


## 生成两级属性组的辅助通道对照

`research/generative-group-recall` 曾用冻结用户查询预测 118→102 的广告属性组，再从组内选冷尾候选。在固定 5,000 人复核队列与 Top-500 预算下，生成查组的冷启动 Recall@500 为 8.22%，历史查组为 8.73%，两者 HR@10 均为 1.84%。主分支采用[历史查组与列表排序](ranking_extension.md)。

生成路径有时能匹配近期历史没有出现过的组。组内只补到很少的真实点击，补入后又停在榜尾。实验在本机可用 `git show research/generative-group-recall:reports/generated_group/README.md` 查看。


## 在 Top-100 后做生成式重排的另一项尝试

在 `research/generative-reranking` 上，实验将静态打分器和条件于已选广告的 GRU 重排器接在旧 MLP Top-100 后面。两者共享候选、输入和训练设置。2,000 人开发集的原 MLP HR@10 为 2.40%，两种重排都为 2.35%，配对差值区间跨零。长尾目标进入该入口的比例只有 0.47%。训练列表由真实日志跨请求拼成、每份最多 12 条，推理列表为 100 条高分广告。候选的漏失可以直接测量，代理列表和监督差异给出改善训练设计的方向。对应源码与记录可用 `git show research/generative-reranking:reports/slate_ranking/README.md` 查看。 [当前排序决策](../reports/channel_rank/README.md#遇到的问题与选择依据)
