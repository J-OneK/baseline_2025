# 项目文档

主分支从序列召回与历史同组提议得到固定 Top-500，再用 DCN V2 基础分与 OneTrans 风格联合残差输出推荐。

| 想了解的内容 | 入口 |
|---|---|
| 项目介绍、安装、默认命令 | [中文 README](../README_CN.md) / [English README](../README.md) |
| 完整交互式教程：源码、公式、贯穿样例与实际执行轨迹 | [离线 HTML 讲义](project_masterclass.html) |
| 主召回、辅助通道与联合排序的完整报告：背景、数据、参数、结果、失败与处理 | [项目完整报告](PROJECT_FULL_REPORT.md) |
| 面试深挖：64道核心问题及排序补充、参考回答、连续追问、失败案例与证据 | [项目面试深挖文档](PROJECT_INTERVIEW_DEEP_DIVE.md) |
| InfoNCE负例数16/64/128：三种子对照、配对区间、候选覆盖与成本 | [负样本数量补充实验](negative_count_study.md) |
| 生成式召回：RQ-VAE、Semantic ID、自回归解码、对照与真实结果 | [生成式探索入口](generative_retrieval.md) |
| 真实用户从 Parquet、缓存、候选、token 到 Top-10 的样式和形状 | [数据链路](data_flow.md) |
| 联合排序入口和 10,000 人确认 | [排序链路](ranking_extension.md) / [实验与计时](../reports/sequence_rank/README.md) |
| 原 MLP 与 DCN V2 选型及 6,000 人验证 | [DCN V2 依据](../reports/channel_rank/README.md#遇到的问题与选择依据) |
| 从原始数据到 Top-10 的代码讲解 | [代码导读](code_walkthrough.md) |
| 哪些指标可以引用、如何复现历史演示 | [版本与证据](release.md) |
| 主线与探索实验如何分开 | [分支说明](branches.md) |
| 原始消融、三个种子、Faiss 测量 | [历史研究报告](../reports/objective_study/README.md) |
| 上游基线用法与原始说明 | [上游中文文档](upstream_baseline_zh.md) |

## 当前目录

```text
configs/main.json                 主方案固定配置
scripts/run_main.py               召回 prepare/train/evaluate/export/index/recommend
scripts/recommend_sequence_rank.py 联合排序 Top-10 和数据 trace
scripts/recommend_channel_rank.py  双通道与 DCN V2 单独复查
scripts/run_baseline.py           原始小规模基线对照复现
scripts/run_objective_study.py    目标消融与三种子确认实验复现
taac_retrieval/                    当前数据、训练、评估、检索代码
tests/                            协议、损失、模型与检索单元测试
reports/objective_study/          历史结果、小型 JSON 与图表，保留原始内容
docs/                             当前文档与上游文档
model.py、dataset.py 等            保留的原始基线及兼容代码
TencentGR-1M/                      本地原始数据，Git 忽略
artifacts/                        本地缓存、权重、索引与新运行输出，Git 忽略
```

目录不再按学习天数命名。历史 JSON、报告标题和代码摘要中的旧名称记录当时的实验环境，不是当前执行入口；解释见[版本说明](release.md)。

## 推荐的展示顺序

1. 用 README 中的流程图说明输入、训练目标和检索出口。
2. 打开 `prepare.py` 的 `clean_and_split`，解释严格时间截断和首次出现点击留出。
3. 打开 `model.py` 和 `objectives.py`，解释广告共享编码、因果序列和 InfoNCE-64。
4. 展示历史报告的对照表、冷尾分桶以及原始内积 HNSW 的失败记录。
5. 运行 `python scripts/recommend_sequence_rank.py --rid 311428 --show-target --trace`，追踪冷启动广告从补入候选到第 8 名。

第 5 步需要本地已生成相应权重、向量和索引；从 Git clone 得到的代码和报告本身不包含这些大文件。已有历史产物的演示命令见[版本说明](release.md)。
