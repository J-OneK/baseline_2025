# 项目文档

主分支展示一个固定的一层序列召回方案，以及支持它的训练目标消融、离线评估和向量检索链路。

| 想了解的内容 | 入口 |
|---|---|
| 项目介绍、安装、默认命令 | [中文 README](../README_CN.md) / [English README](../README.md) |
| 从原始数据到 Top-10 的代码讲解 | [代码导读](code_walkthrough.md) |
| 哪些指标可以引用、如何复现历史演示 | [版本与证据](release.md) |
| 主线与探索实验如何分开 | [分支说明](branches.md) |
| 原始消融、三个种子、Faiss 测量 | [历史研究报告](../reports/objective_study/README.md) |
| 上游基线用法与原始说明 | [上游中文文档](upstream_baseline_zh.md) |

## 当前目录

```text
configs/main.json                 主方案固定配置
scripts/run_main.py               单模型 prepare/train/evaluate/export/index/recommend
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
5. 运行 `run_main.py --stage recommend --split val --index 0`，查看广告 OID 和原始内积分数。

第 5 步需要本地已生成相应权重、向量和索引；从 Git clone 得到的代码和报告本身不包含这些大文件。已有历史产物的演示命令见[版本说明](release.md)。
