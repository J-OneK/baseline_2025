# 腾讯广告序列召回与向量检索

基于 TAAC2025 / TencentGR-1M 的本地推荐召回项目：从公开 Parquet 构建严格时间留出样本，融合广告 ID、匿名属性和文本向量，用因果 Transformer 编码用户兴趣，再通过精确内积与 Faiss HNSW 检索广告。

项目按数据、研究问题和工程模块组织，已完成的实验保留可追溯记录。当前继续研究冷启动、长尾与候选覆盖，再依据 Recall@500 决定是否引入独立排序模型。

| 阅读入口 | 内容 |
|---|---|
| [项目概览](PROJECT_OVERVIEW.md) | 架构、协议、环境和常用命令 |
| [完整代码讲解](PROJECT_FULL_WALKTHROUGH.md) | 原始数据、ID 映射、样本、模型、训练和推荐全流程 |
| [文档导航](docs/index.md) | 全部文档和研究报告入口 |
| [目录结构](docs/directory_layout.md) | 语义目录与旧路径兼容说明 |
| [数据与基线入门](BASELINE_GUIDE.md) | 最初的基线、行为分支与数据适配 |
| [目标函数与检索报告](reports/objective_study/README.md) | 消融、扩大复核、Faiss、分桶与失败分析 |
| [监督范围对照](reports/infonce_supervision/README.md) | 固定 InfoNCE-64，全部事件与仅点击监督 |
| [冷启动与长尾研究](reports/cold_start/README.md) | Top-500 覆盖、ID 遮蔽、文本通道与固定预算融合 |
| [Transformer 层数对照](reports/depth_study/README.md) | 1／2／4 层、共享初值核对、总体与冷／尾覆盖及成本 |
| [层数实验协议](docs/depth_study.md) | 控制变量、嵌套初始化、选择规则与复现命令 |

## 项目结构

```text
taac_retrieval/                 数据、模型、损失、评估、导出与检索
configs/                       按实验问题命名的配置
scripts/                       可复现运行入口
tests/                         关键逻辑和实验协议检查
artifacts/baseline/             基线缓存与匹配模型
artifacts/objective_study/      目标函数研究、扩大模型与索引
artifacts/infonce_supervision/  监督范围补充消融
artifacts/cold_start/           冷启动和长尾研究
artifacts/depth_study/          层数对照、嵌套初始权重与覆盖诊断
reports/                       对应实验的指标、曲线与解释
docs/                          导航、目录说明和上游原始文档
```

`TencentGR-1M/` 是本地原始数据。约 28.1 万有效用户、270 万候选和 1 万评估用户对应本地代理任务，不是官方私有测试；当前只使用 81 号预提取文本向量。没有线上 A/B 或比赛排名声明。

## 运行一次推荐

当前机器使用 `.project_venv`。环境说明见[项目概览](PROJECT_OVERVIEW.md)。

```bash
cd /home/jyk/project/baseline_2025
.project_venv/bin/python -m taac_retrieval.recommend \
  --checkpoint artifacts/objective_study/confirmation/selected_seed2026/best.pt \
  --cache artifacts/objective_study/scale_data \
  --export-dir artifacts/objective_study/export \
  --index-dir artifacts/objective_study/ann \
  --device cuda:0 --split test --index 0
```

这是离线固定用户历史的完整推荐演示。兴趣向量编码、索引与历史过滤分别计时；单次预热请求不是端到端生产 p95。

## 复现实验

新实验使用新输出目录：

```bash
.project_venv/bin/python scripts/run_baseline.py \
  --root artifacts/baseline_reproduction \
  --report-dir reports/baseline_reproduction --gpus 0,2

.project_venv/bin/python scripts/run_objective_study.py \
  --root artifacts/objective_study_reproduction \
  --baseline-root artifacts/baseline_reproduction \
  --report-dir reports/objective_study_reproduction --gpus 0,2,4
```

旧日期目录保留为相对符号链接，使历史 checkpoint 和结果仍可读取；新命令使用语义名字。冻结的旧计划和源码快照保持原样，不通过改写历史文件来绕过重入检查。

## 上游来源

序列建模起点来自 [腾讯广告算法大赛开源基线](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025)。根目录 `main.py/dataset.py/model.py/infer.py/eval.py` 是旧基线链路；当前报告由 `taac_retrieval/` 生成，二者的 checkpoint 不能混用。原基线说明保留为[中文](docs/upstream_baseline_zh.md)和[英文](docs/upstream_baseline_en.md)。
