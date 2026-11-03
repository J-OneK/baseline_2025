# 目录结构与历史兼容

当前目录按工作内容组织：

```text
baseline_2025/
├── taac_retrieval/            数据准备、序列模型、损失、评估与检索
├── configs/
│   ├── baseline.json         数据与基线/行为分支对照
│   ├── objective_study.json  目标函数消融和扩大复核
│   ├── ann_geometry.json     索引几何与搜索预算
│   ├── infonce_supervision.json  固定 InfoNCE 的监督范围对照
│   ├── cold_start.json       冷启动、长尾与候选覆盖
│   └── depth_study.json      Transformer 层数对照
├── scripts/
│   ├── run_baseline.py       基线复现入口
│   ├── run_objective_study.py  目标函数与检索研究入口
│   ├── run_infonce_supervision.py  监督范围研究入口
│   ├── run_cold_start_pipeline.py  冷启动研究完整入口
│   └── run_depth_study.py    共享初值与层数对照入口
├── artifacts/
│   ├── baseline/            基线缓存与匹配种子的模型
│   ├── objective_study/     开发消融、扩大模型、导出向量和 ANN
│   ├── infonce_supervision/ 全部事件 InfoNCE 对照
│   ├── cold_start/          召回覆盖诊断与内容泛化实验
│   └── depth_study/         嵌套初始权重、深度模型与覆盖诊断
├── reports/
│   ├── baseline/            数据与基线报告
│   ├── objective_study/     完整模型和 ANN 实验报告
│   ├── infonce_supervision/ 监督范围报告
│   ├── cold_start/          冷启动、长尾和候选覆盖分析
│   └── depth_study/         层数、分桶、配对区间和成本
├── tests/                   关键行为与实验协议检查
├── docs/                    文档导航、结构说明、上游原始说明
├── PROJECT_OVERVIEW.md      项目概览与常用命令
├── PROJECT_FULL_WALKTHROUGH.md  逐代码完整讲解
└── BASELINE_GUIDE.md        数据与基线入门
```

`TencentGR-1M/` 是原始公开数据。重构没有复制、重新下载或改写数据；缓存、权重和索引也只做同一文件系统内的目录改名。

## 旧名字为什么还存在

为了让历史 checkpoint、JSON、源码快照和已经分享的路径继续可读，保留以下相对符号链接：

| 旧入口 | 当前真实位置 |
|---|---|
| `artifacts/day1_3` | `artifacts/baseline` |
| `artifacts/day4_7` | `artifacts/objective_study` |
| `reports/day1_3` | `reports/baseline` |
| `reports/day4_7` | `reports/objective_study` |
| `configs/day1_3.json` | `configs/baseline.json` |
| `configs/day4_7.json` | `configs/objective_study.json` |
| `configs/day6_ann_metric.json` | `configs/ann_geometry.json` |
| `WEEK_PROJECT.md` | `PROJECT_OVERVIEW.md` |
| `PROJECT_GUIDE.md` | `BASELINE_GUIDE.md` |

这些链接不重复占用一份数据。日常阅读和新命令使用右侧名字。

原 `run_day1_3.py` 与 `run_day4_7.py` 的源码字节保留，语义入口调用它们并传入新的默认配置和输出路径，避免无必要改写已经完成的实验程序。

## 历史证据与新实验

已有 `plan.json`、checkpoint 中的配置、`source_snapshot/` 和已保存的原报告内容保持其记录的历史语境。报告正文或 JSON 里仍有旧日期目录名称，是当时运行的证据，不是当前推荐的入口。

旧路径保证历史产物可读取，不承诺旧冻结计划在原目录直接继续运行。`Path.resolve()` 会解析到新目录，代码继续开发也会改变 SHA；原计划的完整性检查可能按设计拒绝复用。需要复现时使用新的输出 `--root`，不要改旧计划来绕过检查。

例如：

```bash
.project_venv/bin/python scripts/run_baseline.py \
  --root artifacts/baseline_reproduction \
  --report-dir reports/baseline_reproduction --gpus 0,2

.project_venv/bin/python scripts/run_objective_study.py \
  --root artifacts/objective_study_reproduction \
  --baseline-root artifacts/baseline_reproduction \
  --report-dir reports/objective_study_reproduction --gpus 0,2,4
```

完整代码讲解留在根目录，是为了保持几千行文档内的源码相对链接简单、稳定。入口统一放在[文档导航](index.md)中。
