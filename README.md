# 监督范围对照：全部事件与仅点击 InfoNCE-64

保持模型结构和64负例预算，比较曝光+点击监督与仅点击监督；该补充实验未形成稳定优于主版本的结论。

稳定展示版本在 [main](https://github.com/J-OneK/baseline_2025/tree/main)，完整原始研究快照在 [research/full-study](https://github.com/J-OneK/baseline_2025/tree/research/full-study)。本分支从稳定主版本派生，保留与本主题相关的实现、协议、测试和历史实测报告。

- [主题实验报告](reports/infonce_supervision/README.md)
- [实验入口](scripts/run_infonce_supervision.py)：先运行 `python scripts/run_infonce_supervision.py --help` 查看阶段和产物路径。
- [主链路说明](docs/code_walkthrough.md)与[安装/运行说明](README_CN.md)。

历史报告的源文件哈希、绝对路径和实验计划属于当时的研究状态。当前分支做了入口与依赖隔离；复现请使用新的输出目录，不能覆盖历史产物或把旧计划哈希当作当前源码哈希。数据、权重、缓存和索引按主分支规则保留在本地。
