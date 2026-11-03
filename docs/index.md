# 文档导航

项目按研究问题和工程模块组织，不再按完成日期组织。

- [项目概览与运行命令](../PROJECT_OVERVIEW.md)：架构、已完成实验、环境和推荐入口。
- [从原始数据到推荐的完整代码讲解](../PROJECT_FULL_WALKTHROUGH.md)：数据格式、ID、模型、损失、检索、实际样例。
- [基线与数据管线入门](../BASELINE_GUIDE.md)：最初的数据适配和行为分支对照。
- [目录结构与旧路径兼容](directory_layout.md)：语义目录、复现实验和历史记录的边界。
- [目标函数与检索实验报告](../reports/objective_study/README.md)：扩大复核、种子、分桶、ANN 和失败案例。
- [InfoNCE 监督范围消融](../reports/infonce_supervision/README.md)：全部事件与仅点击监督的配对比较。
- [冷启动与长尾研究](../reports/cold_start/README.md)：验证集候选覆盖、广告 ID 遮蔽、独立文本召回与固定预算融合。
- [冷启动代码讲解](../PROJECT_FULL_WALKTHROUGH.md#s21)：上述实验的输入、控制变量、选择规则和边界。
- [Transformer 层数实验协议](depth_study.md)：固定其余设置，比较 1／2／4 层并核对嵌套初值。
- [层数对照报告](../reports/depth_study/README.md)：三种子验证指标、冷启动／长尾覆盖、参数与训练成本。

原开源 baseline 说明保留为[中文版](upstream_baseline_zh.md)和[英文版](upstream_baseline_en.md)，用于核对根目录旧训练/推理入口。
