# 分支与实验范围

`main` 包含完整召回与排序链路。研究分支保留实验源码与各自的指标口径。

| 分支 | 内容 | 结论的边界 |
|---|---|---|
| `main` | 一层点击 InfoNCE-64、精确评估、MIPS→L2 Faiss，以及历史查组、DCN V2 列表排序与单用户演示 | 三种子召回测试和两种子排序确认各自记录 |
| `research/channel-aware-ranking` | 列表排序的原始冻结实验与完整研究历史 | 6,000 人确认，使用验证点击 |
| `research/tail-recall` | 零点击长尾 ID 置零通道 | 另一种历史候选补充方法 |
| `research/generative-group-recall`、`research/generative-retrieval` | 属性路径生成与语义 ID 探索 | 见[生成式探索](generative_retrieval.md) |
| `research/supervision` | 全部有效事件 InfoNCE-64 对比仅点击 InfoNCE-64 | 后续补充研究；不能倒推为原始方案选择依据 |
| `research/cold-start` | 多 K 分桶诊断、广告 ID 随机遮挡、内容召回和候选融合 | 观察到候选覆盖的收益与总体覆盖代价；不代表 Top-10 问题已解决 |
| `research/depth` | 一层、两层、四层的容量对照与共享初始化 | 验证集探索；尚无独立测试确认可替代主方案 |
| `research/full-study` | 分支整理前的完整工作快照，包含所有探索及原有文档 | 用于保全历史，不作为默认运行入口 |

## 如何切换

先用 `git status` 检查自己的修改。查看某个分支的文件而不切换工作目录，可以用：

```bash
git show research/depth:docs/depth_study.md
git log --oneline --all --decorate -12
```

需要单独研究环境时，新建 worktree：

```bash
git worktree add ../baseline_2025_depth research/depth
```

原始数据、虚拟环境、缓存、checkpoint 和 Faiss 索引未被提交，不随分支复制。对照时需显式指向正确的数据与输出目录，避免把一份模型的导出向量与另一份索引混用。

## 主方案的指标归属

召回准确率来自三种子测试，索引测量固定 seed=2026。排序准确率来自 6,000 人、两种子的独立留出验证队列。各自的数据规模与候选预算见[版本说明](release.md)。
