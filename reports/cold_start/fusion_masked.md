# 探索性修正：基线与 ID-mask70 的保护式候选融合

**这是观察到文本融合失败、ID-mask70 单模型存在冷/尾取舍之后提出的验证驱动 amendment。配额、选择和区间重复使用验证用户，不属于独立确认。** 没有读取测试目标。

## 固定规则与身份检查

- 两路均为相同训练种子的 learned Top-500：原零遮蔽模型与 ID-mask70 模型，严格核对用户行号、目标、全部历史、缓存/队列签名和 checkpoint SHA。
- 每人仍为500候选；保护原Top-10及所有频次≤5的原有候选，只追加 mask70 通道中频次≤5、且不在整个原Top-500中的新广告。
- 只从基线尾部的非稀有候选移除等量广告，保留其他基线顺序。两路内积分数属于不同训练模型，不直接相加。
- 配额固定为0/10/25/50：总体Recall@500至少保留95%，冷启动和长尾各自不退步，再最大化两组等权平均，平分选更小配额。
- 原Top-10逐项不变；本次仅分析候选池，不声称Top-10改善，也未训练排序。第二模型/索引的在线代价尚未测试。

所选配额：**25**。

| 配额 | 总体 Recall@500 | 冷启动 | 长尾1–5 | 中频6–50 | 头部>50 | 平均替换数 | 通过约束 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 23.0600% | 8.3966% | 0.7290% | 6.8555% | 40.3578% | 0.00 | 是 |
| 10 | 22.7867% | 8.8518% | 0.8296% | 6.7318% | 39.8033% | 9.99 | 是 |
| 25 | 22.4167% | 9.2564% | 0.9301% | 6.5183% | 39.1167% | 24.50 | 是 |
| 50 | 21.7000% | 9.5599% | 1.2569% | 6.1812% | 37.7699% | 46.46 | 否 |

## 所选固定500与基线的配对差值

区间按用户与三个训练种子配对重采样，仅为重复使用验证集后的开发性描述。

| 分组 | Recall@500差值 | 用户与种子配对95%区间 |
|---|---:|---|
| all | -0.006433 | [-0.008300, -0.004500] |
| cold_0 | +0.008599 | [+0.003035, +0.016186] |
| tail_1_5 | +0.002011 | [+0.000503, +0.004022] |
| mid_6_50 | -0.003372 | [-0.005170, -0.001686] |
| head_gt50 | -0.012412 | [-0.015515, -0.009375] |

## 来源

- [原固定模型诊断](diagnostics.md)、[首次融合](fusion.md)、[保护式文本融合](fusion_protected.md)。
- [先登记的 amendment](../../artifacts/cold_start/masked_fusion_plan.json)。
- [实现](../../taac_retrieval/masked_candidate_fusion.py)、[复用的保护合并](../../taac_retrieval/protected_fusion.py)。
- [完整指标](../../artifacts/cold_start/fusion_masked/metrics.json)、[选择](../../artifacts/cold_start/fusion_masked/selection.json)。
- 所选候选保存在 `artifacts/cold_start/fusion_masked/selected_seed*.npz`；原有模型、索引和前两次融合结果均保留。

```bash
.project_venv/bin/python -m taac_retrieval.masked_candidate_fusion \
  --output artifacts/cold_start/fusion_masked_reproduction \
  --report reports/cold_start/fusion_masked_reproduction.md
```
