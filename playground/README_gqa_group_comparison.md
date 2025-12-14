# Qwen2 GQA Group Size Comparison

这个修改版本的实验代码允许你比较不同group大小对KV cache并集的影响。

## 新增功能

### 1. 自定义Group大小实验
可以指定自定义的group大小（每组head数量）来运行实验：

```bash
# 使用默认的模型配置（每组7个head）
python qwen2_gqa_experiment.py

# 使用自定义group大小（例如每组14个head）
python qwen2_gqa_experiment.py --group-size 14

# 使用每组1个head（等于没有分组）
python qwen2_gqa_experiment.py --group-size 1
```

### 2. Group大小比较实验
比较多个不同group大小的效果：

```bash
# 比较不同group大小的效果（1, 2, 4, 7, 14, 28个head/组）
python qwen2_gqa_experiment.py --compare-groups
```

## 实验配置

- **总Query头数**: 28（Qwen2-7B模型配置）
- **测试的Group大小**: [1, 2, 4, 7, 14, 28] (28的因子)
- **每个head保留token比例**: 20%
- **测试序列长度**: 2048（在group比较模式下）

## 输出结果

### 1. 单个实验输出
- 不同序列长度下的并集分析
- 实际vs理论并集大小比较
- 可视化图表：`qwen2_gqa_real_analysis.png`

### 2. Group比较实验输出
- 不同group大小的详细比较表格
- 关键发现和趋势分析
- 可视化图表：`qwen2_gqa_group_comparison.png`

## 关键指标

- **并集占比**: 并集大小相对于总序列长度的比例
- **理论占比**: 基于独立假设的理论期望值
- **膨胀倍数**: 相对于单个head(20%)的膨胀倍数
- **实际/理论比值**: 实际结果与理论期望的比值

## 预期发现

理论上，随着group大小增加：
- **Group大小1**: 并集占比 = 20%（最优，无重叠）
- **Group大小28**: 并集占比 ≈ 99.7%（最差，几乎全部token）
- **中间值**: 随着group大小增加，并集占比逐渐增大

实际结果可能会因attention权重的相关性而偏离理论值。

## 使用场景

1. **优化GQA配置**: 找到最适合的group大小配置
2. **内存使用分析**: 评估不同配置下的KV cache内存需求
3. **性能权衡**: 平衡计算效率和内存使用

## 注意事项

- 实验需要大量GPU内存和计算时间
- Group比较实验只在序列长度2048上运行以节省时间
- 确保模型路径 `/share/models/Qwen2-7B` 正确可访问
