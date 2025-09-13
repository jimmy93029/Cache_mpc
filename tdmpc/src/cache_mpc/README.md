# TD-MPC 轨迹重叠分析工具使用说明

## 概述

这套工具用于分析TD-MPC规划器生成的精英轨迹之间的重叠情况，验证轨迹重用的可能性和效果。

## 文件说明

### 1. 修改后的 `TDMPC.plan()` 方法
- 添加了 `test_mode`, `t`, `test_dir` 参数
- 在测试模式下保存每个时间步的精英轨迹到CSV文件
- CSV文件包含：时间步动作序列、轨迹价值、轨迹得分

### 2. `test.py` - 主测试脚本
- 加载训练好的模型
- 运行多个测试episodes
- 为每个时间步保存规划的动作轨迹
- 创建结构化的测试目录

### 3. `overlap_analysis.py` - 重叠分析脚本
- 分析轨迹相似性和重叠模式
- 生成多种可视化图表
- 提供定量的重叠统计

## 使用步骤

### 步骤1：运行测试
```bash
python test.py
```

这将：
- 创建 `test/{task}/{exp_name}/{seed}/{test_number}/` 目录
- 运行测试episodes
- 保存每个时间步的规划轨迹

### 步骤2：分析重叠
```python
# 修改 overlap_analysis.py 中的路径设置
test_dir = Path("test/your_task/your_exp/your_seed/1")
horizon = 10  # 你的规划地平线
action_dim = 4  # 你的动作维度

# 运行分析
python overlap_analysis.py
```

## 生成的图表类型

### 1. 轨迹相似性热图 (Similarity Heatmap)
- **目的**: 显示同一时间步内精英轨迹之间的相似性
- **用途**: 识别高度相似的轨迹组
- **指标**: 余弦相似性矩阵
- **解读**: 颜色越亮表示轨迹越相似

### 2. 时序重叠趋势图 (Temporal Overlap Trends)
- **目的**: 分析连续时间步之间的轨迹重叠情况
- **用途**: 展示重叠比例如何随时间变化
- **指标**: 重叠百分比和平均相似性
- **解读**: 高重叠表明轨迹重用的潜力

### 3. PCA聚类可视化 (PCA Clusters)
- **目的**: 在二维空间中可视化高维轨迹的聚类情况
- **用途**: 识别轨迹的聚类模式和分布
- **方法**: 主成分分析降维
- **解读**: 聚集的点表示相似的轨迹

### 4. 价值-得分关系分析 (Value-Score Analysis)
- **目的**: 分析轨迹价值与选择概率的关系
- **用途**: 验证评分机制的合理性
- **包含**: 散点图、分布图、相关性分析
- **解读**: 高相关性表明评分与价值一致

## 重叠分析的关键指标

### 1. 相似性阈值
- **默认**: 0.95 (95%相似性)
- **作用**: 判断两个轨迹是否"重叠"
- **调整**: 可根据需要修改阈值

### 2. 重叠百分比
- **计算**: 超过阈值的轨迹对数量 / 总轨迹对数量
- **意义**: 表示可重用轨迹的比例

### 3. 平均相似性
- **计算**: 所有轨迹对相似性的平均值
- **意义**: 整体轨迹多样性的度量

## 实验分析建议

### 分析重叠模式
1. **时序连续性**: 观察t→t+1时间步的重叠情况
2. **价值保持**: 检查重叠轨迹的价值分布
3. **聚类稳定性**: 分析轨迹聚类的时序稳定性

### 验证重用效果
1. **性能对比**: 比较使用重叠轨迹vs新生成轨迹的性能
2. **计算效率**: 测量轨迹重用带来的计算节省
3. **鲁棒性**: 验证重用策略在不同环境下的表现

### 优化策略
1. **缓存机制**: 基于重叠分析设计轨迹缓存
2. **重用阈值**: 确定最优的相似性阈值
3. **混合策略**: 结合重用和新生成的混合方法

## 配置说明

### 测试配置 (test.py)
```python
# 可调整的参数
num_test_episodes = 5     # 测试episode数量
similarity_threshold = 0.95  # 相似性阈值
horizon = cfg.horizon     # 规划地平线
action_dim = cfg.action_dim  # 动作维度
```

### 分析配置 (overlap_analysis.py)
```python
# 可调整的参数
time_steps_to_analyze = 5  # 分析的时间步数量
pca_components = 2         # PCA降维维度
similarity_method = 'cosine'  # 相似性计算方法
```

## 输出文件结构

```
test/{task}/{exp_name}/{seed}/{test_number}/
├── episode_1/
│   ├── planned-actions-time0.csv
│   ├── planned-actions-time1.csv
│   └── ...
├── episode_2/
│   └── ...
├── test_summary.txt
└── analysis_plots/
    ├── similarity_heatmap_ep1_t0.png
    ├── temporal_overlap_ep1.png
    ├── pca_clusters_ep1.png
    └── value_score_analysis_ep1_t0.png
```

## 注意事项

1. **内存使用**: 大horizon和多episode会产生大量数据
2. **计算时间**: 相似性计算可能较耗时
3. **参数调整**: 根据具体环境调整相似性阈值
4. **模型路径**: 确保test.py中的模型路径正确