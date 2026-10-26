# Pretrain Sanity Check 总报告

**更新日期:** 2026-03-30  
**定位:** 这是 `Pretrain_Sanity check` 的唯一总入口文档。  
**覆盖范围:**  

- Step 0 baseline 建立与验证
- Fusion 系列实验演进与最终结论
- LiDAR-guided distillation 的设计、teacher 选择、warm-start 决策与当前进展

---

## 1. 总体结论

当前 `Pretrain Sanity Check` 的结论已经明确：

1. `Step 0` baseline 已冻结且可信
2. `Fusion` 系列已经找到明确的 **overall best** 与 **ROI best**
3. `Distillation` 方向具备继续推进的前提条件，但当前仍处于实验验证阶段

当前最重要的三个锚点是：

- **Baseline:** `centerpoint_radar_baseline @ ep5`
  - Entire Area mAP: `31.96`
  - ROI mAP: `53.59`
- **Best Fusion Student (Overall / Entire-first):** `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior @ ep8`
  - Entire Area mAP: `43.55`
  - ROI mAP: `64.59`
- **Best Fusion Student (ROI-first):** `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight @ ep11`
  - Entire Area mAP: `42.80`
  - ROI mAP: `65.75`
- **Best LiDAR Teacher:** `centerpoint_lidar_teacher @ ep11`
  - Entire Area mAP: `43.81`
  - ROI mAP: `60.78`

---

## 2. Step 0：Baseline 建立

### 2.1 冻结的 baseline

- 模型：`centerpoint_radar_baseline`
- 最优 checkpoint：`ep5-centerpoint_radar_baseline.ckpt`
- 正式指标：
  - Entire Area mAP: `31.96`
  - ROI mAP: `53.59`
  - Car: `23.97`
  - Pedestrian: `20.67`
  - Cyclist: `51.25`

### 2.2 数据层面的关键发现

- Radar 点云非常稀疏：
  - Train: `355.65` pts/frame
  - Val: `330.90` pts/frame
- GT box 内 radar 点比例极低：
  - Train: `0.20%`
  - Val: `0.28%`
- 非空 pillar 比例仅约：
  - Train: `0.75%`
  - Val: `0.71%`

这意味着：

- 模型不能只依赖 box 内点
- 必须依赖上下文和更大范围的结构信息
- Pedestrian 检测天然困难

### 2.3 训练层面的关键发现

- baseline 训练整体健康
- heatmap / bbox loss 无 NaN、无异常
- 模型在 `ep2` 就已达到：
  - Entire Area mAP: `28.92`
- 从 `ep5` 后开始出现明显过拟合：
  - `ep5`: `31.96`
  - `ep8`: `28.58`
  - `ep11`: `26.00`

结论：

- `ep5` 是合理最佳点
- baseline 可以作为所有后续实验的统一参考

### 2.4 Step 0 的作用

Step 0 已经回答了后续所有实验必须先知道的问题：

- baseline 是什么
- 数据有多稀疏
- 哪个类别最难
- 训练在哪个阶段开始过拟合
- 哪些方向最值得继续做

---

## 3. Fusion：从能融合到最优 student

### 3.1 实验演进

Fusion 系列一共经历了 5 个清晰版本：

1. `fusion_freeze`
2. `centerpoint_radar_gated_fusion_v2`
3. `fusion_finetune_layer3`
4. `fusion_finetune_layer3_stable`
5. `fusion_finetune_layer3_stable_radar_to_lidar`

这 5 版对应解决的核心问题分别是：

1. 图像加进来有没有用
2. 更好的融合结构有没有额外收益
3. 图像 backbone 一起学能不能提高峰值
4. 如何把这个峰值变成稳定收益
5. 坐标系统一是否还能继续拉高上限

### 3.2 各版本正式结果

| 排名 | 模型 | 最优 checkpoint | Entire mAP | ROI mAP |
|------|------|-----------------|------------|---------|
| 1 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior` | ep8 | **43.55** | 64.59 |
| 2 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_geom` | ep8 | 43.34 | 64.35 |
| 3 | `fusion_finetune_layer3_stable_radar_to_lidar_big3` | ep5 | 43.16 | 64.78 |
| 4 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_heatmap_only` | ep8 | 42.96 | 64.84 |
| 5 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight` | ep11 | 42.80 | **65.75** |
| 6 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior_ped_only` | ep8 | 42.59 | 64.64 |
| 7 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_bbox_only` | ep11 | 42.37 | 65.09 |
| 8 | `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres` | ep11 | 41.79 | 64.59 |
| 9 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_ped_car` | ep8 | 41.72 | 64.22 |
| 10 | `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_radar_attn` | ep11 | 41.55 | 63.55 |
| 11 | `fusion_finetune_layer3_stable_radar_to_lidar_main_struct_radar_changes` | ep8 | 41.34 | 64.18 |
| 12 | `fusion_finetune_layer3_stable_radar_to_lidar_main_struct` | ep11 | 39.05 | 62.84 |
| 13 | `fusion_finetune_layer3_stable_radar_to_lidar` | ep5 | 37.55 | 61.91 |
| 14 | `fusion_finetune_layer3_stable` | ep5 | 35.37 | 58.71 |
| 15 | `fusion_finetune_layer3` | ep2 | 34.35 | 57.33 |
| 16 | `centerpoint_radar_gated_fusion_v2` | ep11-v2 | 33.82 | 57.37 |
| 17 | `fusion_freeze` | ep5 | 32.59 | 54.67 |
| 18 | `centerpoint_radar_baseline` | ep5 | 31.96 | 53.59 |

### 3.3 Fusion 最重要的三个结论

**结论 1：图像分支是有价值的**

- `fusion_freeze` 已经证明图像融合有效
- 但首版增益有限，说明“有图像”不等于“图像被正确利用”

**结论 2：可学习融合结构比简单拼接更重要**

- `gated_fusion_v2` 明显优于早期简单方案
- gate 统计说明图像特征是被选择性注入的，而不是盲目叠加

**结论 3：坐标系统一是决定上限的关键因素**

- `stable_radar_to_lidar` 相比 `stable` 有显著提升
- 说明几何参考系一致性不是细节问题，而是结构性收益来源

### 3.4 当前最佳 student

当前所有 Fusion 实验已经需要区分两个“最佳”：

- **Overall best (Entire-first):** `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior @ ep8`
  - Entire Area mAP: `43.55`
  - ROI mAP: `64.59`
- **ROI best (ROI-first):** `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight @ ep11`
  - Entire Area mAP: `42.80`
  - ROI mAP: `65.75`

相对 baseline：

- Overall best:
  - Entire `+11.59`
  - ROI `+11.00`
- ROI best:
  - Entire `+10.84`
  - ROI `+12.16`

因此后续如果目标是：

- **整体性能**：以 `big3_roi_train_weight_camera_prior @ ep8` 为 student 起点
- **ROI 性能**：以 `big3_roi_train_weight @ ep11` 为 student 起点

### 3.4.1 结构迁移实验：引入 `main` 中的 radar 结构改动

在不改变当前 fusion 主线的前提下，额外把 `main` 分支中一组 radar-only 结构改动迁移到：

- `fusion_finetune_layer3_stable_radar_to_lidar`

形成实验：

- `fusion_finetune_layer3_stable_radar_to_lidar_main_struct`

这次迁移的改动核心包括：

- 更深的 voxel encoder 配置
- `avg` pooling
- `with_distance=true`
- 更深的 shared conv
- class-wise `min_radius`
- 调整后的 `code_weights`

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 36.17 | 57.78 | 23.68 | 25.56 | 59.26 | 59.49 | 33.28 | 80.58 |
| `ep5` | 36.06 | 58.16 | 24.23 | 22.63 | 61.31 | 57.79 | 30.73 | 85.96 |
| `ep8` | 37.23 | 61.44 | 29.78 | 22.30 | 59.60 | 67.78 | 29.86 | 86.69 |
| `ep11` | **39.05** | **62.84** | 28.02 | 26.02 | **63.11** | 67.49 | **34.98** | 86.04 |

相对旧 best student `fusion_finetune_layer3_stable_radar_to_lidar @ ep5 = 37.55 / 61.91`：

- Entire `+1.49`
- ROI `+0.93`

阶段性结论：

1. `main` 中那组结构改动并不是对 fusion 路线必然负迁移
2. 它们在前中期表现并不稳定，但在后期 `ep11` 反而超过旧 best student
3. 这说明当前最优 fusion 路线仍有继续通过 radar-side 结构微调获益的空间
4. 当前最优结果已经从旧的 `radar_to_lidar @ ep5` 更新为：
   - **`main_struct @ ep11`**

### 3.4.2 叠加 `Radar changes`：3-frame radar + temporal weighting + RCS Top-K

在 `main_struct` 的基础上，进一步叠加了 `src/Radar changes/` 中的输入侧增强内容，形成实验：

- `fusion_finetune_layer3_stable_radar_to_lidar_main_struct_radar_changes`

这次叠加的关键改动包括：

- 使用 `radar_3frames`
- 对旧帧数据的 `RCS / v_r / v_r_compensated` 进行时间衰减
- 对时间特征做 recency remap
- voxel 内先收集更多点，再按 `RCS` 保留 Top-K

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 36.41 | 54.43 | 27.65 | 24.68 | 56.91 | 59.11 | 30.44 | 73.74 |
| `ep5` | 39.95 | 63.42 | 30.50 | 29.52 | 59.82 | 68.50 | 36.48 | 85.29 |
| `ep8` | **41.34** | **64.18** | 30.92 | 28.11 | 65.00 | 68.90 | 35.68 | **87.96** |
| `ep11` | 41.57 | 60.51 | 30.25 | 27.75 | **66.71** | 59.94 | 33.99 | 87.60 |

相对上一版 best student `main_struct @ ep11 = 39.05 / 62.84`：

- Entire `+2.29`
- ROI `+1.34`

相对旧的 `stable_radar_to_lidar @ ep5 = 37.55 / 61.91`：

- Entire `+3.79`
- ROI `+2.27`

阶段性结论：

1. `Radar changes` 在 fusion 主线上不是中性改动，而是明确有效的输入侧增强
2. 与单纯 `main_struct` 相比，3-frame radar + temporal weighting + RCS Top-K 可以继续把 fusion student 拉高
3. `ep11` 的 Entire 更高，但 ROI 明显回落，说明这条线路仍然存在后期偏置重分配问题
4. 因此当前最合理的归档结论是：
   - **`main_struct_radar_changes @ ep8`**

### 3.4.3 `Pedestrian` 定向改进：引入高分辨率图像特征

基于分类别分析，当前代码进一步尝试在保持 `main_struct_radar_changes` 主线不变的前提下，通过更高分辨率视觉特征去补 `Pedestrian`。形成实验：

- `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres`

这次改动的核心是：

- 图像 backbone 从单一 `layer3` 扩展为 `layer2 + layer3`
- 对每个 radar point 同时从 `layer2` 和 `layer3` 采样图像特征
- 在 point 级别拼接双尺度 image feature 后，再送入现有 `PointGatedFusion`

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 40.65 | 62.27 | 30.61 | 29.47 | 61.86 | 67.94 | 36.30 | 82.57 |
| `ep5` | 40.65 | 63.12 | 30.80 | 27.50 | 63.65 | 70.44 | 35.70 | 83.21 |
| `ep8` | 40.47 | 62.19 | 29.71 | 29.86 | 61.86 | 61.22 | 36.58 | **88.77** |
| `ep11` | **41.79** | **64.59** | 29.92 | **30.44** | **65.01** | 69.27 | **37.33** | 87.18 |

相对当前 overall best student `main_struct_radar_changes @ ep8 = 41.34 / 64.18`：

- `ep11`
  - Entire `+0.45`
  - ROI `+0.41`

相对上一版 best 在 `Pedestrian` 上的表现：

- `ep11`
  - Entire `+2.33`
  - ROI `+1.65`

相对 `Pedestrian` 指标本身，双尺度视觉特征确实带来了定向收益：

- 最佳 Entire Pedestrian：`30.44`（相对原 best `28.11`，`+2.33`）
- 最佳 ROI Pedestrian：`37.33`（相对原 best `35.68`，`+1.65`）

阶段性结论：

1. `Pedestrian` 定向增强方向被进一步验证是正确的，而且这次已经转化成新的 overall best
2. 与 `main_struct_radar_changes @ ep8` 相比，`Pedestrian` 的提升足以覆盖 `Car` 的小幅波动
3. `Cyclist` 的 Entire 基本持平且 ROI 仍保持高位，说明高分辨率视觉特征并没有破坏此前的 radar-side 增益主线
4. 因此当前最合理的归档判断是：
   - **新的 best fusion student = `ped_highres @ ep11`**
   - **这条线路已经不只是定向增强，而是新的整体最优模型**

### 3.4.4 `Pedestrian` class-wise supervision 微调

在 `ped_highres @ ep11` 成为当前 overall best 之后，进一步尝试直接加强 `Pedestrian` 的 supervision，形成实验：

- `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres_cls_tuned`

这次微调只改了两项：

- `train_cfg.min_radius = [4, 3, 2]`
- `train_cfg.task_loss_weights = [1.0, 1.25, 1.0]`

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 40.83 | 61.71 | 30.30 | 26.62 | 65.58 | 65.12 | 33.97 | 86.02 |
| `ep5` | 39.89 | 61.30 | 30.64 | 26.97 | 62.06 | 65.54 | 35.78 | 82.57 |
| `ep8` | **40.98** | **63.86** | **30.71** | 26.68 | 65.54 | **69.80** | 35.33 | **86.44** |
| `ep11` | 34.78 | 57.82 | 24.57 | 24.72 | 55.04 | 57.50 | 32.59 | 83.35 |

相对当前 best fusion student `ped_highres @ ep11 = 41.79 / 64.59`：

- `ep8`
  - Entire `-0.81`
  - ROI `-0.73`

相对 `ped_highres @ ep11` 在 `Pedestrian` 上的表现：

- Entire `30.44 -> 26.68`，`-3.76`
- ROI `37.33 -> 35.33`，`-2.00`

阶段性结论：

1. 当前这组 class-wise supervision 参数 **没有继续提升** 当前 best 模型
2. 更大的 `Pedestrian min_radius` 和更高 task weight 并没有把高分辨率视觉特征转化成额外收益，反而削弱了 `Pedestrian` 本身的表现
3. 这说明当前 `ped_highres` 已经处在一个较好的类别平衡点，继续用更强的 `Pedestrian` supervision 去拉，容易打破这种平衡
4. 因此当前最合理的归档判断是：
   - **保留 `ped_highres @ ep11` 作为 overall best**
   - **当前这组 `cls_tuned` 参数记为负结果，不继续沿原值扩大**

### 3.4.5 大改组合实验：point + BEV fusion、temporal radar encoder、LiDAR auxiliary geometry supervision

在 `ped_highres @ ep11` 成为当时 overall best 之后，进一步把三条更大的结构升级组合到同一个实验中，形成：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3`

这次组合实验包含：

- point-level fusion 之上的轻量 BEV-aware fusion
- voxelization 之后、voxel encoder 之前的显式 temporal radar encoding
- 训练期基于 LiDAR 的 BEV occupancy / density auxiliary supervision

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 40.32 | 59.78 | 29.20 | 27.13 | 64.63 | 59.65 | 33.39 | 86.29 |
| `ep5` | **43.16** | **64.78** | 29.74 | **31.77** | **67.96** | 65.55 | **39.15** | **89.64** |
| `ep8` | 39.41 | 63.18 | 30.56 | 28.40 | 59.28 | 67.40 | 36.49 | 85.65 |
| `ep11` | 41.24 | 64.39 | 30.47 | 27.84 | 65.40 | 68.47 | 37.56 | 87.14 |

相对上一版 best fusion student `ped_highres @ ep11 = 41.79 / 64.59`：

- `ep5`
  - Entire `+1.37`
  - ROI `+0.19`

分类别变化也很清楚：

- Entire
  - Car `-0.18`
  - Pedestrian `+1.33`
  - Cyclist `+2.95`
- ROI
  - Car `-3.72`
  - Pedestrian `+1.82`
  - Cyclist `+2.46`

阶段性结论：

1. 这组大改不是“只增加复杂度”，而是确实把总体表现继续推高了
2. 最大增益来自 `Pedestrian` 与 `Cyclist` 的同步提升，说明：
   - 更强的时序建模
   - 更强的 BEV 融合
   - 更轻的 LiDAR geometry supervision
   这三条线在当前系统里是互补的
3. `Car` 的 ROI 有所回落，说明当前大改组合仍存在类别平衡问题，但并未阻止 overall mAP 创新高
4. 训练曲线仍然不是单调稳定，`ep8` 明显回落、`ep11` 又回升，说明这条大改路线后续仍值得做更细的 schedule / loss 权重调优
5. 因此当前最合理的归档判断是：
   - **新的 best fusion student = `big3 @ ep5`**

### 3.4.6 `big3_car_balance`：降低 geometry aux 权重后的 ROI 平衡修正

在 `big3 @ ep5` 成为当前 overall best 之后，进一步尝试只降低 geometry auxiliary supervision 的权重，形成实验：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_car_balance`

这次修正只改了一项：

- `geometry_aux.loss_weight: 0.2 -> 0.1`

当前已记录的关键 checkpoint 为：

| Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | 40.62 | 61.59 | 29.70 | 27.10 | 65.06 | 65.64 | 34.38 | 84.74 |
| `ep5` | 39.88 | 63.01 | 29.18 | 28.68 | 61.78 | 65.24 | 36.67 | 87.13 |
| `ep8` | 41.96 | 63.76 | **31.77** | 28.98 | 65.12 | 66.97 | 35.62 | **88.67** |
| `ep11` | 40.69 | 63.86 | 31.17 | 28.54 | 62.37 | **67.97** | **37.32** | 86.30 |

相对当前 ROI 最优模型 `big3 @ ep5 = 64.78`：

- `ep8`
  - ROI `-1.02`
- `ep11`
  - ROI `-0.92`

相对 `big3 @ ep5` 在 `Car ROI` 上的表现：

- `65.55 -> 67.97`，`+2.42`

阶段性结论：

1. 降低 geometry aux 权重后，`Car ROI` 的确明显回升
2. 但整体 ROI mAP 仍然没有超过 `big3 @ ep5`
3. 这说明：
   - `geometry_aux` 的强度确实会影响 `Car ROI`
   - 但当前大改组合的整体最优点，仍然不是简单把 geometry aux 权重往下压就能得到
4. 当前关于 ROI 的最合理判断是：
   - **全实验最高 ROI 仍然是 `big3 @ ep5 = 64.78`**
   - **若单看 `Car ROI`，`big3_car_balance @ ep11 = 67.97` 是当前更强的修正方向信号**

### 3.5 Fusion 调试框架与已完成验证

当前 `Fusion/` 目录下还保留了一组调试与运行文档，它们的关键信息也已经纳入当前总报告：

- `FUSION_DEBUG_GUIDE.md`
- `FUSION_DEBUG_QUICK_START.md`
- `FUSION_DEBUG_RESULTS.md`
- `FUSION_TRAINING_README.md`
- `Gated_Fusion_v2_Results_Summary.md`

这些文档的核心作用不是给出新的最终排名，而是回答：

- 融合实现是否在工程上跑通
- 调试时应该先看什么
- 哪些问题会直接导致训练无效
- 训练前是否已经满足最低质量门槛

### 3.6 已完成的 Fusion 调试结论

根据调试文档，融合流水线在正式训练前已经通过了以下检查：

1. **投影正确性**
   - 有效投影率约 `83.7%`
   - out-of-bounds 比例约 `16.3%`
   - 处于合理范围，说明 `radar -> camera -> image` 的投影链路基本正确

2. **数据一致性**
   - 同一样本重复加载，投影坐标方差 `std = 0.0000`
   - 说明调试场景下数据读入是稳定、可复现的

3. **特征维度检查**
   - Radar 点原始特征维度：`7`
   - 图像采样特征维度：`256`
   - 融合后点特征维度：`263`
   - 这说明 image feature sampling 与 feature concat 已经正确接入

4. **配置加载**
   - `fusion_freeze`
   - `fusion_finetune_layer3`
   - 两套配置都已成功加载

### 3.7 Fusion 调试阶段形成的工程规则

训练前最重要的检查项已经明确：

- 投影 OOB 不应高于 `50%`
- 数据增强不应破坏投影对齐
- 融合后特征维度必须从 `7 -> 263`
- 前几个 batch 的 loss 必须有限且有下降趋势

也就是说，Fusion 的工程工作不是“直接训练然后看 mAP”，而是先经过：

1. projection 检查
2. augmentation 对齐检查
3. shape 检查
4. loss 检查

通过这些前置验证后，后续训练结果才有可解释性。

### 3.8 Fusion 调试与训练操作流

`FUSION_DEBUG_GUIDE.md`、`FUSION_DEBUG_QUICK_START.md` 与 `FUSION_TRAINING_README.md` 除了给出结论，还定义了一套当时实际采用的工程执行顺序。为避免这些内容只留在分散文档里，这里统一收敛：

#### 3.8.1 调试脚本与用途

- 核心脚本：`src/tools/debug_fusion.py`
- 当时已支持的检查项共 5 个：
  1. `projection`
  2. `bounds`
  3. `augment`
  4. `shapes`
  5. `loss`

这些检查项分别回答：

- 雷达点能否正确投影到图像平面
- 投影越界比例是否处于合理范围
- 数据增强是否破坏投影对齐
- 融合后的点特征与 voxel 化形状是否正确
- 前几个 batch 的训练 loss 是否有限、稳定、无 NaN

#### 3.8.2 当时采用的标准执行顺序

1. **先做 projection 单项检查**
   - 命令：
     ```bash
     python src/tools/debug_fusion.py \
         --config-name fusion_freeze \
         --checks projection \
         --sample-idx 0
     ```
   - 目标：
     - 生成 `outputs/debug_projection_fusion_freeze.png`
     - 确认点不是随机散落
     - 确认 OOB 比例大致在 `10%-30%`

2. **再做所有快速检查**
   - 命令：
     ```bash
     python src/tools/debug_fusion.py \
         --config-name fusion_freeze \
         --checks projection bounds augment shapes
     ```
   - 目标：
     - 一次性确认投影、越界、增强对齐、shape 全部正常

3. **最后做 loss 检查**
   - 命令：
     ```bash
     python src/tools/debug_fusion.py \
         --config-name fusion_freeze \
         --checks loss
     ```
   - 目标：
     - 前 5 个 batch loss 有限
     - loss 总体向下
     - 无 NaN

4. **以上通过后才进入正式训练**
   - 先训 `fusion_freeze`
   - 再训 `fusion_finetune_layer3`
   - 最终逐步推进到 `stable` 与 `stable_radar_to_lidar`

#### 3.8.3 各检查项的判定标准

- **Projection / Bounds**
  - Good: OOB `10%-30%`
  - Warning: OOB `30%-50%`
  - Bad: OOB `>50%`
  - 若明显异常，优先检查：
    - `t_camera_radar`
    - `camera_projection_matrix`
    - 深度过滤条件 `depth > 0`

- **Augmentation Alignment**
  - 同一样本多次加载后的投影坐标方差应接近 0
  - 若增强后投影明显漂移，说明增强与几何变换没有同步更新

- **Shape Check**
  - 原始 radar 点特征维度应为 `7`
  - 图像采样特征维度应为 `256`
  - 融合后点特征维度应为 `263`
  - voxel 特征尺寸应与 `32 x 263` 这一预期兼容

- **Loss Check**
  - 前几个 batch 不应出现 NaN
  - loss 应至少保持有限值并整体下降
  - 若一开始就爆炸，应先回头查 projection、shape 和 config，而不是直接长时间训练

#### 3.8.4 当时的工程状态记录

这些文档还保留了 Fusion 刚完成实现时的工程状态，现统一记录如下：

- 已完成：
  - `ResNet18/layer3` 图像 backbone 接入
  - `t_camera_radar + camera_projection_matrix` 投影链路接入
  - 双线性插值的 point-wise image feature sampling
  - 图像特征与雷达点特征拼接
  - `fusion_freeze` / `fusion_finetune_layer3` 两套配置

- 当时定义的后续推进顺序：
  1. debug pipeline
  2. 训练 `fusion_freeze`
  3. 训练 `fusion_finetune_layer3`
  4. 与 baseline `31.96 mAP` 做比较

这一工程过程的重要性在于：Fusion 后续所有正式实验结果，都不是“盲训后再解释”，而是在一套固定的调试 SOP 通过后才开始产生的。

### 3.9 v2 的补充解释

`Gated_Fusion_v2_Results_Summary.md` 里补充了一些主报告之外更细的观察，这里一并记录：

- v2 相比 v1 的最大收益点是 **Pedestrian**
  - Entire `+5.53`
  - ROI `+6.43`
- v2 还有一个很明确的特点：
  - **far objects** 的 gate 更高
  - 说明图像对远距离稀疏目标更重要
- v2 的问题也很清楚：
  - Car 相对 v1 有小幅回落
  - 说明好的融合结构并不自动保证所有类别同时获益

这些补充信息支持了后续版本为什么要继续向：

- backbone 稳定化
- 坐标统一
- 更细的 teacher/student 对齐

这几个方向推进。

---

## 4. Distillation：Teacher 已确认，训练策略已调整

### 4.1 Distillation 的目标

当前蒸馏路线不是改推理结构，而是：

- 训练时用 LiDAR teacher 提供 privileged supervision
- 推理 / 测试时仍然只使用 fusion student

蒸馏设计为：

- `LiDAR-only frozen teacher`
- `single-scale BEV feature distillation`
- `GT box masked L1 loss`
- 蒸馏位置固定在 `neck`

### 4.2 LiDAR teacher 的实测结果

teacher checkpoint 经过单独 `eval.py` 复核，关键结果如下：

| Teacher Checkpoint | Entire Area mAP | ROI mAP | 备注 |
|------|------------|---------|------|
| `ep2` | `43.10` | **`65.00`** | ROI 更高 |
| `ep5` | `39.88` | `62.17` | 较弱 |
| `ep11` | **`43.81`** | `60.78` | Entire 主指标最高 |

当前 teacher 选择：

- **`centerpoint_lidar_teacher @ ep11`**

理由：

- 当前 checkpoint 监控逻辑按 `validation/entire_area/mAP`
- `ep11` 的 Entire Area mAP 最高

### 4.3 已经发现的问题

在 distill 推进过程中，已经确认几个关键工程问题：

1. 没有 LiDAR-only teacher 时，蒸馏路线本身无法成立
2. 直接用 `checkpoint_path=best_student.ckpt` 会恢复：
   - model weights
   - optimizer state
   - epoch / scheduler state
3. 这种“完整续训”方式不适合在旧任务 `L_det` 的基础上突然切到新任务 `L_det + lambda * L_distill`

### 4.4 当前推荐方案：warm start only

现在训练入口已经支持：

- `checkpoint_path`
  - 恢复完整训练状态
- `pretrained_student_checkpoint_path`
  - 只 warm-start student 权重
  - 不恢复 optimizer state
  - 不恢复 epoch state

对 distillation，推荐方案是：

- teacher:
  - `ep11-centerpoint_lidar_teacher.ckpt`
- student warm start:
  - `ep5-centerpoint_radar_gated_fusion_finetune_layer3_stable_radar_to_lidar-v1.ckpt`

也就是说，下一轮正式 distill 应当：

- 只加载 best student 权重
- 从新的 optimizer 状态开始训练
- 不再使用完整 resume 的方式切换 loss

### 4.5 当前 distill 的中间观察

截至 2026-03-23，已经完成了一次 **resume full state** 版本的 distill 训练。日志中明确出现：

- `Restored all states from the checkpoint ...`

这意味着本次实验使用的仍然是：

- `checkpoint_path=best_student.ckpt`

因此它属于“完整恢复训练状态再切换到 distillation loss”，而**不是**“warm-start only”实验。

当前已记录的关键 checkpoint 如下：

| Distill Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep8` | **`38.84`** | **`61.38`** | 29.38 | 27.72 | **59.43** | 63.68 | **36.72** | **83.74** |
| `ep11` | `35.60` | `60.18` | 28.20 | 22.53 | 56.08 | **65.96** | 32.15 | 82.43 |

相对当时的 best fusion student `37.55 / 61.91`：

- `ep8`
  - Entire `+1.29`
  - ROI `-0.53`
- `ep11`
  - Entire `-1.95`
  - ROI `-1.73`

阶段性解读：

1. **LiDAR supervision 不是完全无效**
   - `ep8` 在 Entire Area mAP 上已经超过当前 best student
2. **收益并不稳定**
   - 到 `ep11` 出现明显回落，说明这版训练设置存在后期退化
3. **收益形态更像 trade-off，而不是全面提升**
   - Entire 改善
   - ROI 略降
   - Cyclist 增益显著
   - Car / Pedestrian 没有同步受益
4. **当前结果仍受“旧 optimizer 状态 + 新 loss”干扰**
   - 因为这次不是 warm-start only
   - 所以还不能把这次结果当成最终蒸馏结论

当前更合理的阶段性判断是：

- **best resume-distill checkpoint = `ep8`**
- 这证明蒸馏方向具备潜在收益
- 但还没有证明“warm-start only 版本也能稳定复现并超过 best student”

因此当前策略保持不变：

- 继续把 **warm-start only** 作为正式结论实验的推荐路径。

### 4.6 warm-start only 实测结果

截至 2026-03-24，已经开始用：

- `pretrained_student_checkpoint_path=best_student.ckpt`

来运行正式的 warm-start only distillation。按这条命令的语义，本轮实验：

- 只加载 best student 权重
- 不恢复 optimizer state
- 不恢复 epoch / scheduler / trainer state

当前已记录的 warm-start only checkpoint 如下：

| Distill Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | **`38.61`** | **`61.86`** | **30.45** | **27.66** | **57.72** | **68.33** | **33.48** | **83.78** |
| `ep5` | `35.51` | `60.43` | 25.94 | 22.98 | 57.60 | 65.66 | 33.11 | 82.52 |

相对当时的 best fusion student `37.55 / 61.91`：

- `ep2`
  - Entire `+1.06`
  - ROI `-0.05`
- `ep5`
  - Entire `-2.04`
  - ROI `-1.48`

相对 best resume-distill checkpoint `38.84 / 61.38`：

- `warm-start ep2`
  - Entire 略低 `-0.23`
  - ROI 更高 `+0.48`

当前阶段可得出的判断是：

1. **warm-start only 也确实能带来正收益**
   - `ep2` 已经在 Entire Area mAP 上超过 best student
2. **warm-start only 比完整 resume 更干净**
   - 现在至少可以排除“旧 optimizer 状态”这个主要干扰项
3. **但收益仍然不稳定**
   - 到 `ep5` 已经明显回落
4. **最值得保留的现象是：**
   - `ep2` 几乎不损失 ROI
   - Entire 有正增益
   - Car 与 Cyclist 都有提升
   - Pedestrian 仍然偏弱

目前若只看已完成的 warm-start only checkpoint：

- **best warm-start-distill checkpoint = `ep2`**

这使得当前结论比前一天更明确：

- LiDAR-guided distillation **不是无效方向**
- 它在 warm-start only 设定下曾经超过旧 best student
- 但是否能稳定保持到更后期 epoch，仍然需要继续验证

### 4.7 基于上一版 best student (`main_struct_radar_changes @ ep8`) 的 warm-start distill 结果

截至 2026-03-28，已经进一步基于当时的 best fusion student：

- `fusion_finetune_layer3_stable_radar_to_lidar_main_struct_radar_changes @ ep8`
- `41.34 / 64.18`

继续进行新一轮 warm-start only distillation。当前已记录的 checkpoint 如下：

| Distill Checkpoint | Entire Area mAP | ROI mAP | Car | Pedestrian | Cyclist | ROI Car | ROI Pedestrian | ROI Cyclist |
|------|------------|---------|------|-------------|---------|---------|----------------|-------------|
| `ep2` | `36.00` | `60.31` | 28.52 | 23.11 | 56.37 | 64.45 | 33.79 | 82.70 |
| `ep5` | **`38.62`** | **`61.89`** | **29.86** | 25.49 | **60.52** | **67.15** | 33.84 | **84.67** |
| `ep8` | `37.80` | `61.71` | 29.14 | 24.15 | 60.10 | 66.48 | 34.41 | 84.25 |
| `ep11` | `37.16` | `60.12` | 27.84 | **26.24** | 57.41 | 67.35 | **35.71** | 77.30 |

相对当时的 best fusion student `41.34 / 64.18`：

- `ep5`
  - Entire `-2.72`
  - ROI `-2.29`
- `ep8`
  - Entire `-3.54`
  - ROI `-2.47`
- `ep11`
  - Entire `-4.18`
  - ROI `-4.06`

当前阶段可得出的判断是：

1. 这轮蒸馏 **没有超过** 当前最新 best fusion student
2. 最好的 checkpoint 是：
   - **`best distill-on-new-student checkpoint = ep5`**
3. 这说明旧的蒸馏 recipe 不能直接无缝迁移到新的更强 student 上
4. 当前更像是：
   - teacher supervision 与更强的时序 radar / Top-K student 表征之间存在目标不一致
   - 或当前 distillation weight / feature matching 方式过强，压制了新 student 已学到的有效表征
5. 因此在继续扩大蒸馏实验前，更合理的顺序应是：
   - 先同步 distill 配置到当前 student 完整配方
   - 再降低 distillation weight（例如 `0.1 -> 0.05`）
   - 然后再判断 teacher 是否仍然适合作为当前 student 的监督源

---

## 5. 当前阶段最重要的工程判断

### 5.1 什么已经被证明

- baseline 已经可靠
- Fusion 方向已经可靠
- 最优 student 已经明确
- LiDAR teacher 已经训练并验证完成
- distillation 的训练基础设施已经接通
- 但当前蒸馏方案尚未超过最新 best fusion student

### 5.2 什么还没有被证明

- 当前这版 LiDAR-guided distillation 是否能在 **最新 best student（big3 @ ep5）** 之上继续带来稳定增益
- 蒸馏收益是否能同时覆盖 Entire Area 与 ROI
- 当前 `lambda=0.1` 是否最合适

### 5.3 现在最合理的实验策略

- 不再回头做老的 fusion 版本对比
- 已有 `resume-distill` 结果保留为参考，不再继续扩大这条分支
- 当前不再继续盲目扩大蒸馏训练轮次
- 下一步优先做：
  - 以 `big3 @ ep5` 为新的 warm-start student 起点
  - 同步 distill config 到当前最新 student 配方
  - 降低 distillation weight
  - 再做新一轮 warm-start only 验证
- 对 `Pedestrian` 的 class-wise supervision 微调已完成第一轮验证，当前这组参数不作为继续扩大的主线

### 5.4 分类别分析与后续结构建议

基于当前 best fusion student `big3 @ ep5 = 43.16 / 64.78` 的表现，可以更清楚地按类别拆解当前系统的优势与短板：

| 类别 | 当前现象 | 最可能的问题来源 | 最可能有效的改法 |
|------|----------|------------------|------------------|
| `Car` | 有提升，但不同实验间波动较大；有时会被结构改动拖累 | 更依赖稳定几何，容易被过强图像注入、decode 参数和类别偏置影响 | 距离感知 gate；更稳的 class-wise decoding；少做激进结构改动 |
| `Pedestrian` | 已明显改善，但仍是最敏感、最容易限制整体上限的类别 | 目标小、点少、雷达回波弱；更缺高分辨率图像细节和更积极监督 | 引入更浅层图像特征；提高 Pedestrian task 权重；单独微调 Pedestrian `min_radius` / assignment |
| `Cyclist` | 当前最强、最容易从新方案中获益 | 明显受益于图像补充、时序 radar、RCS/速度信息 | 保留 `radar_3frames`、RCS Top-K、temporal weighting；进一步试 Doppler/RCS feature engineering |

这三类目标对系统的需求明显不同：

- `Car` 更需要稳定的几何主导，而不是更强的图像依赖
- `Pedestrian` 更需要高分辨率视觉细节和更积极的 task supervision，而这条方向已经被新 best 结果验证
- `Cyclist` 更吃时序 radar、速度信息和当前已经验证有效的 `Radar changes`

#### 当前最值得优先做的一项

如果只选一个最值得继续投入、且预期收益最高的方向，当前最推荐的是：

- **在保持当前 `main_struct_radar_changes` 不变的基础上，为 `Pedestrian` 引入更高分辨率图像特征**

推荐理由：

1. `Pedestrian` 是当前最明显的短板，也是最限制整体上限的类别
2. 当前 fusion 路线已经证明图像分支有效，但主力 `layer3` 特征对小目标仍偏粗
3. `Radar changes` 已经把 radar 侧信息榨得更充分，下一步更大的边际收益更可能来自视觉分支对小目标的补偿
4. 这类改动比重做 backbone 或大改检测头风险更低，也更符合你当前“保留已有最优点、按短板补齐”的策略

预期收益判断：

- 对 `Pedestrian` 的提升潜力最大
- 如果实现得当，最可能带来 **中等幅度** 的整体 mAP 增益
- 相比继续做更重的通用结构改动，这条线更有机会在不破坏 `Cyclist` 优势的前提下继续提高总体结果

#### 已实施的代码修改

基于上面的分析，当前代码已经完成一版专门面向 `Pedestrian` 的视觉分支增强，且保持现有 `main_struct_radar_changes` 主线不变。具体改动为：

1. **图像 backbone 支持双尺度输出**
   - 在 `ResNet18` 图像分支中保留原来的 `layer3` 语义特征，同时新增对 `layer2 + layer3` 联合输出的支持
   - 这样可以在不改变 radar 主干的前提下，把更高分辨率的视觉细节引入 fusion

2. **point-level 图像特征采样支持多尺度拼接**
   - 对每个 radar point，分别从 `layer2` 和 `layer3` 上做 image feature sampling
   - 然后在 point 级别拼接为新的 image feature，再送入现有 `PointGatedFusion`
   - 设计目标是：保留 `layer3` 的语义能力，同时补充 `layer2` 对小目标更友好的空间细节

3. **新增基于当前 best 路线的实验配置**
   - 新增：
     - `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres`
   - 它继承当前已经验证有效的：
     - `main_struct_radar_changes`
   - 并仅额外开启：
     - `image_branch.feature_layer = layer2_layer3`

4. **保留已验证有效的 radar 侧增强**
   - `radar_3frames`
   - temporal weighting
   - RCS Top-K
   - `main` 结构迁移带来的 radar-side 改动

这样做的目的不是重新设计一套新模型，而是在当前 best fusion 路线之上，最小化地补强 `Pedestrian` 所需的高分辨率视觉信息。

#### 第二轮已实施的代码修改：structured dual-scale fusion + context-aware gate

基于 `ped_highres` 的结果分析，当前代码又进一步补上了两个更关键的结构改动：

1. **更结构化的双尺度融合**
   - 不再只是把 `layer2 + layer3` 粗暴拼接后送进单一 gate
   - 现在会按尺度拆分：
     - `layer2`
     - `layer3`
   - 分别做图像特征投影与独立 gate，再与 radar feature 融合
   - 设计目标是让高分辨率细节和高层语义各自保留独立作用，而不是在进入 gate 前就完全混在一起

2. **distance / density-aware gate**
   - 在 pillar 级别额外引入两类上下文：
     - 距离（distance）
     - 稠密度（density / num_points）
   - 让 gate 不再只依赖 `radar feature + image feature`
   - 而是能显式学习：
     - 远距离 / 稀疏区域更信图像
     - 近距离 / 几何充足区域更信 radar

3. **新增实验配置**
   - 新增：
     - `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres_gateaware`
   - 它继承：
     - `main_struct_radar_changes`
   - 并进一步开启：
     - `feature_layer = layer2_layer3`
     - `structured_fusion = true`
     - `use_context_gate = true`

这一步的设计目的，是把上一轮已经验证有效的 “Pedestrian 需要更高分辨率视觉信息” 进一步落实为：

- 更合理的双尺度融合结构
- 更聪明的按条件选择图像信息的 gate

而不是继续单纯增加图像特征量。

#### 第三轮已实施的代码修改：Pedestrian 的 class-wise supervision 微调

基于 `ped_highres @ ep11` 已经成为当前 best fusion student 的结果，当前代码又进一步补上了一轮更直接面向 `Pedestrian` 的 supervision 微调。目标不是再扩大结构复杂度，而是让现有检测头在训练时对 `Pedestrian` 获得更积极、更稳定的监督。

这轮修改包含两部分：

1. **新增 task 级 loss 权重**
   - 在 `CenterHead.loss()` 中加入 `task_loss_weights`
   - 允许对 `Car / Pedestrian / Cyclist` 三个 task 分别设置整体 supervision 强度
   - 当前默认值仍保持：
     - `[1.0, 1.0, 1.0]`
   - 因此不会影响已有实验的可复现性

2. **新增 `Pedestrian` supervision 微调配置**
   - 新增：
     - `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres_cls_tuned`
   - 它继承当前 best 路线：
     - `fusion_finetune_layer3_stable_radar_to_lidar_ped_highres`
   - 并仅额外做两项 class-wise 调整：
     - `train_cfg.min_radius = [4, 3, 2]`
     - `train_cfg.task_loss_weights = [1.0, 1.25, 1.0]`

这两项调整的意图分别是：

- **增大 `Pedestrian` 的最小热图半径**
  - 给小目标更多正样本覆盖范围
  - 降低其因为目标过小而 supervision 过弱的问题

- **提高 `Pedestrian` task 的整体 loss 权重**
  - 让当前已经被证明有效的高分辨率视觉信息，更容易真正转化为检测头参数更新
  - 同时保持 `Car` 与 `Cyclist` 路线不被推翻

这一步的设计原则仍然是：

- 不回退当前已验证有效的 `radar_3frames + temporal weighting + RCS Top-K`
- 不推翻当前 best 的 `ped_highres` 路线
- 只对当前最短板类别补更强的 class-wise supervision

#### 第四轮已实施的代码修改：point + BEV fusion、temporal radar encoder、LiDAR auxiliary geometry supervision

在前面几轮实验已经证明：

- point-level radar-image fusion 有效
- `radar_3frames + temporal weighting + RCS Top-K` 有效
- 直接 feature KD 暂未稳定超过 best student

之后，当前代码进一步补上了三项更大的结构升级，并把它们组合进一个新的实验配置中：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3`

这三项升级分别是：

1. **point fusion 之上再加轻量 BEV-aware fusion**
   - 在原有 point-level image sampling + point fusion 之后
   - 额外把 image pillar feature 投影并散射成 image BEV guidance
   - 在 neck 特征上再做一次 gated BEV fusion / refinement
   - 目标是让图像信息不只在点级别进入系统，也能在 BEV 空间帮助上下文聚合

2. **显式 temporal radar encoder**
   - 不再只把 `radar_3frames` 当作输入侧工程增强
   - 在 voxelization 后、voxel encoder 前对 `RCS / v_r / v_r_compensated / time` 做轻量 residual temporal encoding
   - 目标是让模型更明确地利用时间与速度之间的关系，而不是只依赖人工时间衰减

3. **训练期 LiDAR auxiliary geometry supervision**
   - 不使用完整 feature KD
   - 改为在 neck 特征上新增 geometry aux head
   - 用训练期 LiDAR 点云生成 BEV occupancy / density target
   - 只在训练时加 auxiliary loss，推理时不依赖 LiDAR
   - 目标是把 LiDAR 作为更轻、更稳的 privileged geometry supervision 引入

对应代码层面的主要实现包括：

- `CenterPoint` 中新增：
  - temporal radar encoder
  - image BEV guidance 生成
  - neck-level BEV gated fusion
  - geometry aux target 构建与 loss 计算
- `train.py` 中新增：
  - 当 `geometry_aux.enabled=true` 时自动加载 teacher LiDAR
- 新增模块：
  - `BEVGatedFusion`
  - `TemporalRadarEncoder`
  - `GeometryAuxHead`

这一步的设计意图是：

- 保留当前 best `ped_highres` 的有效点
- 同时把“更强融合”“更显式时间建模”“更轻的 LiDAR supervision”三条大改方向一起接通
- 为后续大幅度结构提升提供一个可直接训练的统一起点

#### 第四轮后的平衡修正配置：`big3_car_balance`

基于 `big3 @ ep5` 的结果分析，当前最主要的剩余问题不是整体 mAP 不够高，而是：

- `Pedestrian / Cyclist` 提升明显
- 但 `Car ROI` 相对上一版 `ped_highres` 有所回落

因此当前代码进一步新增了一个最小改动版本：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_car_balance`

这版配置不改动 `big3` 的主结构，只做一项平衡修正：

- `geometry_aux.loss_weight: 0.2 -> 0.1`

这一步的设计意图是：

- 保留 `big3` 已经验证有效的：
  - point + BEV fusion
  - temporal radar encoder
  - LiDAR auxiliary geometry supervision
- 但减弱 geometry auxiliary loss 对 neck feature 的牵引强度
- 观察是否能在尽量保住 `Pedestrian / Cyclist` 增益的同时，把 `Car ROI` 拉回更好的平衡点

当前对这版配置的判断已经更新为：

- **已完成第一轮验证**
- **对 `Car ROI` 有帮助，但整体 ROI 仍未超过 `big3 @ ep5`**

#### 第四轮后的 ROI 定向改进配置：`big3_roi_geom`

在确认：

- decode / threshold sweep 对 ROI 几乎没有影响
- `geometry_aux.loss_weight` 会明显改变 `Car ROI`

之后，代码中进一步加入了 **ROI-aware geometry supervision**：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_geom`

该版本的核心不是改 `big3` 主结构，而是把 LiDAR auxiliary geometry supervision 改成：

- **ROI 内强监督**
- **ROI 外弱监督或不监督**

当前实现细节为：

- 在 `geometry_aux` 中加入 `roi_aware` 子配置
- 用 BEV 网格中心坐标构造 driving-corridor 风格的 ROI mask
- occupancy / density loss 都按 ROI mask 加权
- 新配置当前默认使用：
  - `x_range = [0.0, 51.2]`
  - `y_range = [-10.0, 10.0]`
  - `outside_weight = 0.0`

这样做的目的，是让 geometry auxiliary loss 更贴近最终关注的 ROI 指标，而不是被整张 BEV 的非 ROI 区域一起主导。

当前对这版配置的判断为：

- **已完成第一轮验证**

其验证结果为：

- `ep2`: `37.91 / 57.57`
- `ep5`: `38.14 / 62.68`
- `ep8`: `43.34 / 64.35`
- `ep11`: `40.32 / 63.39`

当前结论为：

- **它没有超过当前 ROI best `big3 @ ep5 = 64.78`**
- 但它在 `ep8` 达到了：
  - Entire `43.34`
  - ROI `64.35`
- 说明 ROI-aware geometry supervision 这个方向本身不是无效，而是当前这版 ROI mask 偏硬，早期训练明显不稳，后期也没有完全转成更高 ROI

更细地看：

- 当前配置使用的是：
  - `y_range = [-10, 10]`
  - `outside_weight = 0.0`
- 这相当于几乎只在 corridor 内做 geometry auxiliary loss
- 从结果看，这种 **hard ROI masking** 过于激进：
  - 前期收敛明显变差
  - 最终 ROI 也仍略低于 `big3 @ ep5`

因此当前对这版配置的工程判断更新为：

- **ROI-aware geometry supervision 值得继续，但应改成 soft ROI-aware，而不是 hard mask**
- 下一步更合理的是：
  - 保留 ROI-aware
  - 但给 ROI 外保留少量 geometry supervision
  - 例如 `outside_weight = 0.2 ~ 0.5`

为此，代码中已进一步加入：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_geom_soft`

该配置保持：

- `x_range = [0.0, 51.2]`
- `y_range = [-10.0, 10.0]`

但把：

- `outside_weight: 0.0 -> 0.3`

作为当前 soft ROI-aware geometry supervision 的第一版验证点。

该版本的验证结果为：

- `ep2`: `41.00 / 62.46`
- `ep5`: `40.97 / 63.35`
- `ep8`: `42.20 / 62.98`
- `ep11`: `41.11 / 63.94`

当前对 soft ROI-aware 的判断为：

- **比 hard ROI-aware 明显更稳**
- **最好 ROI 为 `63.94`，但仍低于 hard ROI-aware 的 `64.35`**
- **但仍未超过当前 ROI best `big3 @ ep5 = 64.78`**

更准确地说：

- hard ROI-aware best: `ep8 = 64.35`
- soft ROI-aware best: `ep11 = 63.94`

因此当前阶段的结论应更新为：

- `outside_weight = 0.3` 比 `0.0` 更稳定，尤其后期不再像 hard mask 那样明显失衡
- 但 soft ROI-aware 这一版仍没有超越原始 `big3`
- 说明问题并不只是 “ROI 外 geometry supervision 太多”，更可能是：
  - ROI mask 的几何范围还不够精确
  - 或者 geometry aux 的作用方式本身还需要更细的平衡，而不是单纯按空间 hard/soft 加权

#### 第四轮后的 ROI 定向改进配置：`big3_roi_train_weight`

在 ROI-aware geometry supervision 两轮尝试后，当前代码进一步加入了 **ROI-aware training weighting**：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight`

这次不再改 geometry auxiliary loss，而是直接对 detection supervision 做轻量 ROI 偏置：

- ROI 内的正样本框赋予更高 bbox loss 权重
- heatmap loss 也按 ROI 正样本比例做轻量放大

当前实现方式为：

- 在 `head.train_cfg` 中加入 `roi_aware_training`
- 依据每个 GT box center 是否落在 ROI 内，为正样本框生成 `obj_weight`
- 当前第一版配置使用：
  - `x_range = [0.0, 51.2]`
  - `y_range = [-10.0, 10.0]`
  - `roi_weight = 1.2`

这版设计刻意保持轻量，原因是前面的 class-wise supervision 已经说明 detection supervision 很敏感；因此这里先只做小幅 ROI 偏置，不直接大改 target assignment。

当前对这版配置的判断为：

- **已完成第一轮验证**

其验证结果为：

- `ep2`: `39.36 / 59.35`
- `ep5`: `42.10 / 64.01`
- `ep8`: `42.00 / 63.19`
- `ep11`: `42.80 / 65.75`

当前结论为：

- **它已经超过当前旧 ROI best `big3 @ ep5 = 64.78`**
- **新的 ROI best 已更新为 `big3_roi_train_weight @ ep11 = 65.75`**
- 这说明 ROI-aware training weighting 比 ROI-aware geometry supervision 更适合当前系统

分类别看：

- ROI Car: `65.55 -> 67.32`
- ROI Pedestrian: `39.15 -> 42.92`
- ROI Cyclist: `89.64 -> 87.01`

也就是说：

- `Car` 与 `Pedestrian` 明显受益
- `Cyclist` 有所回落
- 但总 ROI 仍然达到当前最高

工程判断更新为：

- **如果目标是 ROI，当前最值得保留的不是 ROI-aware geometry，而是 ROI-aware training weighting**
- 当前这版已经足以作为 ROI-oriented 主结果

另外，后续又补做了一次 **bbox-only ROI weighting** 的最小补充验证，使用配置：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_bbox_only`

这版保持相同 ROI 范围与 `roi_weight = 1.2`，但只对 bbox regression 应用 ROI-aware weighting，不对 heatmap 做 ROI 加权。记录到的关键结果为：

- `ep2`: `41.03 / 62.90`
- `ep5`: `42.51 / 61.62`
- `ep8`: `42.83 / 62.05`
- `ep11`: `42.37 / 65.09`

对这组追加结果的判断是：

- bbox-only ROI weighting 这条线本身是有效的
- 但它并**没有超过**完整 ROI-aware training weighting 的最佳结果 `65.75`
- 因此当前 **ROI best 仍然保持为 `big3_roi_train_weight @ ep11 = 65.75`**

这也说明：

- 当前 ROI 提升不只是 bbox weighting 带来的
- heatmap weighting 也确实对最终 ROI best 有贡献
- 所以如果目标是 ROI，最终应保留：
  - `big3_roi_train_weight`
  - 而不是 `big3_roi_bbox_only`

另外，后续又补到一条 `big3_roi_bbox_only` 的补充结果：

- Entire: `42.25`
- ROI: `63.61`

这条结果同样没有刷新 `bbox-only` 自己的最好点，也没有改变当前总判断：

- `bbox-only` 不是最终 ROI 主线
- 当前最优 ROI 仍然来自完整的 `big3_roi_train_weight`

如果还要继续做 ROI-aware detection supervision 的最小补充验证，当前代码也已加入：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_heatmap_only`

该配置与 `bbox-only` 对称：

- `apply_to_heatmap: true`
- `apply_to_bbox: false`

它的目的非常直接：

- 验证当前 ROI 提升里，heatmap / 正样本分配是否比 bbox regression 更关键
- 形成 `full / bbox-only / heatmap-only` 三者之间的最小对照组

该版本现已完成第一轮验证，记录到的关键结果为：

- `ep2`: `40.62 / 59.60`
- `ep5`: `42.96 / 64.15`
- `ep8`: `42.25 / 64.84`
- `ep11`: `41.16 / 60.06`

当前对 `heatmap-only` 的判断为：

- 它本身是有效的，说明 ROI heatmap / 正样本分配确实能带来明显收益
- 它的最好 ROI `64.84` 高于 `big3_roi_geom` 与 `ped_highres`
- 但仍低于：
  - `big3_roi_bbox_only = 65.09`
  - `big3_roi_train_weight = 65.75`

因此当前更明确的结论是：

- ROI-aware detection supervision 的两部分都有效：
  - heatmap
  - bbox
- 但 **单独只做其中一部分都不如两者一起做**
- 这也是为什么最终 ROI best 仍然是：
  - `big3_roi_train_weight @ ep11 = 65.75`

如果继续沿“ROI 主线 + 轻量单目先验”方向推进，当前代码也已加入：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior`

这版不再继续深挖 LiDAR 或 ROI weighting 变体，而是复用现有冻结图像分支，把 image BEV guidance 变成一个轻量的 camera prior：

- 不改主干 detector
- 不新增外部依赖
- 不训练独立相机模型
- 只对各 task 的 heatmap logits 注入一个小幅 image prior bias

它的目的很明确：

- 在当前 ROI best 主线上，测试单目语义/外观先验是否还能进一步改善 `Pedestrian` 与整体 ROI
- 形成一条与 LiDAR supervision 正交的新尝试

该版本现已完成第一轮验证，记录到的关键结果为：

- `ep2`: `40.35 / 60.08`
- `ep5`: `41.39 / 63.87`
- `ep8`: `43.55 / 64.59`
- `ep11`: `42.57 / 64.56`

当前对 `camera_prior` 的判断为：

- 它**刷新了当前最高 Entire Area mAP**
- 但**没有超过当前 ROI best `65.75`**
- 因此它更适合作为：
  - 当前 **overall best**
  - 而不是新的 ROI best
- 后续复现实验也验证了相同趋势：
  - `ep5`: `41.39 / 63.87`
  - `ep8`: `43.55 / 64.59`
  - `ep11`: `42.57 / 64.56`
  - 说明该线的最佳点仍稳定出现在 `ep8` 左右，且主要贡献是整体 Entire 提升，而不是 ROI 刷新

分类别看，`camera_prior` 的主要效果是：

- `Car` 整体更强
- `Cyclist` 保持高位
- `Pedestrian ROI` 没有超过 `big3_roi_train_weight`

所以更准确的工程判断是：

- 如果目标是 **整体性能**，当前应优先保留 `camera_prior`
- 如果目标是 **ROI**，当前仍应保留 `big3_roi_train_weight`

在此基础上，代码层面已继续接入一个更收敛的 ROI 新尝试：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_ped_car`

这版不是再增加新模块，而是把已有的 ROI-aware detection weighting 从“全类统一加权”改成“按类别区别加权”：

- `Car`: `1.1`
- `Pedestrian`: `1.2`
- `Cyclist`: `1.0`

设计动机是：

- 保留 ROI weighting 已经验证有效的主线
- 继续优先增强当前最关键的 `Pedestrian`
- 对 `Car` 只做轻量加权
- 避免继续对 `Cyclist` 施加统一 ROI 拉动，从而减少类别平衡被破坏的风险

该版本现已完成验证，关键结果为：

- `ep2`: `40.50 / 61.52`
- `ep5`: `40.88 / 62.57`
- `ep8`: `41.72 / 64.22`
- `ep11`: `41.78 / 64.16`

当前对 `roi_train_weight_ped_car` 的判断是：

- 它**没有超过**当前 ROI best `65.75`
- 也**没有超过**原始统一 full ROI weighting 的最佳点
- 说明在当前系统里，把 ROI weighting 从“全类统一”改成“只重点推 Pedestrian + Car”并没有带来更好的总 ROI

更准确地说，这条线验证了一个反向结论：

- 当前 ROI 增益并不只是来自 `Pedestrian` 与 `Car`
- `Cyclist` 虽然不是主要优化目标，但统一 ROI weighting 仍然在整体平衡上更优
- 因此 ROI 主线仍应保持：
  - `big3_roi_train_weight @ ep11 = 42.80 / 65.75`

如果继续沿 `camera prior` 方向做最后一个最小实验，代码层面也已接入：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior_ped_only`

这版不再把 camera prior 注入所有 detection task，而是只对 `Pedestrian` task 生效：

- 保留当前 ROI best 主线 `big3_roi_train_weight`
- 只对 `Pedestrian` heatmap logits 注入冻结图像分支产生的轻量 prior bias
- 不再作用于 `Car` 与 `Cyclist`

设计目的很明确：

- 继续利用单目先验对 `Pedestrian` 的潜在帮助
- 避免之前全任务 `camera_prior` 更偏向提升 Entire 而非 ROI 的问题
- 用一个最小、正交、可解释的改动测试 `Pedestrian ROI` 是否还能继续提升

该版本现已完成验证，关键结果为：

- `ep2`: `41.43 / 61.08`
- `ep5`: `41.17 / 63.35`
- `ep8`: `42.59 / 64.64`
- `ep11`: `42.11 / 63.45`

当前对 `camera_prior_ped_only` 的判断是：

- 它**没有超过**当前 ROI best `65.75`
- 但它相对全任务 `camera_prior @ ep8 = 43.55 / 64.59`，在 ROI 上有非常小的回升：
  - `64.59 -> 64.64`
- 同时它牺牲了整体 Entire：
  - `43.55 -> 42.59`

因此这条线说明：

- 把 camera prior 限制到 `Pedestrian` task，确实让这条线更偏 ROI
- 但这个提升幅度太小，仍不足以超过当前最强 ROI 主线
- 所以 `camera_prior` 系列更适合作为：
  - 整体性能增强分支
  - 而不是最终 ROI 最优解

如果继续做真正新的结构尝试，当前代码层面也已接入：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_radar_attn`

这版不再继续拆 ROI weighting 或 camera prior，而是在 `neck` 输出的 radar BEV feature 上增加一个轻量 self-attention refinement block：

- 位置：`neck -> radar_bev_attention -> BEV fusion -> head`
- 不修改 voxelization
- 不修改点级几何流程
- 不引入重型 transformer

当前实现采用的是一个轻量 pooled self-attention：

- query 使用完整 BEV feature
- key / value 先做空间下采样
- 再接一个 1x1 FFN residual block

设计目的很明确：

- 在 detection head 前增强 radar BEV 的长程上下文建模
- 先做 radar 侧 refinement，再进入现有的 image BEV fusion
- 用一个最小结构改动测试 ROI 上下文是否仍有可挖空间

该版本现已完成验证，关键结果为：

- `ep2`: `40.42 / 59.34`
- `ep5`: `38.60 / 59.59`
- `ep8`: `39.03 / 61.95`
- `ep11`: `41.55 / 63.55`

当前对 `radar_attn` 的判断是：

- 它**没有超过**当前 ROI best `65.75`
- 也**没有超过**原始 `big3 @ ep5 = 43.16 / 64.78`
- 甚至没有超过较早的 `ped_highres @ ep11 = 41.79 / 64.59`

更关键的是，这条线还带来了额外训练开销：

- 原有主线通常约 `8:05 ~ 8:20 / epoch`
- `radar_attn` 提升到约 `9:45 ~ 10:00 / epoch`

所以当前这版 lightweight radar BEV self-attention 的工程结论很直接：

- 增加了训练成本
- 但没有带来足够的性能回报
- 因此**当前实现不值得继续深挖**

另外，代码层面也已把早期写死的 `radar_3frames` 数据增强改成通用的 `radar_Nframes` 版本：

- 当前已支持通过 `dataset.radar_folder=radar_5frames` 直接切换
- 历史帧的 `RCS / v_r / v_r_compensated` 会按帧龄递减加权
- 时间通道会按帧龄重新归一化到 `[0, 1]`

对应新实验入口：

- `fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_radar5frames`

该版本目前仅完成代码接入，**尚未记录正式结果**。

---

## 6. 当前推荐使用的模型与文件

### 6.1 baseline

- `centerpoint_radar_baseline @ ep5`

### 6.2 best fusion student

- Overall best:
  - `outputs/centerpoint_radar_gated_fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior/checkpoints/ep8-centerpoint_radar_gated_fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight_camera_prior.ckpt`
- ROI best:
  - `outputs/centerpoint_radar_gated_fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight/checkpoints/ep11-centerpoint_radar_gated_fusion_finetune_layer3_stable_radar_to_lidar_big3_roi_train_weight.ckpt`

### 6.3 best LiDAR teacher

- `outputs/centerpoint_lidar_teacher/checkpoints/ep11-centerpoint_lidar_teacher.ckpt`

### 6.4 后续正式 distill 的启动方式

应该使用：

- `pretrained_student_checkpoint_path=...best_student.ckpt`
- `model.distillation.teacher_checkpoint_path=...teacher.ckpt`

不应再使用：

- `checkpoint_path=...best_student.ckpt`

因为那会恢复 optimizer 与 epoch 状态，解释性较差。

---

## 7. 文档体系说明

当前 `Pretrain_Sanity check` 的文档结构建议理解为：

- `PRETRAIN_SANITY_MASTER_REPORT_CN.md`
  - 唯一总入口
- `archive/`
  - Step 0 历史文档
- `Fusion/archive/`
  - Fusion 历史文档、中间融合稿
- `Fusion/` 当前层级保留的调试说明 / 快速运行说明
  - 作为工程执行文档继续保留
  - 但其中关键信息已经并入本总报告

---

## 8. 一句话总结

`Pretrain Sanity Check` 已经完成从 baseline 建立、fusion 收敛、结构迁移带来的 student 再提升、叠加 `Radar changes` 后的输入侧增强验证、高分辨率视觉特征对 `Pedestrian` 的有效补强、point + BEV fusion / temporal radar encoder / LiDAR auxiliary geometry supervision 的大改组合验证、ROI-aware geometry supervision 与 ROI-aware training weighting 的对比，以及轻量 camera prior 的验证；当前最清楚的结论是：**如果追求整体最好，选 `43.55 / 64.59`；如果追求 ROI 最好，选 `42.80 / 65.75`。**

---

## 9. 最终结论写法摘要

如果需要把当前结论直接写进论文或最终报告，可以使用下面这版摘要逻辑：

1. 我们先通过结构改进、雷达时序/特征增强以及高分辨率图像特征，逐步将 radar-image 基线从 `31.96 / 53.59` 提升到多个显著更强的 student 版本。
2. 在整体性能上，`big3_roi_train_weight_camera_prior @ ep8` 取得了当前最高 Entire Area mAP：`43.55`，说明在当前 ROI 主线之上补充一个轻量的冻结单目先验，能够进一步提升整体检测性能。
3. 在 driving corridor ROI 上，最佳结果来自 `big3_roi_train_weight @ ep11 = 42.80 / 65.75`，说明相比继续深挖 ROI-aware geometry supervision，轻量的 ROI-aware detection weighting 更适合当前系统的 ROI 优化目标。
4. `big3_roi_bbox_only` 的结果进一步表明，ROI 提升并不只来自 bbox regression weighting；完整的 ROI-aware training weighting 比 bbox-only 版本更优，因此最终 ROI 主线应保留 `big3_roi_train_weight`。
5. 总体而言，LiDAR 在当前项目中更适合作为训练期的辅助几何监督，而不是持续深挖的主增益来源；如果目标是 ROI，最有效的增益主要来自 detection supervision 的重新分配，而不是更复杂的 LiDAR-guided 约束。
