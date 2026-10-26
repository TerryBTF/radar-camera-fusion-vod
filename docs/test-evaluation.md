# V2 Gated Fusion Test Set Evaluation Instructions

## 说明：版本对应关系

在运行测试之前，务必理解版本的对应关系：

- **v1 (baseline, no suffix)**: `ep2, ep5, ep8, ep11, last.ckpt`
  - 配置：fusion_freeze (image_sampling mode)
  - 特点：Bilinear采样投影点，拼接radar和image特征
  - 性能：Entire Area mAP 32.59, ROI mAP 54.67

- **v2 (current, -v2 suffix)**: `ep2-v2, ep5-v2, ep8-v2, ep11-v2, last-v2.ckpt`
  - 配置：fusion_freeze (pillar_gated_fusion mode)
  - 特点：Pillar层级门控融合，learned sigmoid gate控制融合强度
  - 性能：Preliminary Entire Area mAP ~33.8, ROI mAP ~57.5

## 评估命令

### Prerequisites

```bash
# 确保在GPU节点或有GPU的terminal中运行
# 确保conda环境激活
conda activate amp

# 进入项目目录
cd <repo_root>
```

---

## V2 Checkpoints 评估指令

### 1. Evaluate ep2-v2 (Epoch 3)
```bash
python src/tools/test.py \
  --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep2-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0
```

### 2. Evaluate ep5-v2 (Epoch 6)
```bash
python src/tools/test.py \
  --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep5-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0
```

### 3. Evaluate ep8-v2 (Epoch 9)
```bash
python src/tools/test.py \
  --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep8-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0
```

### 4. Evaluate ep11-v2 (Epoch 12, Final)
```bash
python src/tools/test.py \
  --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep11-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0
```

### 5. Evaluate last-v2 (Latest, usually same as ep11)
```bash
python src/tools/test.py \
  --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/last-v2.ckpt \
  num_workers=0
```

---

## Batch Execution

如果想一次跑完所有评估（推荐），可以用以下脚本：

```bash
#!/bin/bash
cd <repo_root>
conda activate amp

echo "========== Evaluating ep2-v2 =========="
python src/tools/test.py --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep2-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0

echo "========== Evaluating ep5-v2 =========="
python src/tools/test.py --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep5-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0

echo "========== Evaluating ep8-v2 =========="
python src/tools/test.py --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep8-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0

echo "========== Evaluating ep11-v2 =========="
python src/tools/test.py --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/ep11-centerpoint_radar_image_sampling_freeze-v2.ckpt \
  num_workers=0

echo "========== Evaluating last-v2 =========="
python src/tools/test.py --config-name fusion_freeze \
  checkpoint_path=outputs/centerpoint_radar_image_sampling_freeze/checkpoints/last-v2.ckpt \
  num_workers=0

echo "========== All evaluations complete =========="
```

---

## Expected Output

每个评估命令会输出：

1. **Console输出：**
   - Per-class metrics (Car, Pedestrian, Cyclist) 
   - mAP scores (Entire Area, ROI)
   - 按距离的分桶性能
   - Validation summary

2. **WandB日志：**
   - 所有指标自动上传到wandb.ai
   - 可在dashboard查看对比

3. **保存文件：**
   - 预测结果pkl文件
   - 可能的可视化结果

---

## 运行时间预估

- 每个checkpoint评估：约10-15分钟（1296个test样本）
- 总耗时（5个checkpoints）：约50-75分钟
- GPU内存：不超过2GB（足够）

---

## 输出收集方法

完成测试后，你可以：

**方法1：复制console输出**
- 在terminal中运行评估
- 复制最后打印的mAP结果（v2版本）
- 粘贴给我

**方法2：从WandB收集**
- 打开WandB dashboard
- 查看test/* metrics (v2的结果)
- 截图或导出数据

**方法3：直接提供输出内容**
- 告诉我每个v2 checkpoint的以下指标：
  ```
  ep2-v2:  Entire Area mAP = ??, Car = ??, Ped = ??, Cyclist = ??
           ROI mAP = ??, Car = ??, Ped = ??, Cyclist = ??
  ep5-v2:  ...
  ep8-v2:  ...
  ep11-v2: ...
  last-v2: ...
  ```

---

## 样本输出格式

当你提供v2结果时，按以下格式就行：

```
ep2-v2 Test Results:
  Entire Area mAP: 32.10
  - Car: 26.5
  - Pedestrian: 15.8
  - Cyclist: 52.2
  ROI mAP: 55.5
  - Car: 64.0
  - Pedestrian: 23.5
  - Cyclist: 76.8

ep5-v2 Test Results:
  Entire Area mAP: 33.5
  - Car: 27.8
  - Pedestrian: 17.2
  - Cyclist: 54.8
  ROI mAP: 57.2
  ...
```

---

## 💡 关键信息总结

| 项目 | v1 (no suffix) | v2 (with -v2 suffix) |
|-----|------------------|-----------------|
| **fusion mode** | image_sampling | pillar_gated_fusion |
| **fusion方式** | Bilinear采样 + 拼接 | Pillar层级门控残差 |
| **可学习组件** | 无（仅特征采样） | 有（gate MLP + projection W） |
| **可解释性** | 低 | 高（gate统计/per-class） |
| **mAP (Entire Area)** | 32.59 | ~33.8 (+1.19) |
| **mAP (ROI)** | 54.67 | ~57.5 (+2.83) |
| **Checkpoint后缀** | 无 (ep2, ep5, ep8, ...) | -v2 (ep2-v2, ep5-v2, ...) |

---

## 完成后

提供结果后，我会：

1. ✅ 更新 `centerpoint_radar_gated_fusion_v2.md` 报告
2. ✅ 计算v1 (no suffix) vs v2 (-v2) 的精确改进
3. ✅ 识别最优checkpoint
4. ✅ 生成详细的对比分析
5. ✅ 绘制性能曲线（如需要）

---

现在就可以开始运行测试命令了！推荐选择一个evaluation命令复制粘贴运行。
