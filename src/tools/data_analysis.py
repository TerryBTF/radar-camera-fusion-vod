"""
Data Analysis Script for Step 0 Baseline Establishment
======================================================
This script performs mandatory data checks as per Step 0.3:
  A. 每帧 radar 点数分布
  B. 每类目标数量分布
  C. GT box 数量分布
  D. radar 点落在 box 内的比例
  E. pillar 非空比例

Usage:
  python src/tools/data_analysis.py
"""

import os
import sys
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import torch
from torch.utils.data import DataLoader
from src.dataset import ViewOfDelft, collate_vod_batch
from src.ops import Voxelization
import torch.nn.functional as F


class DataAnalyzer:
    """Analyze data distribution for baseline establishment"""
    
    def __init__(self, data_root='data/view_of_delft', split='train'):
        self.data_root = data_root
        self.split = split
        self.dataset = ViewOfDelft(data_root=data_root, split=split)
        
        # Statistics containers
        self.points_per_frame = []                  # A. Radar points per frame
        self.num_objects_per_frame = []             # C. GT boxes per frame
        self.objects_per_class = defaultdict(int)   # B. Target count per class
        self.points_in_box_ratio = []               # D. Ratio of points in boxes
        self.pillar_sparsity = []                   # E. Non-empty pillar ratio
        
        # Model config (from baseline)
        self.point_cloud_range = torch.tensor([0, -25.6, -3, 51.2, 25.6, 2])
        self.voxel_size = torch.tensor([0.32, 0.32, 5])
        self.grid_size = ((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0],
                          (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1],
                          (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        self.grid_size = tuple(int(s) for s in self.grid_size)
        
        # Voxelization config
        # Using CPU-based calculation, so no voxel_layer needed
        # self.voxel_layer = Voxelization(...)
        
    def is_point_in_box(self, point, box):
        """
        Check if a point (x, y, z) is inside a box.
        Box format: [x, y, z, l, w, h, rot] (center-based, rot around z-axis)
        """
        x, y, z = point
        cx, cy, cz, l, w, h, rot = box
        
        # Rotate point to box coordinate frame
        cos_rot = np.cos(-rot)
        sin_rot = np.sin(-rot)
        dx = x - cx
        dy = y - cy
        
        local_x = dx * cos_rot - dy * sin_rot
        local_y = dx * sin_rot + dy * cos_rot
        local_z = z - cz
        
        # Check if point is inside box bounds
        # Box is centered, so we check [-l/2, l/2], [-w/2, w/2], [-h/2, h/2]
        return (abs(local_x) <= l/2 and 
                abs(local_y) <= w/2 and 
                abs(local_z) <= h/2)
    
    def extract_grid_info(self, points_np):
        """Extract grid information (non-empty pillars) - CPU version"""
        try:
            # Simple CPU-based voxelization to count non-empty pillars
            # Quantize points to voxel coordinates
            pc_min = np.array([0, -25.6, -3])
            voxel_size = np.array([0.32, 0.32, 5])
            
            voxel_coords = ((points_np[:, :3] - pc_min) / voxel_size).astype(np.int32)
            
            # Check boundaries
            grid_size = np.array(self.grid_size)
            mask = (voxel_coords >= 0).all(axis=1) & (voxel_coords < grid_size).all(axis=1)
            valid_coords = voxel_coords[mask]
            
            if len(valid_coords) > 0:
                # Count unique occupied voxels
                occupied_voxels = set(map(tuple, valid_coords))
                non_empty_voxels = len(occupied_voxels)
            else:
                non_empty_voxels = 0
            
            total_voxels = self.grid_size[0] * self.grid_size[1]
            sparsity = non_empty_voxels / total_voxels if total_voxels > 0 else 0
            
            return sparsity, non_empty_voxels, total_voxels
        except Exception as e:
            print(f"  Warning: Could not compute voxelization: {e}")
            return 0, 0, self.grid_size[0] * self.grid_size[1]
    
    def analyze_sample(self, idx):
        """Analyze a single sample"""
        sample = self.dataset[idx]
        
        points = sample['lidar_data'].numpy()  # Shape: (N, 7)
        gt_labels = sample['gt_labels_3d'].numpy()
        gt_bboxes = sample['gt_bboxes_3d'].tensor.numpy()
        
        # A. Points per frame
        num_points = len(points)
        self.points_per_frame.append(num_points)
        
        # C. GT boxes per frame
        num_boxes = len(gt_bboxes)
        self.num_objects_per_frame.append(num_boxes)
        
        # B. Objects per class
        for label in gt_labels:
            class_name = self.dataset.CLASSES[label]
            self.objects_per_class[class_name] += 1
        
        # D. Points in boxes
        points_in_boxes = 0
        for i, box in enumerate(gt_bboxes):
            for point in points:
                if self.is_point_in_box(point[:3], box):
                    points_in_boxes += 1
        
        ratio = points_in_boxes / num_points if num_points > 0 else 0.0
        self.points_in_box_ratio.append(ratio)
        
        # E. Pillar sparsity
        sparsity, non_empty, total = self.extract_grid_info(points)
        self.pillar_sparsity.append(sparsity)
        
        return {
            'num_points': num_points,
            'num_boxes': num_boxes,
            'points_in_boxes': points_in_boxes,
            'ratio': ratio,
            'pillar_sparsity': sparsity,
            'pillar_non_empty': non_empty,
            'pillar_total': total
        }
    
    def run_analysis(self):
        """Run full data analysis"""
        print(f"\n{'='*80}")
        print(f"DATA ANALYSIS: {self.split} split")
        print(f"{'='*80}\n")
        
        total_samples = len(self.dataset)
        
        for idx in range(total_samples):
            if idx % max(1, total_samples // 10) == 0:
                print(f"Progress: {idx}/{total_samples}")
            
            try:
                self.analyze_sample(idx)
            except Exception as e:
                print(f"  Error processing sample {idx}: {e}")
                continue
        
        self.print_statistics()
        self.save_statistics()
    
    def print_statistics(self):
        """Print analysis results"""
        points_per_frame = np.array(self.points_per_frame)
        num_objects_per_frame = np.array(self.num_objects_per_frame)
        points_in_box_ratio = np.array(self.points_in_box_ratio)
        pillar_sparsity = np.array(self.pillar_sparsity)
        
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        
        # A. Radar points per frame
        print("\nA. RADAR POINTS PER FRAME DISTRIBUTION")
        print(f"   Count:     {len(points_per_frame)}")
        print(f"   Min:       {points_per_frame.min()}")
        print(f"   Max:       {points_per_frame.max()}")
        print(f"   Mean:      {points_per_frame.mean():.2f}")
        print(f"   Median:    {np.median(points_per_frame):.2f}")
        print(f"   Std:       {points_per_frame.std():.2f}")
        print(f"   Q1 (25%):  {np.percentile(points_per_frame, 25):.2f}")
        print(f"   Q3 (75%):  {np.percentile(points_per_frame, 75):.2f}")
        
        # B. Objects per class
        print("\nB. TARGET QUANTITY DISTRIBUTION BY CLASS")
        total_objects = sum(self.objects_per_class.values())
        for class_name in self.dataset.CLASSES:
            count = self.objects_per_class[class_name]
            pct = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"   {class_name:15s}: {count:6d} ({pct:5.2f}%)")
        print(f"   {'Total':15s}: {total_objects:6d}")
        
        # C. GT boxes per frame
        num_objects_per_frame = np.array(self.num_objects_per_frame)
        print("\nC. GT BOX QUANTITY DISTRIBUTION (per frame)")
        print(f"   Count:     {len(num_objects_per_frame)}")
        print(f"   Min:       {num_objects_per_frame.min()}")
        print(f"   Max:       {num_objects_per_frame.max()}")
        print(f"   Mean:      {num_objects_per_frame.mean():.2f}")
        print(f"   Median:    {np.median(num_objects_per_frame):.2f}")
        print(f"   Std:       {num_objects_per_frame.std():.2f}")
        fr_no_objects = (num_objects_per_frame == 0).sum()
        print(f"   Frames with 0 objects: {fr_no_objects} ({fr_no_objects/len(num_objects_per_frame)*100:.2f}%)")
        
        # D. Points in boxes
        print("\nD. RADAR POINTS FALLING IN GT BOXES")
        print(f"   Count:     {len(points_in_box_ratio)}")
        print(f"   Min:       {points_in_box_ratio.min():.4f}")
        print(f"   Max:       {points_in_box_ratio.max():.4f}")
        print(f"   Mean:      {points_in_box_ratio.mean():.4f}")
        print(f"   Median:    {np.median(points_in_box_ratio):.4f}")
        print(f"   Std:       {points_in_box_ratio.std():.4f}")
        
        # E. Pillar sparsity
        print("\nE. PILLAR (BEV GRID) NON-EMPTY RATIO")
        print(f"   Grid size: {self.grid_size}")
        print(f"   Count:     {len(pillar_sparsity)}")
        print(f"   Min:       {pillar_sparsity.min():.4f}")
        print(f"   Max:       {pillar_sparsity.max():.4f}")
        print(f"   Mean:      {pillar_sparsity.mean():.4f}")
        print(f"   Median:    {np.median(pillar_sparsity):.4f}")
        print(f"   Std:       {pillar_sparsity.std():.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def save_statistics(self):
        """Save statistics to file"""
        output_file = f'outputs/data_analysis_{self.split}.pkl'
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        stats = {
            'split': self.split,
            'points_per_frame': self.points_per_frame,
            'num_objects_per_frame': self.num_objects_per_frame,
            'objects_per_class': dict(self.objects_per_class),
            'points_in_box_ratio': self.points_in_box_ratio,
            'pillar_sparsity': self.pillar_sparsity,
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(stats, f)
            print(f"Statistics saved to: {output_file}")


if __name__ == '__main__':
    # Analyze training split
    analyzer = DataAnalyzer(split='train')
    analyzer.run_analysis()
    
    # Analyze validation split
    analyzer = DataAnalyzer(split='val')
    analyzer.run_analysis()
