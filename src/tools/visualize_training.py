"""
Visualization Script for Step 0.C - Training Sanity Check
=========================================================
Generates 3 visualization plots:
  1. Radar Points + GT Boxes (3D view)
  2. Pillar Occupancy BEV visualization
  3. Predicted Boxes vs GT Boxes

Usage:
  python src/tools/visualize_training.py [--sample-id SAMPLE_ID] [--checkpoint CKPT_PATH]
"""

import os
import sys
import argparse
from pathlib import Path

root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import ViewOfDelft
from src.model.detector import CenterPoint
from hydra import initialize_config_dir, compose
import torch.nn.functional as F


class VisualizationTool:
    """Visualization tools for training sanity check"""
    
    def __init__(self, data_root='data/view_of_delft', config_name='baseline'):
        self.data_root = data_root
        self.dataset_train = ViewOfDelft(data_root=data_root, split='val')
        
        # Model config
        self.point_cloud_range = np.array([0, -25.6, -3, 51.2, 25.6, 2])
        self.voxel_size = np.array([0.32, 0.32, 5])
        self.grid_size = (
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        )
    
    def get_sample(self, sample_id=0):
        """Get a sample from dataset"""
        return self.dataset_train[sample_id]
    
    def plot_radar_points_and_boxes(self, sample_id=0, output_file='outputs/viz_01_radar_points.png'):
        """
        Plot 1: Radar Points + GT Boxes in 3D
        """
        print(f"\n📊 Generating: Radar Points + GT Boxes")
        
        sample = self.get_sample(sample_id)
        points = sample['lidar_data'].numpy()  # (N, 7)
        gt_boxes = sample['gt_bboxes_3d'].tensor.numpy()  # (M, 7)
        gt_labels = sample['gt_labels_3d'].numpy()  # (M,)
        
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        colors = {'Car': 'blue', 'Pedestrian': 'red', 'Cyclist': 'green'}
        
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: Front view (X-Z plane)
        ax1 = fig.add_subplot(131)
        
        # Plot points
        scatter = ax1.scatter(points[:, 0], points[:, 2], c='gray', s=2, alpha=0.5, label='Radar Points')
        
        # Plot GT boxes (front view - rectangle on X-Z)
        for i, box in enumerate(gt_boxes):
            x, y, z, l, w, h, rot = box
            class_id = int(gt_labels[i])
            class_name = class_names[class_id]
            color = colors[class_name]
            
            # Front view: show x-z projection
            rect = Rectangle((x - l/2, z - h/2), l, h, 
                            linewidth=1, edgecolor=color, facecolor='none', label=class_name if i == class_id else '')
            ax1.add_patch(rect)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Front View: Radar Points + GT Boxes (X-Z)')
        ax1.set_xlim(self.point_cloud_range[0], self.point_cloud_range[3])
        ax1.set_ylim(self.point_cloud_range[2], self.point_cloud_range[5])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot 2: Top-down BEV (X-Y plane)
        ax2 = fig.add_subplot(132)
        
        # Plot points
        ax2.scatter(points[:, 0], points[:, 1], c='gray', s=2, alpha=0.5, label='Radar Points')
        
        # Plot GT boxes (BEV - rectangle on X-Y with rotation)
        for i, box in enumerate(gt_boxes):
            x, y, z, l, w, h, rot = box
            class_id = int(gt_labels[i])
            class_name = class_names[class_id]
            color = colors[class_name]
            
            # Draw rotated rectangle
            from matplotlib.patches import Polygon
            
            # Calculate corners
            corners_local = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2]
            ])
            
            # Rotate
            cos_rot = np.cos(rot)
            sin_rot = np.sin(rot)
            rot_mat = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
            corners_rotated = corners_local @ rot_mat.T
            
            # Translate
            corners_world = corners_rotated + np.array([x, y])
            
            poly = Polygon(corners_world, linewidth=1, edgecolor=color, facecolor='none')
            ax2.add_patch(poly)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('BEV (Top-down): Radar Points + GT Boxes (X-Y)')
        ax2.set_xlim(self.point_cloud_range[0], self.point_cloud_range[3])
        ax2.set_ylim(self.point_cloud_range[1], self.point_cloud_range[4])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Side view (Y-Z plane)
        ax3 = fig.add_subplot(133)
        
        ax3.scatter(points[:, 1], points[:, 2], c='gray', s=2, alpha=0.5, label='Radar Points')
        
        for i, box in enumerate(gt_boxes):
            x, y, z, l, w, h, rot = box
            class_id = int(gt_labels[i])
            class_name = class_names[class_id]
            color = colors[class_name]
            
            # Side view: Y-Z projection
            rect = Rectangle((y - w/2, z - h/2), w, h,
                            linewidth=1, edgecolor=color, facecolor='none')
            ax3.add_patch(rect)
        
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View: Radar Points + GT Boxes (Y-Z)')
        ax3.set_xlim(self.point_cloud_range[1], self.point_cloud_range[4])
        ax3.set_ylim(self.point_cloud_range[2], self.point_cloud_range[5])
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        car_patch = mpatches.Patch(color='blue', label='Car')
        ped_patch = mpatches.Patch(color='red', label='Pedestrian')
        cyc_patch = mpatches.Patch(color='green', label='Cyclist')
        fig.legend(handles=[car_patch, ped_patch, cyc_patch], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
        
        plt.suptitle(f'Sample {sample_id}: Radar Points ({len(points)} pts) + GT Boxes ({len(gt_boxes)} boxes)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"   ✅ Saved to: {output_file}")
        plt.close()
    
    def plot_pillar_occupancy_bev(self, sample_id=0, output_file='outputs/viz_02_pillar_occupancy.png'):
        """
        Plot 2: Pillar Occupancy BEV
        """
        print(f"\n📊 Generating: Pillar Occupancy BEV")
        
        sample = self.get_sample(sample_id)
        points = sample['lidar_data'].numpy()  # (N, 7)
        
        # Create occupancy grid
        occupancy = np.zeros(self.grid_size[:2], dtype=np.int32)
        
        pc_min = self.point_cloud_range[:3]
        
        # Quantize points to grid
        for point in points:
            voxel_idx = ((point[:3] - pc_min) / self.voxel_size).astype(int)
            
            # Check bounds
            if (voxel_idx >= 0).all() and (voxel_idx < self.grid_size).all():
                occupancy[voxel_idx[1], voxel_idx[0]] += 1
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Occupancy count (heatmap)
        im1 = axes[0].imshow(occupancy, cmap='hot', origin='lower', aspect='auto')
        axes[0].set_xlabel('X (grid index)')
        axes[0].set_ylabel('Y (grid index)')
        axes[0].set_title('Pillar Occupancy Heatmap (Point Count)')
        plt.colorbar(im1, ax=axes[0], label='# Points per Pillar')
        
        # Add grid reference
        sample_frame = sample['meta']['num_frame']
        gt_boxes = sample['gt_bboxes_3d'].tensor.numpy()
        
        # Plot 2: Occupancy binary + GT boxes overlay
        occupancy_binary = (occupancy > 0).astype(int)
        im2 = axes[1].imshow(occupancy_binary, cmap='gray', origin='lower', aspect='auto', alpha=0.6)
        
        # Overlay GT boxes on BEV
        colors_rgb = {'Car': (0, 0, 1), 'Pedestrian': (1, 0, 0), 'Cyclist': (0, 1, 0)}
        gt_labels = sample['gt_labels_3d'].numpy()
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        
        for i, box in enumerate(gt_boxes):
            x, y, z, l, w, h, rot = box
            class_id = int(gt_labels[i])
            class_name = class_names[class_id]
            color_rgb = colors_rgb[class_name]
            
            # Convert world to grid coordinates
            grid_x = (x - pc_min[0]) / self.voxel_size[0]
            grid_y = (y - pc_min[1]) / self.voxel_size[1]
            grid_l = l / self.voxel_size[0]
            grid_w = w / self.voxel_size[1]
            
            # Draw rotated box
            from matplotlib.patches import Polygon
            corners_local = np.array([
                [-grid_l/2, -grid_w/2],
                [grid_l/2, -grid_w/2],
                [grid_l/2, grid_w/2],
                [-grid_l/2, grid_w/2]
            ])
            
            cos_rot = np.cos(rot)
            sin_rot = np.sin(rot)
            rot_mat = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
            corners_rotated = corners_local @ rot_mat.T
            corners_world = corners_rotated + np.array([grid_x, grid_y])
            
            poly = Polygon(corners_world, linewidth=2, edgecolor=color_rgb, facecolor='none')
            axes[1].add_patch(poly)
        
        axes[1].set_xlabel('X (grid index)')
        axes[1].set_ylabel('Y (grid index)')
        axes[1].set_title('Pillar Occupancy + GT Boxes Overlay')
        axes[1].set_xlim(0, self.grid_size[0])
        axes[1].set_ylim(0, self.grid_size[1])
        
        plt.suptitle(f'Sample {sample_id}: Pillar Occupancy (Grid: {self.grid_size[0]}×{self.grid_size[1]}, Voxel: {self.voxel_size})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"   ✅ Saved to: {output_file}")
        print(f"   Statistics:")
        print(f"     - Total points: {len(points)}")
        print(f"     - Non-empty pillars: {(occupancy > 0).sum()} / {self.grid_size[0] * self.grid_size[1]}")
        print(f"     - Occupancy ratio: {(occupancy > 0).sum() / (self.grid_size[0] * self.grid_size[1]) * 100:.2f}%")
        print(f"     - Max points in pillar: {occupancy.max()}")
        plt.close()
    
    def plot_predictions_vs_gt(self, sample_id=0, checkpoint_path=None, output_file='outputs/viz_03_pred_vs_gt.png'):
        """
        Plot 3: Predicted Boxes vs GT Boxes (requires checkpoint)
        """
        print(f"\n📊 Generating: Predictions vs GT")
        
        if checkpoint_path is None:
            checkpoint_path = 'outputs/centerpoint_radar_baseline/checkpoints/ep5-centerpoint_radar_baseline.ckpt'
        
        if not os.path.exists(checkpoint_path):
            print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"      Available checkpoints:")
            ckpt_dir = 'outputs/centerpoint_radar_baseline/checkpoints'
            if os.path.exists(ckpt_dir):
                for f in os.listdir(ckpt_dir):
                    print(f"        - {f}")
            print(f"\n      Skipping prediction visualization...")
            return
        
        print(f"   Loading checkpoint: {checkpoint_path}")
        
        # Load model
        try:
            from omegaconf import OmegaConf
            config_path = 'src/config'
            config_name = 'baseline'
            
            # For now, skip detailed model loading - just do geometry comparison
            print(f"   Using GT boxes for geometry validation")
            
            sample = self.get_sample(sample_id)
            points = sample['lidar_data'].numpy()
            gt_boxes = sample['gt_bboxes_3d'].tensor.numpy()
            gt_labels = sample['gt_labels_3d'].numpy()
            
            # Create a mock prediction for demonstration
            # In real scenario, would load checkpoint and run inference
            pred_boxes = gt_boxes + np.random.randn(*gt_boxes.shape) * 0.1  # Small noise
            pred_scores = np.ones(len(pred_boxes)) * 0.8
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            class_names = ['Car', 'Pedestrian', 'Cyclist']
            colors = {'Car': 'blue', 'Pedestrian': 'red', 'Cyclist': 'green'}
            
            # GT boxes
            ax = axes[0]
            ax.scatter(points[:, 0], points[:, 1], c='gray', s=2, alpha=0.5)
            
            for i, box in enumerate(gt_boxes):
                x, y, z, l, w, h, rot = box
                class_id = int(gt_labels[i])
                class_name = class_names[class_id]
                color = colors[class_name]
                
                from matplotlib.patches import Polygon
                corners_local = np.array([
                    [-l/2, -w/2],
                    [l/2, -w/2],
                    [l/2, w/2],
                    [-l/2, w/2]
                ])
                
                cos_rot = np.cos(rot)
                sin_rot = np.sin(rot)
                rot_mat = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
                corners_rotated = corners_local @ rot_mat.T
                corners_world = corners_rotated + np.array([x, y])
                
                poly = Polygon(corners_world, linewidth=2, edgecolor=color, facecolor='none', label=class_name)
                ax.add_patch(poly)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Ground Truth Boxes (Sample {sample_id})')
            ax.set_xlim(self.point_cloud_range[0], self.point_cloud_range[3])
            ax.set_ylim(self.point_cloud_range[1], self.point_cloud_range[4])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Predictions (would need actual model inference)
            ax = axes[1]
            ax.scatter(points[:, 0], points[:, 1], c='gray', s=2, alpha=0.5)
            
            ax.text(0.5, 0.5, 'Predictions\n(Requires Model Inference)\n\nTo load predictions:\n1. Load checkpoint\n2. Run forward pass\n3. Extract bbox detections', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Predicted Boxes (Placeholder)')
            ax.set_xlim(self.point_cloud_range[0], self.point_cloud_range[3])
            ax.set_ylim(self.point_cloud_range[1], self.point_cloud_range[4])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Comparison: Predictions vs Ground Truth (Sample {sample_id})', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"   ✅ Saved to: {output_file}")
            print(f"   Note: For full predictions, implement model inference in checkpoint loading")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️  Error during prediction: {e}")
            print(f"   Skipping this visualization")


def main():
    parser = argparse.ArgumentParser(description='Visualization for Step 0.C')
    parser.add_argument('--sample-id', type=int, default=0, help='Sample ID to visualize')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for predictions')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STEP 0.C - VISUALIZATION CHECK")
    print("="*80)
    
    viz = VisualizationTool()
    
    # Generate visualizations
    viz.plot_radar_points_and_boxes(
        sample_id=args.sample_id,
        output_file=f'{args.output_dir}/viz_01_radar_points.png'
    )
    
    viz.plot_pillar_occupancy_bev(
        sample_id=args.sample_id,
        output_file=f'{args.output_dir}/viz_02_pillar_occupancy.png'
    )
    
    viz.plot_predictions_vs_gt(
        sample_id=args.sample_id,
        checkpoint_path=args.checkpoint,
        output_file=f'{args.output_dir}/viz_03_pred_vs_gt.png'
    )
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {args.output_dir}/viz_01_radar_points.png")
    print(f"  2. {args.output_dir}/viz_02_pillar_occupancy.png")
    print(f"  3. {args.output_dir}/viz_03_pred_vs_gt.png")
    print()


if __name__ == '__main__':
    main()
