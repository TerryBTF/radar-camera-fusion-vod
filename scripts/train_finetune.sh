#!/bin/bash
#SBATCH --mpi=pmix
#SBATCH --job-name=ro47020_radar_finetune
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=<your-slurm-account>
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# 激活环境
conda activate amp

# 进入工作目录
cd <repo_root>

# 运行训练（会自动从最新检查点继续）
python src/tools/train.py \
  --config-name fusion_finetune_layer3 \
  num_workers=0

echo "Training completed or interrupted!"
