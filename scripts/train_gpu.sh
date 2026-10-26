#!/bin/bash
# 提交训练任务到 GPU 节点

srun --mpi=pmix \
    --job-name="ro47020_radar" \
    --partition=gpu-a100-small \
    --time=04:00:00 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --gpus-per-task=1 \
    --mem-per-cpu=4G \
    --account=<your-slurm-account> \
    --pty /bin/bash -il -c "cd <repo_root> && python src/tools/train.py --config-name fusion_freeze"
