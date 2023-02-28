#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36]
#SBATCH --time 1-00:00:00

DATA_DIR="/projects/tir6/strubell/data/fmow-rgb";
CKPT_PATH="/projects/tir6/strubell/jaredfer/projects/SatMAE/finetune_fmow_temporal.pth";


torchrun --nproc_per_node=2 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir logs_temporal \
    --log_dir logs_temporal \
    --batch_size 32 --accum_iter 2 \
    --model vit_base_patch16 --epochs 50 --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 --model_type temporal \
    --dist_eval --num_workers 4 --dataset temporal \
    --train_path $DATA_DIR/train_62classes.csv \
    --test_path $DATA_DIR/val_62classes.csv;