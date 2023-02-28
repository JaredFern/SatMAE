#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36]
#SBATCH --time 1-00:00:00

CKPT_PATH="/projects/tir6/strubell/jaredfer/projects/SatMAE/finetune-vit-base-e7.pth"
DATA_PATH="/projects/tir6/strubell/data/fmow-sentinel"


torchrun --nproc_per_node=4 main_finetune.py \
    --dist_eval --eval --num_workers 8 --dataset spectral \
--batch_size 16 --input_size 96 --patch_size 8  \
--model_type group_c  \
--dataset_type sentinel --dropped_bands 0 9 10 \
--train_path "${DATA_PATH}/train.csv" \
--test_path "${DATA_PATH}/val.csv" \
--output_dir logs_multispectral \
--log_dir logs_multispectral \
--finetune /home/experiments/pretain/checkpoint-199.pth