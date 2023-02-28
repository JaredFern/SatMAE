#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36],tir-1-[32,36]
#SBATCH --time 1-00:00:00

mamba init bash;
mamba activate sat_env;
DATA_DIR="/projects/tir6/strubell/data/fmow-sentinel";
OUTPUT_DIR="/projects/tir6/strubell/jaredfer/projects/SatMAE";

torchrun --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_pretrain.py \
    --batch_size 8 --accum_iter 16 \
    --norm_pix_loss --epochs 100 \
    --blr 1.5e-4 --mask_ratio 0.75 \
    --input_size 224 --patch_size 16 \
    --model mae_vit_large_patch16 \
    --model_type temporal \
    --dataset_type temporal \
    --train_path $DATA_DIR/train_62classes.csv \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --num_workers 4
