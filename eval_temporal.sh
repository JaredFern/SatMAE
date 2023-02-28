#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36]
#SBATCH --time 1-00:00:00

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --batch_size 16 \
    --model vit_large_patch16 \
    --model_type temporal \
    --resume <PATH_TO_YOUR_FINEtune_CHECKPOINT>  \
    --dist_eval --eval --num_workers 8 --dataset fmow_temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv \
    --test_path <PATH_TO_DATASET_ROOT_FOLDER>/val_62classes.csv