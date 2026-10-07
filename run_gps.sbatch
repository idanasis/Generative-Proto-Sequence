#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-48:00:00 
#SBATCH --job-name=gps_mingpt_maze
#SBATCH --output=gps_maze_job-%J.out
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Load conda and activate your environment
module load anaconda
source activate gps_env

# Bypass the interactive login by setting the key directly
export WANDB_API_KEY="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"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Run your training command
poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained
