#!/bin/bash
# A3 -- Penalise proto-plans that fall outside each head's intended length band.
#
# The most directly targeted of the three experiments. check_z_influence.py showed the
# actor has collapsed onto embeddings that decode to length-1 plans:
#
#   actor z: mean pairwise cosine similarity = 0.7783  (1.0 = all identical)
#   sequences from actor z: unique(trimmed) = 5/256, mean_len = 1.01
#
# while the same (trained) decoder fed *random* z produces 44 unique sequences at
# mean_len 1.77 -- so the decoder is healthy and the actor is the collapsed half. A
# length-1 plan is a safe local optimum: it cannot commit to a wrong multi-step move, so
# the critic prefers it, and the whole proto-plan idea degenerates into a 1-step policy.
#
# sequence_length_weight pushes each head toward its own band -- min_lengths_per_head
# [1,2,4,6,8], max [2,4,6,8,10] -- which attacks mean_len=1.01 head-on and forces the
# heads apart at the same time.
#
# Requires n_proto_plan_candidates 5: the penalty asserts len(min_lengths_per_head) ==
# n_heads, and the five-element defaults imply 5 was the intended configuration.
#
# NOTE this experiment only works because of the floor fix in
# compute_sequence_lengths_differentiable -- torch.floor has zero gradient, so before that
# fix this penalty was logged but contributed nothing to learning.
#
# WEIGHT SCALING: the penalty is a *sum* over batch x heads = 256 x 5 = 1280 terms, each
# up to ~8 when plans are length 1. So the raw penalty is in the thousands while
# actor_loss is ~0.4 -- hence 1e-4, which lands the contribution in the same order as the
# Q term. Watch losses/sequence_length_loss in wandb: multiply it by this weight and
# compare against losses/actor_loss. If it dwarfs the Q term the agent will optimise
# length instead of success; if it is negligible, raise it.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a3_length
#SBATCH --output=sbatch/logs/a3_length-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --no-torch_deterministic \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --total_timesteps 500000 \
    --total_steps_e 15000 \
    --n_proto_plan_candidates 5 \
    --sequence_length_weight 0.0001 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
