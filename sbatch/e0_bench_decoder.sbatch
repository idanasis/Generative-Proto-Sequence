#!/bin/bash
# E0 -- Decoder microbenchmark. Answers: how expensive is one gen_action_seq call,
# for the legacy MLP decoder vs the minGPT decoder at several trunk sizes?
#
# This is the cheapest and most decisive experiment: it isolates the per-call constant
# factor with no RL loop, no environment and no dataset. Run this first.
#
#   sbatch sbatch/e0_bench_decoder.sbatch
#
#SBATCH --partition=main
#SBATCH --time=0-00:20:00
#SBATCH --job-name=e0_bench_decoder
#SBATCH --output=sbatch/logs/e0_bench_decoder-%J.out
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
# Pinned to the node the first E0 ran on, so the new rows are directly comparable with
# the 11.538 ms eager measurement rather than against different silicon. Drop this if it
# queues too long, but then re-measure the eager rows alongside the new ones.
#SBATCH --nodelist=ise-pheno-02

module load anaconda
source activate gps_env

echo "=== E0: decoder microbenchmark ==="
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

poetry run python bench_decoder.py
