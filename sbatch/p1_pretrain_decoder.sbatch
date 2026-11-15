#!/bin/bash
# P1 -- offline pretraining of the Transformer VAE decoder.  CPU only.
#
#   sbatch sbatch/p1_pretrain_decoder.sbatch [epochs] [n_layer] [n_head] [n_embd]
#
#   sbatch sbatch/p1_pretrain_decoder.sbatch                 # 20000 epochs, 2/4/32 (current shape)
#   sbatch sbatch/p1_pretrain_decoder.sbatch 20000 1 4 16    # the smallest shrink-sweep shape
#
# HYPOTHESIS
# ----------
# GPS's sample efficiency does not come from the actor; it comes from the decoder already
# knowing what an action sequence looks like before RL starts. The paper's own sweep config
# (hypersearch-grid-gps.py) runs initialize_decoder_from_pretrained=True and
# train_decoder_end_to_end=False -- a pretrained, FROZEN decoder.
#
# Every run in this repo has used --train_decoder_end_to_end
# --no-initialize_decoder_from_pretrained, for BOTH arms, because the only checkpoint on
# disk is the MLP VAE (2,466-param decoder) and DenseVAE on this branch is a Transformer --
# its state_dict does not fit. The comparison is therefore fair but runs GPS with its
# sample-efficiency mechanism switched off. That is directly relevant to RESULTS.md Table 2
# and Table 3, where the GPT needs ~5x the training pairs the MLP does.
#
# Secondary hypothesis, and the more interesting one: end-to-end training lets the decoder
# collapse its latent space toward whatever narrow region the actor occupies. That is
# consistent with plan length being pinned to EXACTLY 3.000 on every map and every state
# (Table 1 and Table 4) against the MLP's 1.09-1.85. The pretraining corpus holds sequences
# of length 1-10, so a decoder fitted to it starts with a DISTRIBUTION over plan lengths
# rather than a single RL-induced constant. Four decoder-architecture levers failed to make
# plan length state-dependent; this attacks it from the data side instead.
#
# HOW TO READ THE RESULT
# ----------------------
# The FINAL block prints three things that matter:
#
#   val exact ~= train exact      the decoder generalises. The corpus is small (~395 unique
#                                 sequences, 276 train / 60 val), so a large gap means 276
#                                 sequences was not enough and the checkpoint is overfitted.
#
#   unique plans from random z    > 64/256 means the pretrained latent space is richer than
#                                 what end-to-end RL produced (64/256 on 24x24_empty,
#                                 60/256 on room_v1 -- RESULTS.md Table 6).
#
#   plan length ... distinct      > 1 distinct value is the thing no decoder tweak achieved.
#                                 If this collapses to a constant here too, the rigidity is
#                                 NOT caused by end-to-end training and this line is dead.
#
# Then point an RL run at the checkpoint:
#
#   python gps_simplegrid_levels.py ... \
#       --decoder_model_path <path printed by "saved:"> \
#       --initialize_decoder_from_pretrained --no-train_decoder_end_to_end
#
# Decoder shape flags MUST match between pretraining and the RL run or load_state_dict
# rejects the file. That is why n_layer/n_head/n_embd are arguments here: the shrink sweep
# in j1_gpt_map.sbatch (args 6-8) needs a matching checkpoint per shape.
#
# NO --gpus LINE ON PURPOSE. This is a 26,688-parameter model on ~276 sequences; the shipped
# MLP checkpoint was trained on CPU. Keeping it off the GPU partition means it does not eat
# one of the 5 GPU slots the QOSMaxGRESPerUser limit allows.
#
#SBATCH --partition=main
#SBATCH --time=6:00:00
#SBATCH --job-name=p1_pretrain
#SBATCH --output=sbatch/logs/p1_pretrain-%J.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

EPOCHS="${1:-20000}"
N_LAYER="${2:-2}"
N_HEAD="${3:-4}"
N_EMBD="${4:-32}"

echo "=== P1: epochs=${EPOCHS}  decoder=${N_LAYER}L/${N_HEAD}H/${N_EMBD}E ==="

START=$(date +%s)

poetry run python pretrain_decoder.py \
    --epochs "${EPOCHS}" \
    --eval_every 200 \
    --decoder_n_layer "${N_LAYER}" \
    --decoder_n_head "${N_HEAD}" \
    --decoder_n_embd "${N_EMBD}" \
    --device cpu

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
