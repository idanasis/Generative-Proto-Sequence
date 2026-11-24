#!/bin/bash
# J2 -- MLP decoder (paper code, unmodified) on any map.  The baseline for J1.
#
#   sbatch sbatch/j2_mlp_map.sbatch <obstacle_map>
#   sbatch sbatch/j2_mlp_map.sbatch 16x16_obstacles_v1_25p
#
# Runs against the git worktree at 6f627a2 -- the paper's code drop, identical to
# origin/main. Nothing patched, including the torch.floor bug that disables
# --sequence_length_weight. That is deliberate: this is the published method as published,
# which is what J1-with-weight-0 is meant to be compared against.
#
# Create the worktree once, from the repo root:
#   git worktree add ~/gps-mlp-baseline 47d6da4^
#
# Safe to run several of these concurrently, and alongside the Transformer runs: they only
# read the source, and every output path is keyed by the unique wandb run name.
#
# 1M steps, not 500k. On corridors the MLP's best validation landed at step 999,999 -- still
# improving when its run ended, unlike the Transformer which peaks around 200k. Giving it
# less than the full budget would handicap it. Expect ~10-11h per map.
#
# ASYMMETRY TO STATE WHEN REPORTING: this tree also predates the sec. 7-9 fixes, so it runs
# with an accumulating decoder gradient (decoder_optimizer.zero_grad() missing) and
# inference-time dropout. Both were measured to buy ~0 on their own, and the MLP has no
# dropout layers at all so the second does not apply to it -- but the first does.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=j2_mlp_map
#SBATCH --output=sbatch/logs/j2_mlp_map-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

MAP="$1"
TRAIN_PAIRS="${2:-500}"
# 3rd arg: pass 1 for an INDEPENDENT SAMPLE. Default 0 reproduces every historical j2 run.
#
# Config defaults are seed=123 and torch_deterministic=True, and this script passes neither,
# so re-running it unchanged reproduces the same run bit-for-bit -- it is NOT a second seed.
# j1 hardcodes --no-torch_deterministic, which is why the GPT cells in RESULTS.md show seed
# spread (room_v1: 0.961/0.929) while the MLP column is n=1 everywhere.
#
# That single-seed MLP column is the weakest thing in the comparison: the four cells where
# the GPT trails do so by 0.006-0.038, and a 1-seed baseline cannot distinguish that from a
# tie. Pass 1 here to get the replicate.
NONDETERMINISTIC="${3:-0}"

# 4th-7th args: the paper's protocol knobs. Defaults reproduce every historical j2 run.
#
#   j2 <map> [pairs] [nondeterministic] [total_steps] [start_level] [max_level] [paper_decoder]
#   j2 16x16_room_v1 100 1 1500000 20 30 1      <- the paper's actual configuration
#
# hypersearch-grid-gps.py:66-68,92,95 gives the published protocol as total_timesteps
# 1500000, start_level 20, max_level 30, train_decoder_end_to_end False and
# initialize_decoder_from_pretrained True. Every run in this repo used 1M steps, levels
# 1-14 and a from-scratch end-to-end decoder, for BOTH arms. The comparison is internally
# fair but **no number in RESULTS.md can be set against a published figure**.
#
# paper_decoder=1 flips the MLP to pretrained + frozen. Its checkpoint has shipped since
# 6f627a2 (denseAE_generic_2act_seq_VAE_v7_...pt, 2,466-param decoder) and is the Config
# default for --decoder_model_path, so no path argument is needed. The GPT side needs a
# matching checkpoint from pretrain_decoder.py, passed as j1's 9th argument.
#
# 8x8_empty cannot run levels 20-30: its diameter is 14.
TOTAL_STEPS="${4:-1000000}"
START_LEVEL="${5:-1}"
MAX_LEVEL="${6:-14}"
PAPER_DECODER="${7:-0}"
WORKTREE="${HOME}/gps-mlp-baseline"

if [ -z "${MAP}" ]; then
    echo "ERROR: pass the obstacle map, e.g."
    echo "  sbatch $0 16x16_obstacles_v1_25p"
    exit 1
fi

if [ ! -d "${WORKTREE}" ]; then
    echo "ERROR: ${WORKTREE} does not exist. Create it first, from the repo root:"
    echo "  git worktree add ${WORKTREE} 47d6da4^"
    exit 1
fi

# The worktree must hold the MLP decoder. If this finds mingpt it is at the wrong commit and
# the run would silently repeat the Transformer experiment under an MLP job name.
if grep -q "mingpt_decoder" "${WORKTREE}/generative/models/dense_autoencoder/dense_var_auto_encoder_vin_1.py"; then
    echo "ERROR: ${WORKTREE} contains the Transformer decoder, not the MLP."
    echo "       Check it out at 47d6da4^ (= 6f627a2, the paper's code drop)."
    exit 1
fi

DET_ARGS=()
if [ "${NONDETERMINISTIC}" != "0" ]; then
    DET_ARGS=(--no-torch_deterministic)
fi

# Default: from scratch, end-to-end -- as every historical j2 run.
PRETRAIN_ARGS=(--train_decoder_end_to_end --no-initialize_decoder_from_pretrained)
if [ "${PAPER_DECODER}" != "0" ]; then
    PRETRAIN_ARGS=(--initialize_decoder_from_pretrained --no-train_decoder_end_to_end)
fi

cd "${WORKTREE}" || exit 1
echo "=== J2: map=${MAP}  train_pairs=${TRAIN_PAIRS}  nondeterministic=${NONDETERMINISTIC}  total_steps=${TOTAL_STEPS}  levels=${START_LEVEL}-${MAX_LEVEL}  paper_decoder=${PAPER_DECODER}  from $(pwd) at $(git rev-parse --short HEAD) ==="

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --total_timesteps "${TOTAL_STEPS}" \
    --start_level "${START_LEVEL}" \
    --max_level "${MAX_LEVEL}" \
    "${PRETRAIN_ARGS[@]}" \
    "${DET_ARGS[@]}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
