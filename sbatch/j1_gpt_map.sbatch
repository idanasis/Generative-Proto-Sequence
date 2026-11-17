#!/bin/bash
# J1 -- Transformer decoder on any map.  Parameterised: the map is an argument.
#
#   sbatch sbatch/j1_gpt_map.sbatch <map> [length_weight] [train_pairs] [total_steps] \
#                                   [z_all_positions] [n_layer] [n_head] [n_embd]
#
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p                     # tuned (0.002)
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p 0                   # plain, no penalty
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p 0.002 500 1000000   # 1M, matches the MLP
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_empty 0.002 100 1000000 0 1 4 16     # small decoder
#
# The paper evaluates 9 environments; this repo ships 7 of them (both 24x24 obstacle maps
# are absent):
#   8x8_empty  16x16_empty  16x16_room_v1  16x16_corridors_v1
#   16x16_obstacles_v1_15p  16x16_obstacles_v1_25p  24x24_empty
#
# TWO CONFIGS PER MAP, and both are needed:
#   weight 0.002 (default) -> our tuned method
#   weight 0               -> no length penalty, which is what the paper's MLP effectively
#                             ran, because its torch.floor bug silently disabled
#                             --sequence_length_weight. Only this one is a like-for-like
#                             decoder comparison against j2.
#
# Reporting only the tuned number against the MLP would credit the Transformer for a dead
# feature we switched on rather than for the architecture.
#
# Everything else matches the best-known config (d6). 100 training pairs, matching the
# paper's protocol -- note that on corridors this was badly data-starved (0.725 at 100
# pairs, 0.99 at 500), so expect every map here to understate what the setup can reach.
#
# CAVEAT worth checking before trusting 24x24 or 8x8: max_level defaults to 14 and
# n_actions_in_seq to 10. Those were sensible for 16x16. On 24x24 optimal paths are longer,
# so a 10-action plan covers proportionally less ground; on 8x8 the level range may clip.
# Compare "mean num decoder generations" and "agent step ratio" across maps -- if 24x24
# needs far more generations per episode, the plan length is the limit, not the decoder.
#
#SBATCH --partition=main
# 1 day, not 12h: a 500k run takes 3-7h depending on the map, so a 1M run on the slower
# maps (obstacles_15p was 5.45h at 500k) would be killed at a 12h wall mid-training.
#SBATCH --time=1-00:00:00
#SBATCH --job-name=j1_gpt_map
#SBATCH --output=sbatch/logs/j1_gpt_map-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

MAP="$1"
LENGTH_WEIGHT="${2:-0.002}"
# 500 pairs, not the paper's 100. At 100 both decoders are data-starved (0.725 vs 0.794 on
# corridors) and the comparison measures starvation; at 500 both sit at the success ceiling
# (0.99 vs 0.982) and the difference is purely sequence-level efficiency -- 3.36 vs 13.34
# decoder generations per episode, 0.936 vs 0.573 agent step ratio. That is the paper's
# actual claim, and it only becomes visible once accuracy saturates.
TRAIN_PAIRS="${3:-500}"
# 500k was validated on corridors, where validation peaks around 200k. It does NOT transfer:
# on room_v1 and obstacles_25p the best validation landed at step 499,999 -- still improving
# when the budget ran out -- and obstacles_15p at 470,000. Those three need 1M to be
# comparable against the MLP, which gets 1M everywhere. Check "best val mean global step" in
# the log: within ~10% of the budget means the run was cut short and its score is a floor.
TOTAL_STEPS="${4:-500000}"
# 5th arg: pass 1 to enable --decoder_z_all_positions. Single-flag A/B against the matching
# run without it.
#
# Why: z enters the decoder once, as the prefix token, so its influence on a late token is
# mediated by the accumulating token history and small differences between similar z wash
# out. The MLP decoder had no such bottleneck -- one linear layer mapped z to all 50 logits.
# Measured: the decoder gives 60-64 distinct plans out of 256 from a wide random z but only
# 15-26 from the actor's real z (std/norm ~= 0.12, nearly constant). Open maps suffer most,
# because they need the most distinct plans; corridors suffers least, which is why the
# method looked fine there.
#
# FIRST TEST: 16x16_empty at 100 pairs, where the gap is 0.520 vs the MLP's 0.930 and the
# plan-repertoire limit should bind hardest. Mechanism check afterwards with
# check_z_influence.py -- "plans reached from actor z" should rise above 15-26/256.
#
# Adds no parameters (reuses z_proj). Verified off is bit-identical to the historical path,
# and that cached decode matches the full forward when on (6e-08).
Z_ALL_POSITIONS="${5:-0}"

# 6th-8th args: decoder trunk shape. Defaults 2/4/32 reproduce every run before 2026-11-15
# bit-for-bit, so this is a no-op unless set.
#
# Why: the decoder we are replacing has 2,466 parameters. Ours has 26,688 -- 10.8x larger --
# and needs ~5x the training pairs to match it (Table 3: MLP saturates at 300 pairs, GPT
# needs 500). Those are plausibly the same fact: more parameters fitted from the same RL
# signal. Nothing has ever tested a smaller trunk.
#
#   n_layer n_head n_embd   params   vs MLP
#         2      4     32   26,688    10.8x   <- current
#         1      4     32   13,984     5.7x
#         2      4     16    7,200     2.9x
#         1      4     16    3,920     1.6x
#         1      2      8    1,192     0.5x
#
# n_head must divide n_embd. Test on 16x16_empty at 100 pairs, where the gap to the MLP is
# widest in the whole result set (0.520 vs 0.930) and a data-efficiency effect should bind
# hardest.
DECODER_N_LAYER="${6:-2}"
DECODER_N_HEAD="${7:-4}"
DECODER_N_EMBD="${8:-32}"

# 9th arg: path to a decoder checkpoint from pretrain_decoder.py. Empty (the default) keeps
# the from-scratch end-to-end path every run before 2026-11-16 used.
#
# This is THE PAPER'S PROTOCOL and it has never been run in this repo. hypersearch-grid-gps.py
# uses initialize_decoder_from_pretrained=True and train_decoder_end_to_end=False -- a
# pretrained, frozen decoder -- while every run here used neither, for BOTH arms, because the
# only checkpoint on disk was the MLP VAE and DenseVAE is now a Transformer. The comparison
# was fair but ran GPS with its sample-efficiency mechanism switched off.
#
# The decoder SHAPE must match the checkpoint or load_state_dict rejects it. Pretrain the
# matching shape first:
#   sbatch sbatch/p1_pretrain_decoder.sbatch 5000 1 2 8
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_empty 0.002 100 1000000 0 1 2 8 \
#       generative/models/dense_autoencoder/gptVAE_l1_h2_e8_in10_lat16_var1_ep5000_seed42.pt
#
# 10th arg: 1 (default) freezes the decoder -- the paper's setting. 0 fine-tunes it from the
# pretrained init, which is a different experiment: it tests whether pretraining helps as an
# initialisation, separately from whether freezing prevents the actor/decoder co-adaptation
# that pins plan length to a constant (EVIDENCE.md §11.3).
DECODER_CKPT="${9:-}"
DECODER_FREEZE="${10:-1}"

if [ -z "${MAP}" ]; then
    echo "ERROR: pass the obstacle map, e.g."
    echo "  sbatch $0 16x16_obstacles_v1_25p"
    echo "  sbatch $0 16x16_obstacles_v1_25p 0        # no length penalty"
    echo "  sbatch $0 16x16_obstacles_v1_25p 0.002 100  # paper's 100-pair protocol"
    exit 1
fi

# The length-band flags must be omitted entirely when the weight is 0: passing them with
# the penalty disabled is harmless but misleading in the logged config.
ZPOS_ARGS=()
if [ "${Z_ALL_POSITIONS}" != "0" ]; then
    ZPOS_ARGS=(--decoder_z_all_positions)
fi

# Default: from scratch, end-to-end. With a checkpoint: pretrained init, frozen or not.
PRETRAIN_ARGS=(--train_decoder_end_to_end --no-initialize_decoder_from_pretrained)
if [ -n "${DECODER_CKPT}" ]; then
    if [ ! -f "${DECODER_CKPT}" ]; then
        echo "ERROR: decoder checkpoint not found: ${DECODER_CKPT}"
        exit 1
    fi
    if [ "${DECODER_FREEZE}" != "0" ]; then
        PRETRAIN_ARGS=(--initialize_decoder_from_pretrained --no-train_decoder_end_to_end)
    else
        PRETRAIN_ARGS=(--initialize_decoder_from_pretrained --train_decoder_end_to_end)
    fi
    PRETRAIN_ARGS+=(--decoder_model_path "${DECODER_CKPT}")
fi

LENGTH_ARGS=()
if [ "${LENGTH_WEIGHT}" != "0" ]; then
    LENGTH_ARGS=(--sequence_length_weight "${LENGTH_WEIGHT}"
                 --min_lengths_per_head 4 --max_lengths_per_head 10)
fi

echo "=== J1: map=${MAP}  length_weight=${LENGTH_WEIGHT}  train_pairs=${TRAIN_PAIRS}  total_steps=${TOTAL_STEPS}  z_all_positions=${Z_ALL_POSITIONS}  decoder=${DECODER_N_LAYER}L/${DECODER_N_HEAD}H/${DECODER_N_EMBD}E  pretrained=${DECODER_CKPT:-none}  freeze=${DECODER_FREEZE} ==="

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --no-torch_deterministic \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    "${PRETRAIN_ARGS[@]}" \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --total_timesteps "${TOTAL_STEPS}" \
    --total_steps_e 15000 \
    --n_proto_plan_candidates 1 \
    --min_qf_value -20.0 \
    --max_qf_value 1.0 \
    --decoder_n_layer "${DECODER_N_LAYER}" \
    --decoder_n_head "${DECODER_N_HEAD}" \
    --decoder_n_embd "${DECODER_N_EMBD}" \
    "${LENGTH_ARGS[@]}" \
    "${ZPOS_ARGS[@]}" \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
