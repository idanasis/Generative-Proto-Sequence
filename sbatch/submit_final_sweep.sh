#!/bin/bash
# Final sweep at the right-sized decoder (1,192 params = 1L/2H/8E).
#
#   bash sbatch/submit_final_sweep.sh
#
# WHY THIS EXISTS
# ---------------
# Every table in RESULTS.md except Table 2b was measured with a 26,688-parameter decoder
# against an MLP with 2,466 -- 10.8x oversized. EVIDENCE.md 11.1/11.2 show that sizing it
# correctly is worth +0.33 to +0.89 depending on the map, and turns `24x24_empty` from the
# project's one catastrophic failure (0.110) into 1.000, beating the MLP's 0.973.
#
# Table 1, the main 7-map 500-pair comparison, is therefore still measured on the wrong
# decoder. This batch rebuilds it. Jobs beyond the 5-slot QOSMaxGRESPerUser limit queue as
# PD and start as slots free, so submission order IS priority order.
#
# Already done at 1,192 / 500 pairs (all n=1): 24x24_empty 1.000, obstacles_25p 0.991,
# 16x16_empty 1.000, room_v1 0.939.
#
# HOW TO READ THE RESULT when it lands:
#   module load anaconda && source activate gps_env
#   poetry run python compare_runs.py sbatch/logs/j1_gpt_map-*.out sbatch/logs/j2_mlp_map-*.out
# Check `valid actions/seq` stays near 3.0 on every cell -- a cell that wins by collapsing
# to ~1-action plans is not a sequence-level result (EVIDENCE.md 9).

set -u
G="sbatch sbatch/j1_gpt_map.sbatch"
M="sbatch sbatch/j2_mlp_map.sbatch"
DEC="1 2 8"          # 1,192 params; args 6-8 of j1
STD="0.002 500 1000000 0"   # length_weight, pairs, steps, z_all_positions

echo "== TIER 1: complete the 500-pair sweep at 1,192 =="
# Three maps never run at this size, plus a second seed on the headline.
$G 24x24_empty            $STD $DEC   # seed 2 -- 1.000 is currently n=1
$G 16x16_corridors_v1     $STD $DEC   # never run at 1,192; 0.990 at 26,688
$G 16x16_obstacles_v1_15p $STD $DEC   # never run at 1,192; 0.949 at 26,688
$G 8x8_empty              $STD $DEC   # never run at 1,192; 1.000 at 26,688 (saturated)

echo "== TIER 2: second seeds on the existing 1,192 cells =="
# room is the ONLY cell where 1,192 scored below 26,688 (0.939 vs 0.945). Worth knowing
# whether that is real before it goes in a table as a regression.
$G 16x16_room_v1          $STD $DEC
$G 16x16_obstacles_v1_25p $STD $DEC
$G 16x16_empty            $STD $DEC   # at 1.000, saturated -- lowest value of the tier

echo "== TIER 3: second seeds for the Tier 1 maps =="
$G 16x16_corridors_v1     $STD $DEC
$G 16x16_obstacles_v1_15p $STD $DEC
$G 8x8_empty              $STD $DEC

echo "== TIER 4: 24x24 at the paper's 100-pair protocol =="
# Never run on this map for either arm. If the GPT holds up at 0.16% coverage it is a much
# stronger claim than the 500-pair cell; the MLP arm is required for it to mean anything.
$G 24x24_empty 0.002 100 1000000 0 $DEC
$G 24x24_empty 0.002 100 1000000 0 $DEC
$M 24x24_empty 100 1
$M 24x24_empty 100 1

echo "== TIER 5: MLP second seeds where the comparison is tightest =="
# The MLP column is n=1 on most cells. These two decide the headline (24x24) and the one
# map the GPT trails (corridors). Third arg 1 = --no-torch_deterministic; WITHOUT it a
# re-run of j2 reproduces the same run bit-for-bit and is not a seed (EVIDENCE.md 11.5).
$M 24x24_empty        500 1
$M 16x16_corridors_v1 500 1

echo
echo "Submitted. 16 GPU jobs; ~7h each on 16x16, ~10h on 24x24, 5 at a time => ~1.5 days."
echo
echo "SEPARATELY, costs no GPU slot -- pretrain the matching decoder shape so the paper"
echo "protocol (pretrained + frozen, j1 args 9-10) is runnable when you are back:"
echo "    sbatch sbatch/p1_pretrain_decoder.sbatch 5000 1 2 8"
echo "That path has never been exercised end-to-end, so run ONE job against it and check"
echo "it starts before committing slots to it."
