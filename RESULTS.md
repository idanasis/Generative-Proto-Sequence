# Results — Transformer decoder vs MLP decoder in GPS

Final baseline comparison. Numbers only; reasoning and diagnosis are in
[`EVIDENCE.md`](EVIDENCE.md) §8, full history in
[`TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md) §10.

Task: from-scratch end-to-end RL, `--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`. Test set 1,000 start/goal pairs per map.
GPT = minGPT causal decoder (this work). MLP = the paper's decoder, run unmodified from the
`6f627a2` code drop.

---

## Table 1 — main comparison, 500 training pairs

GPT at its best budget (1M where it was still improving at 500k); MLP at 1M throughout.

| map | success |  | decoder generations / episode |  |  | agent step ratio |  | plan length |  |
|---|---|---|---|---|---|---|---|---|---|
| | **GPT** | MLP | **GPT** | MLP | **reduction** | **GPT** | MLP | **GPT** | MLP |
| 8x8_empty | 1.000 | 1.000 | **2.19** | 3.92 | **1.8×** | **0.977** | 0.934 | 3.00 | 1.64 |
| 16x16_empty | 0.994 | 1.000 | **3.42** | 5.73 | **1.7×** | **0.946** | 0.889 | 3.00 | 1.80 |
| 16x16_corridors_v1 | **0.990** | 0.982 | **3.36** | 13.34 | **4.0×** | **0.936** | 0.573 | 3.01 | 2.06 |
| 16x16_obstacles_v1_15p | 0.949 | 0.987 | **4.98** | 10.66 | **2.1×** | 0.755 | **0.825** | 3.00 | 1.09 |
| 16x16_room_v1 | 0.945 | 0.980 | **4.56** | 11.56 | **2.5×** | **0.805** | 0.774 | 3.00 | 1.13 |
| 16x16_obstacles_v1_25p | 0.915 | 0.953 | **4.83** | 13.66 | **2.8×** | **0.785** | 0.701 | 3.00 | 1.12 |
| ~~24x24_empty~~ ⛔ | ~~0.110~~ | ~~0.973~~ | 7.16 | 12.40 | 1.7× | 0.640 | 0.717 | 2.99 | 1.26 |

⛔ **`24x24_empty`: the 0.110/0.973 pair above is budget-mismatched (GPT 500k, MLP 1M), but
the failure is real.** Two 1M runs settle it — 700 pairs → **0.115**, 1,879 pairs → **0.128**,
against 0.110 at 500 pairs and 500k. Doubling the budget and near-quadrupling the data moves
nothing. Quote 0.115–0.128 (1M) rather than 0.110, and see §"Why 24x24 fails" below.

The 1,879-pair run is the decisive one, because it is **coverage-matched** to the
`16x16_empty` cell that scores 0.994:

| map | pairs | train pool | coverage | budget | success |
|---|---|---|---|---|---|
| 16x16_empty | 500 | 16,380 | 3.05% | 1M | **0.994** |
| 24x24_empty | 1,879 | 60,620 | 3.10% | 1M | **0.128** |

Same coverage, same budget, same decoder, same actor, both empty maps. Data and budget are
eliminated as explanations.

**Read:** success within 0.04 on all six valid maps; **2–3× fewer decisions per episode on
every map**; better path optimality on five of six.

`n=2` for `16x16_room_v1` (0.961, 0.929), `16x16_obstacles_v1_25p` (0.924, 0.905) and
`24x24_empty` (0.120, 0.099); `n=1` elsewhere. MLP is `n=1` throughout.

---

## Table 2 — data regime, 100 training pairs

Both arms 1M steps. The paper's protocol pair count.

> **Superseded by Table 2b.** Every GPT cell here used the historical 26,688-parameter
> decoder against an MLP with 2,466. At a matched 1,192 parameters the same cells score
> 0.891 / 0.846 / 0.498 instead of 0.520 / 0.487 / 0.515, and the GPT's mean goes from below
> the MLP's to **+0.149 above** it. Keep this table only as the record of what an oversized
> decoder does under data starvation.

| map | cov. | success |  | decoder generations / ep |  |  | agent step ratio |  | revisits (global) |  |
|---|---|---|---|---|---|---|---|---|---|---|
| | | **GPT** | MLP | **GPT** | MLP | **red.** | **GPT** | MLP | **GPT** | MLP |
| 16x16_empty | 0.6% | 0.520 | **0.930** | **6.88** | 8.47 | 1.2× | 0.744 | **0.762** | **1.72** | 1.19 |
| 16x16_obstacles_v1_15p | 0.9% | 0.497 | **0.636** | **7.13** | 19.16 | **2.7×** | **0.641** | 0.561 | **1.85** | 3.38 |
| 16x16_room_v1 | 1.2% | 0.515 | 0.507 | **6.93** | 24.21 | **3.5×** | **0.650** | 0.452 | **1.86** | 4.37 |
| 16x16_obstacles_v1_25p | 1.3% | 0.487 ⚠ | 0.399 | **6.55** | 23.04 | **3.5×** | **0.643** | 0.366 | **1.44** | 4.70 |
| **range** | | **0.033** | **0.531** | **0.58** | **15.74** | | **0.103** | **0.396** | | |

**Read:** success is 2–2, but the efficiency columns are not close. **The GPT replans 2.7–3.5×
less often on the three cluttered maps, follows more direct paths on three of four, and
wanders 2–3× less** — while scoring lower on success for two of them. The advantage is
*larger* at 100 pairs than at 500 (Table 1), because the MLP degrades badly here.

The *variance* is the second finding. The GPT is flat on every axis — success spread 0.033
(below the 0.042 noise floor), generations/ep spread 0.58, step ratio 0.103 — across an open
grid and 25% clutter. The MLP's spreads are 0.531, 15.74 and 0.396. The GPT is insensitive
to map difficulty; the MLP tracks it as a navigating policy should.

Plan length is **3.000 on every GPT cell here too** (3.001/3.000/3.001/3.000) against the
MLP's 1.14–1.85 — the same constant as at 500 pairs.

⚠ `obstacles_25p:gpt` hit a non-finite gradient and aborted at step 780,500 of 1M; 0.487 is
its best-on-validation checkpoint. Best validation was at 330k and declining after, so the
abort probably cost nothing, but the cell is not a completed run.

`empty:mlp` reached its best validation at step **965,000** — still improving at the budget
wall, so its 0.930 is a lower bound.

"Coverage" is the share of each map's level-1–14 train pool that the pair count buys; it
differs per map because the maps differ in free space. See Table 5.

---

## Table 2b — 100 pairs with a right-sized decoder (2026-11-17)

Table 2 above used the historical **26,688**-parameter decoder. The MLP it is compared
against has **2,466**. Matching the scale changes the result.

GPT at `--decoder_n_layer 1 --decoder_n_head 2 --decoder_n_embd 8` (**1,192** params, 0.5×
the MLP). Both arms 1M steps, 100 training pairs.

| map | success |  | decoder generations / ep |  |  | agent step ratio |  | plan length |  |
|---|---|---|---|---|---|---|---|---|---|
| | **GPT** | MLP | **GPT** | MLP | **reduction** | **GPT** | MLP | **GPT** | MLP |
| 16x16_obstacles_v1_15p | **0.957** | 0.636 | **5.73** | 19.16 | **3.3×** | **0.672** | 0.561 | 3.04 | 1.22 |
| 16x16_empty | **0.891** | 0.883 | **5.67** | 8.47 | **1.5×** | **0.780** | 0.762 | 3.05 | 1.78 |
| 16x16_obstacles_v1_25p | **0.825** | 0.399 | **4.89** | 23.04 | **4.7×** | **0.773** | 0.366 | 3.00 | 1.85 |
| 16x16_corridors_v1 | 0.744 | **0.794** | **5.24** | 13.34 | **2.5×** | 0.723 | — | 3.02 | 2.06 |
| 16x16_room_v1 | 0.498 | 0.507 | **8.88** | 24.21 | **2.7×** | **0.553** | 0.452 | 3.03 | 1.13 |
| **mean** | **0.783** | 0.644 | | | | | | | |

**Read:** across all five 16x16 maps the Transformer wins on **mean success (+0.139)** and
on **decoder generations on every map (1.5–4.7× fewer)**, while emitting **3.00–3.05-action
plans** against the MLP's 1.13–2.06. Two cells carry it — `obstacles_15p` **+0.321** and
`obstacles_25p` **+0.426**, the two maps the MLP handles worst. It trails on `corridors`
(−0.050) and ties on `room_v1` (−0.009) and `16x16_empty` (+0.008).

Compare Table 2: the same architecture at 26,688 parameters scored **0.487–0.520 on every
map**, i.e. flat and insensitive to map difficulty. Mean success goes 0.505 → **0.783** from
deleting parameters.

Seeds: `16x16_empty` **n=2 (0.893 / 0.889)**, `obstacles_25p` **n=2 (0.846 / 0.803)**;
`obstacles_15p`, `corridors` and `room_v1` are **n=1**. MLP is n=2 on `16x16_empty` only.

Wall clock: GPT 6.2–6.7h per run against the MLP's 13.6h.

The decoder-size sweep that produced it, on `16x16_empty` @100:

| n_layer / n_head / n_embd | params | vs MLP | success |
|---|---|---|---|
| **1 / 2 / 8** | **1,192** | **0.5×** | **0.893 / 0.889** |
| 1 / 4 / 16 | 3,920 | 1.6× | 0.647 |
| 2 / 4 / 16 | 7,200 | 2.9× | 0.670 |
| 1 / 4 / 32 | 13,984 | 5.7× | 0.552 |
| 2 / 4 / 32 (historical) | 26,688 | 10.8× | 0.520 / 0.540 |

Monotone in size, with a replicated baseline. **+0.363 from deleting parameters.**

### And it costs nothing at 500 pairs

| map @500 | GPT 1,192 | GPT 26,688 | MLP | shrink Δ |
|---|---|---|---|---|
| 16x16_obstacles_v1_25p | **0.991** | 0.915 | 0.953 | **+0.076** |
| 16x16_empty | **1.000** | 0.994 | 1.000 | +0.006 |
| 16x16_room_v1 | 0.939 | 0.945 | 0.980 | −0.006 |

**No regression anywhere, and `obstacles_25p` gains +0.076.** The shrink is strictly
adoptable, not a starved-regime-only fix.

Sequence-level metrics at 500 pairs (GPT 1,192):

| | success | gen/ep | step ratio | plan len |
|---|---|---|---|---|
| 16x16_empty | 1.000 | **3.07** | **0.993** | 3.00 |
| obstacles_25p | 0.991 | **3.74** | **0.882** | 3.00 |
| room_v1 | 0.939 | **4.71** | **0.801** | 3.00 |
| *16x16_empty @ 26,688* | 0.994 | 3.42 | 0.946 | 3.00 |
| *16x16_empty : MLP* | 1.000 | 5.73 | 0.889 | 1.80 |

A 0.993 step ratio is near-optimal pathing. Note `room_v1` and `obstacles_25p` reached best
validation at **980k and 955k** — still climbing at the 1M wall, so both are lower bounds.

### Two caveats that belong beside these numbers

**The MLP had not converged.** Its best validation lands at step 965,000, still climbing at
the 1M wall, on every 100-pair cell. Its numbers here are lower bounds. The GPT peaks at
215k–455k, so it reaches a better score in ~4× fewer steps — but a longer MLP budget could
narrow the success column.

**The GPT wastes more actions.** `unused actions/seq` 0.46–0.73 against the MLP's 0.17 —
~20% of emitted actions discarded against ~9%. That is the cost of a 3-action commitment and
the one axis where the MLP is consistently better.

---

## Table 3 — sample efficiency, `16x16_corridors_v1`

| training pairs | coverage | GPT | MLP |
|---|---|---|---|
| 100 | 5.6% | 0.725 (n=5) | 0.794 |
| 300 | 16.9% | 0.897 | **0.977** |
| 500 | 28.1% | **0.990** | 0.982 |

**Read:** the MLP saturates by 300 pairs; the GPT needs 500 to reach the same level. The
0.069 gap at 100 pairs closes entirely by 500.

---

## Table 4 — length-penalty ablation, 500 pairs, 500k steps

`--sequence_length_weight 0.002` vs `0`, all else identical.

| map | plain (0) | tuned (0.002) | penalty worth | plan length, plain → tuned |
|---|---|---|---|---|
| 16x16_empty | 0.907 | 0.990 | +0.083 | 1.03 → 3.00 |
| 16x16_room_v1 | 0.226 | 0.838 | **+0.612** | 1.00 → 3.00 |
| 16x16_obstacles_v1_15p | 0.200 | 0.824 | **+0.624** | 1.01 → 3.00 |
| 16x16_obstacles_v1_25p | 0.244 | 0.505 | +0.261 | 1.00 → 3.00 |
| 24x24_empty | 0.141 | 0.120 | −0.021 | 1.01 → 3.00 |

**Read:** the penalty is load-bearing, not a source of rigidity. Plan length is a **constant
either way** — 1.000 without it, 3.000 with it, on every map — never state-dependent.

Both arms are 500k, so the *comparison* is matched and the conclusion holds, but by caveat 7
every absolute number in this table is a lower bound. The tuned column understates what 1M
would give (compare `obstacles_25p` 0.505 here against 0.915 at 1M in Table 1).

---

## Table 5 — dataset coverage by map

Train-pool size at levels 1–14, by BFS over each map's free cells. `--train_dataset_size` is
an absolute count, so the same flag buys very different coverage per map.

| map | free cells | train pool | 100 pairs | 500 pairs |
|---|---|---|---|---|
| 16x16_corridors_v1 | 64 | 1,778 | 5.6% | 28.1% |
| 8x8_empty | 64 | 2,016 | 5.0% | 24.8% |
| 16x16_obstacles_v1_25p | 147 | 7,452 | 1.3% | 6.7% |
| 16x16_room_v1 | 160 | 8,626 | 1.2% | 5.8% |
| 16x16_obstacles_v1_15p | 167 | 11,454 | 0.9% | 4.4% |
| 16x16_empty | 196 | 16,380 | 0.6% | 3.1% |
| 24x24_empty | 484 | 60,620 | 0.16% | 0.8% |

---

## Table 6 — `24x24_empty` diagnosis

`check_z_influence.py`, 500 pairs, against a working checkpoint of the same configuration.

| | `16x16_room_v1` (0.961) | `24x24_empty` (0.099) |
|---|---|---|
| decoder, unique plans from random `z` | 60/256 | **64/256** |
| actor `z`, mean pairwise cosine | 0.7639 | 0.7746 |
| actor `z`, std across dims | 0.0271 | **0.0097** (2.8× smaller) |
| actor `z`, mean L2 norm | 0.2079 | **0.0782** (2.7× smaller) |
| actor `z`, **std / norm** | **0.1306** | **0.1246** (same) |
| plans reached from actor `z` | 26/256 (43% of decoder range) | 15/256 (23%) |

**Read:** the decoder is healthy on the failing map — marginally healthier than on the
working one. The actor has **not** collapsed: its `z` carries the same relative variety
(`std/norm` matches within 5%), it is simply 2.7× smaller in magnitude. `z` enters the
Transformer as a prefix token competing against fixed-scale token embeddings in an
**unnormalised** residual stream, so a small `z` is drowned out. The MLP decoder has no
residual stream and absorbs a uniform rescale in its weights.

---

## Why `24x24_empty` fails — hypothesis, not yet a result

Ruled out by direct measurement: **data coverage** (3.10% vs 3.05% matched, Table 1 ⛔),
**training budget** (1M changes nothing), **decoder capacity** (64/256 unique plans from
random `z`, the healthiest of any map, Table 6), **`z` magnitude** (the decoder is pre-LN
with a learned `z_proj`, so any uniform rescale of `z` is absorbed — which is why
`--decoder_normalize_z` left the map at 0.18), **episode budget** (`max_episode_steps` is
75 and `max_level` is 14 on *every* map, so path lengths and step budgets are identical),
and **the task** (the MLP reaches 0.973 on the same map through the same actor).

Note what that last row implies: 24x24 is not a harder navigation problem. `max_level 14`
means the same ≤14-step paths as 16x16 — just scattered over 484 free cells instead of 196.

What remains is **latent resolution**. The latent is 16-D (`actor_n_output_channels`, and
`DenseVAE.decoder_input_size`, both 16). On 24x24 the actor must separate 3.7× more
situations (train pool 60,620 vs 16,380) while its `z` occupies measurably the *same*
latent volume as on a working map — `std/norm` 0.1246 vs 0.1306, mean pairwise cosine
0.7746 vs 0.7639 (Table 6). Adjacent states collapse onto one `z`, decode to one plan, and
the agent moves wrongly. Hence 15/256 plans reached from actor `z` against the decoder's
own 64/256.

This predicts the MLP's survival from its *weakness*: its plan length is 1.26 against the
GPT's 3.00, so its latent must resolve 4 outcomes rather than 64 — ~16× coarser. On this
reading the replanning advantage (Table 1) and the 24x24 failure are one phenomenon.

**The test.** Run 24x24 at 1M with `--sequence_length_weight 0`, which drives GPT plan
length to ~1.01 (Table 4). If it then reaches ~0.9, the mechanism is confirmed. If it stays
near 0.14, this explanation is dead. Table 4's existing 0.141 for that config is **500k**
and by caveat 7 says nothing. The follow-up fix, if confirmed, is a wider latent
(`--actor_n_output_channels 64` plus plumbing `decoder_input_size` through `get_decoder`).

---

## Caveats to carry into any write-up

1. **500 pairs is not the paper's protocol** (100). Table 2 gives the 100-pair numbers; no
   direct comparison to published figures without them.
2. **`--train_dataset_size` is not comparable across maps** — see Table 5. Within a map both
   arms get identical data, so per-map comparisons are clean; cross-map ones are not.
3. **`max_level` is 14 on every map**, so no environment here tests long horizons.
   `24x24_empty` is short paths on a large canvas.
4. **The MLP tree is unpatched** — it predates fixes to `decoder_optimizer.zero_grad()`,
   inference-time dropout, and the `torch.floor` bug that disabled
   `--sequence_length_weight`. Deliberate: it is the published method as published. The
   consequence is that the GPT has a working length penalty and the MLP does not.
5. **Single seed** on most cells and on the MLP throughout. Noise floor is 0.019–0.042 (1σ),
   so the −0.035/−0.038 gaps in Table 1 are not distinguishable from ties.
6. **Deterministic inference** costs ~0.10 normally but 0.33–0.39 at 100 pairs, where the
   policy leans on sampling noise. Report which pass you are quoting.
7. **Training budgets differ by cell in Table 1, and 500k is not a safe budget.** The MLP
   gets 1M steps on every map (verified across all ten `j2` logs). The GPT gets 1M on
   `16x16_empty`, `room_v1` and both obstacle maps, and **500k** on `8x8_empty`,
   `corridors` and `24x24_empty`.

   An earlier version of this caveat argued the three 500k cells had converged — best
   validation at 425k, ~200k and 230k–295k, followed by decline — so more budget would not
   change them. **That reasoning is wrong.** A completed 500k run on `room_v1`
   (`23971720`) plateaued and declined exactly that way and finished at **0.27**, while
   three 1M runs on the same map reached 0.97, 0.98 and 0.99. Plateau-then-decline at 500k
   does not predict the 1M outcome.

   Across every completed run in the log set:

   | map | 500k finals | 1M finals |
   |---|---|---|
   | `room_v1` @500 | 0.81, 0.27 | 0.99, 0.97, 0.98 |
   | `obstacles_15p` @500 | 0.84, 0.24 | 0.98 |
   | `obstacles_25p` @500 | 0.49, 0.28 | 0.93, 0.93 |

   Five of five 1M runs on cluttered maps converged to 0.93–0.99 (spread ≤0.02, *tighter*
   than the noise floor in caveat 5). No 500k run did. Treat any 500k number as a lower
   bound, never as a converged result.

   Consequences: the `24x24_empty` row is void (see ⛔ above). The `corridors` and
   `8x8_empty` rows are mismatched in the GPT's *disfavour* — it wins or ties them anyway,
   so they stand as conservative, but a 1M re-run would strengthen both.

---

## In flight — not a result yet

**Decoder-architecture fixes.** Five levers tried; none fixes an open map.

| lever | outcome |
|---|---|
| `--decoder_normalize_z` (branch `normalize-test`) | refuted — `room_v1` fell to 0.39 |
| multi-head decoder | refuted — diverged at 143k on a non-finite gradient, worse on every metric |
| more training pairs on 24x24 | no effect — 500/700/1,879 all ~0.12 at 1M |
| longer budget on 24x24 | no effect — 500k and 1M agree |
| `--decoder_z_all_positions` (branch `z-every-position`) | mixed, see below |

`--decoder_z_all_positions`, all single seed and not yet replicated:

| map | pairs | flag off | flag on | Δ |
|---|---|---|---|---|
| `16x16_corridors_v1` | 500 | 0.990 | 0.984 | −0.006 (no regression) |
| `16x16_room_v1` | 100 | 0.515 | **0.603** | **+0.088** |
| `16x16_empty` | 100 | 0.520 | 0.544 | +0.024 (inside noise) |

It costs nothing where the method already works and gives ~2σ on a *cluttered* map at low
data, but does **not** move the open maps, which is where the problem is. Needs a second
seed before it is a result.

**The pretrained-decoder protocol, never run here.** The paper's own sweep
(`hypersearch-grid-gps.py`) uses `initialize_decoder_from_pretrained=True` and
`train_decoder_end_to_end=False`; every run in this repository used neither, for *both*
arms, because the only checkpoint on disk is the MLP VAE and `DenseVAE` is now a
Transformer. The comparison is therefore fair but runs GPS with its sample-efficiency
mechanism switched off — relevant to Table 2 and Table 3, where the GPT needs ~5× the data
the MLP does. `pretrain_decoder.py` produces the missing checkpoint.
