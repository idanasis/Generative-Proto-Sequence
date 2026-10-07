# Transformer Decoder Integration

Replacing the VAE's single-shot MLP decoder with a minimal, latent-conditioned causal
Transformer (minGPT-style) and migrating the surrounding mechanics from a flattened
one-hot representation to integer-token autoregressive modeling.

Files touched:
- **New:** [mingpt_decoder.py](mingpt_decoder.py)
- **Changed:** [dense_var_auto_encoder_vin_1.py](dense_var_auto_encoder_vin_1.py)
- **Changed:** [decoder_utils.py](decoder_utils.py)
- **Changed:** [gps_simplegrid_levels.py](../../../gps_simplegrid_levels.py) (§7)

---

## 1. The new Transformer — `mingpt_decoder.py`

A small, self-contained causal decoder adapted from Andrej Karpathy's minGPT, trimmed to
the essentials and extended with **latent conditioning**.

### Classes

| Class | Role |
|---|---|
| `GPTConfig` | Dataclass of hyper-parameters. Defaults are deliberately tiny: `n_layer=2`, `n_head=4`, `n_embd=32`, `vocab_size=5`, `block_size=10`, `decoder_input_size=16` (shrunk from `n_layer=3`/`n_embd=64` — see §6.3). |
| `CausalSelfAttention` | Multi-head masked self-attention. Fused QKV projection, output projection, and a registered lower-triangular causal mask buffer sized `block_size + 1`. |
| `Block` | Pre-LayerNorm Transformer block: attention + 4×-expansion GELU MLP, each with a residual connection. |
| `GPT` | The decoder itself, plus an autoregressive `generate()` sampler. |

### Latent conditioning — the key idea

The decoder is driven by a **continuous latent `z`** (size `decoder_input_size`), not by a
start token. The mechanism:

1. `z_proj: Linear(decoder_input_size → n_embd)` maps `z` into the Transformer's embedding
   space.
2. That projected vector becomes the **first token** of the sequence — a *prefix
   conditioning token*.
3. Causal masking guarantees every real token attends back to this prefix, so the whole
   generated sequence is a function of `z`. This is the property a VAE decoder needs.

```
position:   0       1       2      ...   T
input:    [ z ]  [tok_0] [tok_1]  ... [tok_{T-1}]
            │       │       │            │
logits:   pred_0  pred_1  pred_2  ...  pred_T   ← pred_t predicts token_t
```

`pred_0` sees only `z` and predicts the **first** action — the prefix supplies the `+1`
autoregressive shift, so no BOS token is needed.

### Public API

- `forward(z, idx=None, targets=None) -> (logits, loss)` — `logits` shape `(B, T+1, vocab)`.
  Passing `targets` computes the cross-entropy internally (used for convenience; the VAE
  computes its own loss externally).
- `generate(z, max_new_tokens=None, temperature=1.0, do_sample=False, top_k=None) -> idx`
  — free-running autoregressive sampling (KV-cached, see §6.1), returns integer tokens `(B, max_new_tokens)`.
- `init_decode(z)` / `decode_step(token, past, position)` — cached incremental-decoding
  helpers (§6.1) used by `generate` and `ActionGen.gen_action_seq`.

Model size at defaults: **~27K parameters** (was ~152K before the §6.3 shrink).

---

## 2. Mechanics changes — `dense_var_auto_encoder_vin_1.py`

The data representation flipped from **flattened one-hot** to **integer token indices**,
and the decoder/loss became **autoregressive**. The encoder side is unchanged.

### 2.1 `DenseVAE.__init__`

- **Removed** the MLP `self.decoder` (the `Linear → InstanceNorm → LeakyReLU …` stack) and
  `self.decoder_last_layer = nn.Sigmoid()`.
- **Added** `self.decoder = GPT(GPTConfig(vocab_size=n_words, block_size=input_length,
  decoder_input_size=decoder_input_size))`.
- The GPT is built **after** `self.apply(self._init_weights)` so the Transformer keeps
  minGPT's own `normal(0, 0.02)` init instead of being overwritten by the encoder's
  Kaiming-leaky_relu scheme.
- Stored `self.input_length` and `self.n_words` for use in `forward`.

### 2.2 `encode` / `decode` / `forward`

- `encode` — **unchanged**. Still consumes a flattened one-hot view and outputs
  `mean, logvar`.
- `decode(z, idx)` — now teacher-forces the Transformer: `logits, _ = self.decoder(z, idx)`,
  returns logits `(B, T+1, n_words)`. (Previously returned `(sigmoid_probs, logits)`.)
- `forward(x, get_z=False)` — **input is now integer tokens `(B, T)`**. It one-hots them
  internally to feed the encoder, samples `z`, then teacher-forces the decoder on the true
  tokens. **Returns `(logits, mean, logvar[, z])`** — note this is **3 (or 4) values**, down
  from the previous 4 (`x_hat, mean, logvar, x_hat_logits`).

### 2.3 `loss_function`  ← signature changed

```python
loss_function(logits, targets, mean, log_var, desired_var, label_smoothing=0.1)
```

- Reconstruction term is now **`F.cross_entropy`** over the vocabulary
  (`reduction='sum'`), replacing the per-element BCE on one-hot vectors.
- Alignment: `logits[:, :T]` vs the full `targets` — the `+1` shift is supplied by the
  z-prefix at position 0.
- **KLD term is kept verbatim.**
- Old params `x` / `x_hat` / `x_seq_len` / `smoothing` are gone.

### 2.4 `MazeDataLoaderV2` — integer-token mode

- New constructor flag **`return_token_indices: bool = True`**.
- `convert_action` returns the bare integer class id when in token mode.
- `prepare_list` emits `(seq_len,)` `LongTensor`s; stacking a batch yields `(B, seq_len)`.
- Padding still uses token `4` (EOS).
- The soft-label jitter augmentation (writing floats into one-hot tensors) is **guarded off**
  in token mode — it's meaningless for integer indices.

### 2.5 `get_reconstructed_action_list_with_embedding`

Updated to feed integer tokens and argmax `logits[:, :T]`. As a side benefit it no longer
depends on the hardcoded length-10 `convert_action_list_to_one_hot_tensor`.

---

## 3. Production inference change — `decoder_utils.py`

`ActionGen.gen_action_seq` was rewritten from a single `self.model.decoder(gen_input)` call
into a **token-by-token autoregressive loop**:

```
step_logits, past = self.model.decoder.init_decode(z)   # process the z-prefix once (KV cache)
for t in range(n_action_seq_length):
    row = step_emit(step_logits)             # mode-specific representation
    if t < n_action_seq_length - 1:
        next_token = argmax(row).detach()    # feed back the chosen token
        step_logits, past = self.model.decoder.decode_step(next_token, past, position=t + 1)
```

*(Originally a full-forward loop `self.model.decoder(z, idx)` each step; rewritten to the
KV-cached form in §6.1 — numerically identical, O(T) instead of O(T²).)*

- The original branching is **fully preserved** — the selected `step_emit` is one of:
  temperature-scaled softmax (probs path), Gumbel-Softmax, straight-through, or argmax
  one-hot — chosen from `get_actions_as_one_hot`, `deterministic_mode`,
  `deterministic_inference`, `exclude_decoder_from_computation_graph`, `use_gumble`.
- **Output shape is unchanged:** `(batch, n_action_seq_length, n_words)`, so existing callers
  (e.g. `decoder.gen_action_seq(...)[1].argmax(-1)`) keep working.
- **Gradients still reach `z`** via the prefix token at every step. Fed-back tokens are
  detached integers (standard straight-through / Gumbel practice — we don't differentiate
  through the discrete sampling history). The old `decoder_last_layer` (Sigmoid) calls were
  replaced with `softmax`, which is the correct posterior for the cross-entropy-trained model.

---

## 4. ⚠️ Still needs updating / fixing / to notice

These spots still use the **old MLP / one-hot API** and will raise at call time. They were
left untouched because they involve evaluation design choices — fix before training/eval.

### 4.1 `__main__` training loop — **BROKEN**  ([dense_var_auto_encoder_vin_1.py:530-548](dense_var_auto_encoder_vin_1.py#L530-L548))

- `reconstructed_batch, mean, log_var, _ = model(flatten_batch)` unpacks **4** values;
  `forward` now returns **3**.
- `loss_function(flatten_batch, reconstructed_batch, mean, log_var, image_len_batch_l,
  desired_var=VAR)` uses the **old signature**.
- `flatten_batch = image_batch_tensor.flatten(1)` is no longer wanted — the model expects
  integer tokens `(B, T)`, i.e. `image_batch_tensor` directly.

**Fix:**
```python
token_batch = torch.stack(image_batch_l)          # (B, T) long
logits, mean, log_var = model(token_batch)
loss = loss_function(logits, token_batch, mean, log_var, desired_var=VAR)
```

### 4.2 `get_model_performance_on_set` — **BROKEN**  ([dense_var_auto_encoder_vin_1.py:375-399](dense_var_auto_encoder_vin_1.py#L375-L399))

- Unpacks 4–5 return values from `eval_model(...)` (now 3–4).
- `inverse_transform_sequence(recon_seq)` assumes one-hot output; `recon_seq` is now logits
  `(B, T+1, vocab)`.
- Calls `loss_function(...)` with the old signature.

**Fix direction:** get `logits, mean, logvar = eval_model(token_seq.unsqueeze(0))`, take
`recon_act_seq = logits[:, :T].argmax(-1)`, compare directly to the integer ground-truth,
and call the new `loss_function`. Decide whether to measure reconstruction with **teacher
forcing** (cheap, what `forward` gives) or **free-running `generate()`** (matches production
but slower).

### 4.3 `evaluate_k_seq_in_decoder` — `DenseVAE` branch **BROKEN**  ([dense_var_auto_encoder_vin_1.py:442-444](dense_var_auto_encoder_vin_1.py#L442-L444))

`decoder.decoder(rand_noise.unsqueeze(0)).reshape(-1, 10, 5)` no longer works — `decoder.decoder`
is a GPT returning a `(logits, loss)` tuple from a single prefix step, not a full sequence.

**Fix:** `rand_action_seq = decoder.decoder.generate(rand_noise.unsqueeze(0))` → already
`(B, 10)` integer tokens, no reshape/argmax needed. *(The `ActorGen` else-branch is already
correct.)*

### 4.4 `inverse_transform_sequence` / `transform_sequence`  ([dense_var_auto_encoder_vin_1.py:364-372](dense_var_auto_encoder_vin_1.py#L364-L372))

With `input_size == 1` in token mode, `reshape(-1, 1).argmax(1)` returns all zeros — these
helpers no longer fit the integer-token format. Only `get_model_performance_on_set` uses
them; retire or rewrite them as part of 4.2.

### 4.5 Checkpoints — incompatible

The decoder's `state_dict` keys changed (MLP → GPT). **Old `.pt` weights will not load** via
`get_decoder(load_pretrained_weights=True)` in [decoder_utils.py](decoder_utils.py). Retrain
and re-save.

### 4.6 Things to notice (not bugs)

- **Loss scale / β:** reconstruction is now CE with `reduction='sum'`; this changes its
  magnitude relative to the (unchanged) summed KLD. Expect to **retune the KLD weight (β)**
  to avoid posterior collapse or under-reconstruction.
- **EOS padding in the loss:** CE is computed over **all** positions including the EOS
  padding tail. The model will spend capacity learning to emit EOS for padding (usually
  fine for an EOS-terminated scheme). If it dominates, consider masking after the first EOS
  or `ignore_index`.
- **Generation cost:** ~~`gen_action_seq` / `generate` recompute the full forward each step
  (no KV cache) — O(T²)~~ **Fixed in §6.1** — a KV cache makes each step O(1) in the prefix.
- **`InstanceNorm1d` warning:** the encoder emits a benign size warning on 2-D input — this
  is **pre-existing**, unrelated to the Transformer change.
- **Unused leftover:** `convert_action_list_to_one_hot_tensor` is no longer called after the
  `get_reconstructed_action_list_with_embedding` rewrite.

---

## 5. Verification done so far

- `mingpt_decoder.py`: import + forward + `generate` smoke test (logits `(B, 11, 5)`).
- `DenseVAE`: forward → `loss_function` → `backward`; gradients reach `decoder.z_proj`.
- `MazeDataLoaderV2`: integer-token output `(B, 10)`, padded with `4`.
- `ActionGen.gen_action_seq`: all five emission modes produce `(B, 10, 5)`; probs sum to 1;
  one-hot modes are valid one-hots; Gumbel and STE paths propagate gradient to `z`.

**Not yet exercised:** a full end-to-end training run (blocked on §4.1–4.3) and loading a
real action-sequence pickle.

---

## 6. Follow-up fixes (performance + quality)

Applied after end-to-end RL runs were working, to address (a) very long runtimes and
(b) a from-scratch GPT under-performing the from-scratch MLP.

### 6.1 KV cache — faster autoregressive decoding  (`mingpt_decoder.py`)

- `CausalSelfAttention` and `Block` now accept `layer_past` / `use_cache` and return
  `(output, present)`. The causal-mask slice `[Tk-Tq:Tk, :Tk]` generalises the old
  `[:T,:T]` mask (identical for the full teacher-forced pass).
- New `_forward_trunk`, **`init_decode(z)`**, **`decode_step(token, past, position)`** keep a
  per-layer key/value cache, so each generation step only computes the *new* token's
  attention instead of re-encoding the whole growing prefix — **O(T) instead of O(T²)**.
- `generate()` and `ActionGen.gen_action_seq` rewritten to use them.
- **Verified numerically identical** to the non-cached decode (max logit diff ~2e-7);
  gradients to `z` preserved for the Gumbel/STE training paths. ~3× faster per call on a
  CPU microbenchmark. (Supersedes the old "no KV cache" note in §4.6.)

### 6.2 `no_grad` audit — no change needed  (`gps_simplegrid_levels.py`)

All *inference* `gen_action_seq` call sites already run under `torch.no_grad()` (eval
functions are `@torch.no_grad()`-decorated; the rollout/target/diversity decodes sit inside
`with torch.no_grad()` blocks). The one **E2E-training** decode correctly keeps its graph.
Nothing to change — the acting path was never building throwaway graphs.

### 6.3 Shrink the decoder  (`GPTConfig` defaults)

`n_layer 3→2`, `n_embd 64→32` (`n_head=4`, `decoder_input_size=16`, `vocab_size=5`,
`block_size=10` unchanged). Params **152,512 → 26,688** (~5.7×). Rationale: the default
minGPT size was ~60× the old MLP decoder (~2.5K) for a 16-dim→50-logit map — over-sized for
length-10/vocab-5, slower, and harder to optimise from sparse RL reward.

### 6.4 Conditioning-aware init — make `z` drive the decoder  (`GPT.__init__`)

The generic `_init_weights` only touches `Linear`/`Embedding`/`LayerNorm`, so two conditioning
parameters were mis-initialised. Two overrides applied **after** `self.apply(_init_weights)`:

- **`pos_emb`** was an `nn.Parameter(zeros)` and stayed **all-zeros** the whole run (positions
  indistinguishable). Now `normal(std=0.02)`, the standard GPT positional init.
- **`z_proj`** got the tiny trunk `std=0.02`; bumped to `std = 1/√decoder_input_size ≈ 0.25`
  (~10×). Because the residual stream is *not* normalised (only branch inputs are), the larger
  prefix makes **`z` dominate the residual reaching the head**, so different `z` produce
  different outputs from step 0. The scale is a tunable knob.
- **Measured effect:** unique greedy sequences over 64 random `z` rose **12 → 29** at init.

### 6.5 Context / open status

- From-scratch E2E (random init, RL reward only): GPT scored **~0.385** test success vs the
  from-scratch MLP's **~0.794**. §6.3–6.4 target that gap (over-parameterisation + z-agnostic
  init); §6.1 targets runtime.
- **Recalibration:** "valid actions per episode" is ~1.6 even for the strong MLP, so short
  per-decision sequences are *normal* for this task — the real gap is **success rate**, not
  sequence length. (An earlier read treated the GPT's 1.29 as a collapse; the MLP's 1.599
  shows that was overstated.)
- **Open:** a post-shrink/-init GPT run has not yet been measured — that's the test of whether
  §6.3–6.4 close the gap toward ~0.79. If it plateaus below target, the next lever is
  **reconstruction pretraining** (which also needs the §4.1–4.3 standalone-loop fixes).
- **Superseded by §7:** the ~0.385 figure above was measured while two E2E training bugs
  (§7.1) were active, so it does not measure the architecture. It needs re-measuring.

---

## 7. Runtime investigation + E2E correctness fixes (round 2)

Prompted by runtime: the MLP baseline completed in **10:24:26** (slurm 18136376, `cs-1080-02`,
`gtx_1080`) while the GPT run took **2-02:19:29** (slurm 19543017, same GPU class) — a 4.8×
regression. Investigation found the regression was only partly the Transformer; two bugs were
amplifying it and simultaneously preventing the decoder from training.

### 7.1 Two bugs in the E2E path  (`gps_simplegrid_levels.py`)

Both are specific to `--train_decoder_end_to_end`, and both were invisible with the old MLP
decoder — which is why they appeared only after the Transformer landed.

#### 7.1.1 `decoder_optimizer.zero_grad()` was never called

`critic_optimizer.zero_grad()` and `actor_optimizer.zero_grad()` were both present; the
decoder optimizer only ever received `.step()`. Because the decoder sits inside the actor's
computation graph, its gradients **accumulated across every actor update for the whole run** —
by step N the optimizer was stepping along the sum of N/4 gradients rather than the current
one. Adam normalises the magnitude, so this never produced a NaN or a crash; it silently
pointed the decoder in the wrong direction.

**Fix:** clear the decoder's gradients on the same schedule as the actor's.

**Verified:** decoder gradient norm now reads 1.0 → 16.7 over 30k steps. Accumulation would
have put it in the hundreds or thousands (7,500 summed updates at that point).

#### 7.1.2 The decoder never left `train()` mode

`decoder.model.to(device).train()` is set for E2E mode and nothing ever set it back.
`eval_model` put the *actor* and *critic* into eval mode but not the decoder. The old MLP
decoder had **no dropout layers at all**, so its train/eval mode was a no-op and nobody
noticed; minGPT has three (`embd_pdrop`, `attn_pdrop`, `resid_pdrop`, all 0.1), which were
therefore **injecting noise into every generated sequence** during action selection, the
target-Q computation, and all evaluation.

**Fix:** two parts —
- `eval_model` saves/restores the decoder's training flag and calls `.eval()`.
- New `--decoder_dropout` (**default 0.0**) matches the old MLP's behaviour, which also
  covers the rollout and target-Q paths that `eval_model` does not touch. Pass `0.1` to
  restore the previous behaviour.

#### Combined effect on runtime

A noisy/mis-trained decoder emits sequences that trim to length 1, so the agent makes far
more decisions per episode. `num_decoder_generations` saturates toward `max_episode_steps`
(75) instead of ~20, which multiplies the **evaluation** term — the largest single cost in a
production run — by roughly 10×. The bugs were both a correctness problem and the bulk of the
runtime regression.

### 7.2 Measurements

All on **GTX 1080 Ti (sm_61)**, torch 2.2.1+cu121, via `bench_decoder.py` and `sbatch/e*.sbatch`.

**Per call** (ms), `bench_decoder.py`:

| decoder | fwd b=1 | fwd b=256 | fwd+bwd b=256 |
|---|---|---|---|
| MLP (legacy, 1 pass) | 0.469 | 0.493 | 1.213 |
| GPT eager `L=2 embd=32` | 10.478 | 10.921 | 29.954 |
| GPT eager `L=1 embd=32` | 7.165 | 7.525 | 19.722 |
| GPT eager `L=1 embd=16` | 7.325 | 7.351 | 22.886 |
| **GPT cuda-graph** `L=2` | **1.228** | **5.051** | n/a (forward only) |
| GPT on CPU (+transfers) | 10.686 | 37.360 | n/a |

**The decoder is ~100% dispatch overhead, not arithmetic.** Three independent confirmations:
batch 1 and batch 256 cost the same to within 1%; `embd` 32→16 is 3.5× fewer params and no
faster; cost fits `3.73 ms + 3.91 ms × n_layer`, so even a 0-layer trunk would cost 7× the
entire old decoder — that constant is the 10-iteration Python loop itself.

**End-to-end** (30k steps, projected to 1M):

| run | config | projected |
|---|---|---|
| E1 | eval off, deterministic, eager | 19.67 h |
| E3 | eval on @5000, deterministic, eager | **30.54 h** |
| E5 | eval off, **non-deterministic**, eager | 11.52 h |
| E8 | eval on @5000, deterministic, **cuda graph** | **15.27 h** |

- **Evaluation costs 10.87 h** (E3 − E1) — 36% of a production run.
- **Determinism costs 8.15 h** (E1 − E5), a 1.71× penalty. Not the convolutions (`conv2d` is
  1.2% of the loop) but **cuBLAS**: `CUBLAS_WORKSPACE_CONFIG=:4096:8` exists only to satisfy
  `torch.use_deterministic_algorithms`, and `torch._C._nn.linear` is ~30% of the loop.
- **The CUDA graph halves the run** (E3 − E8, exactly 2.0×).

**Profile** (`cProfile`, 3k steps, `sbatch/e6_profile_trainloop.sbatch`): `_forward_trunk`
cumtime **120.96 s of 241.11 s — ~50% of the training loop is the decode loop**.
`torch._C._nn.linear` is 555,271 calls / 73.49 s, of which ~508k (91%) are the decoder
(56,450 trunk passes × 9 linears).

### 7.3 Performance changes

#### 7.3.1 Fused attention  (`mingpt_decoder.py`)

`CausalSelfAttention.forward` replaces the matmul / scale / `masked_fill` / softmax / dropout
/ matmul chain with one `F.scaled_dot_product_attention`. The causal mask buffer is now
`torch.bool` (what SDPA wants, no per-call conversion), and when `Tq == 1` — every cached
`decode_step` — the single query sits at the last position, so **every key is already in its
past and no mask is passed at all**. Mathematically identical.

Note: on sm_61 with fp32, flash is unavailable and this likely dispatches to the math
backend, so the win here is small on *this* hardware. It should matter on sm_75+.

#### 7.3.2 CUDA graph for the gradient-free decode paths  (`decoder_utils.py`)

`gen_action_seq`'s loop is extracted into `_generate(z, step_emit)`, so one code path serves
both eager and captured execution. `_generate_graphed` caches graphs keyed by
`(batch_size, emit_mode)`; enabled by **`--decoder_use_cuda_graph`** (default off).

The switch is **`torch.is_grad_enabled()`**, not a call-site flag: grad is off for action
selection, target-Q and evaluation — nearly all the calls — and on for the actor update,
which stays eager so gradients still reach `z` and the decoder parameters.

Three properties make this safe inside a *training* loop:

- **Weights stay live.** The graph holds pointers to parameter tensors and Adam mutates
  `param.data` in place, so replays see current weights. ⚠️ `handle_nan_weights`
  (`gps_simplegrid_levels.py`) does `param.data = torch.randn_like(...)`, which **rebinds** and
  would silently invalidate every captured graph. It is currently dead code — if it is ever
  wired up, the graph cache must be cleared alongside it.
- **The output is cloned.** A graph writes into a fixed buffer that the next replay
  overwrites, and the training loop re-reads `action_list_batch` after stepping the
  environment. Returning the live buffer would be silent data corruption.
- **Capture failure degrades, not crashes.** One warning, `use_cuda_graph = False`, continue
  eager — a 15-hour job should not die on an unsupported op.

**Verified (E8 vs E3, identical config, graph the only difference):** exactly three captures
and no fallback — `(256, gumbel)` target-Q, `(1, gumbel)` action selection, `(1, argmax)`
eval. Success rate 0.27 → 0.25 (within the ±0.044 standard error of a 100-episode val set);
valid actions/sequence 1.237 → 1.408; decoder generations 19.963 → 17.84. No corruption.

#### 7.3.3 Decoder shape + dropout as CLI flags

`--decoder_n_layer`, `--decoder_n_head`, `--decoder_n_embd`, `--decoder_dropout`, plumbed
`Config` → `get_decoder_api` → `get_decoder` → `DenseVAE` → `GPTConfig`. Defaults match the
previously hardcoded values, so nothing changes unless passed. Added so §7.2's sweeps needed
no file edits between runs.

### 7.4 Ruled out

| candidate | result |
|---|---|
| `torch.compile(mode="reduce-overhead")` | **Fails on sm_61**: `Triton only supports devices of CUDA Capability >= 7.0`. Untested on sm_75+, where it could also graph the *backward* pass that §7.3.2 leaves eager. |
| Decoder on CPU | No better at b=1, **3.4× worse** at b=256. |
| Shrinking the model | `embd` 32→16 gains 1.6%. The cost is not arithmetic. |
| Faster GPU | RTX 2080 Ti vs GTX 1080 Ti, same script: 10.73 h vs 11.52 h — **7%**. Dispatch overhead is CPU-side; GPU FLOPS are nearly irrelevant. |
| Replay-buffer sampling | Hypothesised that `random.sample` over a `deque` (O(n) indexing, ~12.5k hops × 256 draws per step) was significant. **Wrong** — all `deque` ops total ~0.02 s. |
| `TensorCache` on `trim_action_sequence_from_eos_tokens` | Real but minor: its key is `tuple(actions.tolist())`, forcing a GPU→CPU sync per call, ~4% of the loop — the key costs more than the ops it caches. **Not applied** (also an aliasing hazard: the function mutates `actions` in place, so a cached tensor lets one caller's edit surface in another's result). |
| Gradient-monitoring block | 103,736 `.item()` calls / 2.47 s (~1%) every 250 steps. **Not applied.** |

### 7.5 Status

Production config is `run_gps.sbatch` (deterministic, ~15 h) and `run_gps_nondet.sbatch`
(~9 h). Both pin **`--gpus=gtx_1080:1`** — the partition is heterogeneous (`gtx_1080`,
`rtx_2080`, `rtx_3090`) and a bare `--gpus=1` silently hands out other hardware, which makes
runtime incomparable to the 10.4 h MLP baseline. Both enable `--decoder_use_cuda_graph` and
disable the periodic test-set evaluations (`--eval_test_dataset_during_training_freq -1`),
which feed no model selection and no training signal — only logging. Verified: of the three
evaluation sites, **only the val evaluation ever saves a checkpoint**.

**Runtime: resolved.** 50.3 h → ~15 h deterministic, ~9 h non-deterministic, against a 10.4 h
MLP baseline on the same GPU class.

**Accuracy: open, and not addressed by any of the above.** §7.3 is mathematically neutral;
§7.1 is what should move the number. The §6.5 comparison (GPT ~0.385 vs MLP ~0.794) was
measured with both bugs active and must be re-run before it means anything.

For calibration when reading the new runs: at 30k steps (3% of a run) E3/E8 reached 0.25–0.27
val success with 1.24–1.41 valid actions per sequence. Per §6.5's own recalibration, ~1.6 is
normal for this task even for the 0.794 MLP, so short sequences are **not** a collapse signal.
