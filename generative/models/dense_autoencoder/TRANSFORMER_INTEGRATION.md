# Transformer Decoder Integration

Replacing the VAE's single-shot MLP decoder with a minimal, latent-conditioned causal
Transformer (minGPT-style) and migrating the surrounding mechanics from a flattened
one-hot representation to integer-token autoregressive modeling.

Files touched:
- **New:** [mingpt_decoder.py](mingpt_decoder.py)
- **Changed:** [dense_var_auto_encoder_vin_1.py](dense_var_auto_encoder_vin_1.py)
- **Changed:** [decoder_utils.py](decoder_utils.py)

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
