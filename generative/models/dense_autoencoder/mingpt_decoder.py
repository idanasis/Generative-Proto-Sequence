"""Minimal causal Transformer decoder (minGPT-style) for conditional sequence generation.

Adapted from Andrej Karpathy's minGPT (https://github.com/karpathy/minGPT) and trimmed
down to the bare essentials needed to replace the MLP decoder of the VAE in
``dense_var_auto_encoder_vin_1.py``.

The key addition over a vanilla GPT is *latent conditioning*: the decoder is driven by a
continuous latent vector ``z`` (size ``decoder_input_size``). ``z`` is linearly projected
into the Transformer's embedding space and inserted as the **first token** of the
sequence -- a prefix conditioning token. Because attention is causal, every real token
can attend back to this prefix and thus to ``z``, which makes the whole sequence a
function of the latent code, exactly as a VAE decoder requires.

Architecture defaults are deliberately tiny (``n_layer=3``, ``n_head=4``, ``n_embd=64``)
because our action sequences are short (length 10) and the vocabulary is small (5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyper-parameters for the conditional GPT decoder.

    Attributes:
        vocab_size: Number of distinct tokens (4 actions + 1 EOS = 5).
        block_size: Maximum length of the *real* token sequence (10). The model
            internally works on ``block_size + 1`` positions to make room for the
            prefix conditioning token derived from ``z``.
        decoder_input_size: Dimensionality of the continuous latent vector ``z``.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads (must divide ``n_embd``).
        n_embd: Embedding / hidden width of the Transformer.
        embd_pdrop: Dropout applied to the summed token + positional embeddings.
        resid_pdrop: Dropout applied to the residual paths inside each block.
        attn_pdrop: Dropout applied to the attention probabilities.
    """

    vocab_size: int = 5
    block_size: int = 10
    decoder_input_size: int = 16
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 64
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.

    Each position may only attend to itself and earlier positions (causal masking),
    which is what makes the decoder autoregressive.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined linear projection producing queries, keys and values in one matmul.
        self.c_attn: nn.Linear = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection applied after attention.
        self.c_proj: nn.Linear = nn.Linear(config.n_embd, config.n_embd)

        # Regularisation.
        self.attn_dropout: nn.Dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout: nn.Dropout = nn.Dropout(config.resid_pdrop)

        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd

        # Causal mask of shape (1, 1, T, T) where T = block_size + 1 (prefix token).
        # Registered as a buffer so it moves with ``.to(device)`` but is not a parameter.
        max_len = config.block_size + 1
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention.

        Args:
            x: Input of shape ``(B, T, n_embd)``.

        Returns:
            Tensor of shape ``(B, T, n_embd)``.
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project and split into query, key, value: each (B, T, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into heads: (B, n_head, T, head_dim).
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal masking.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads

        # Output projection + residual dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """A single Transformer block: causal self-attention followed by an MLP.

    Uses the pre-LayerNorm formulation (LayerNorm before each sub-layer), which trains
    more stably than the original post-LayerNorm GPT.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.attn: CausalSelfAttention = CausalSelfAttention(config)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embd)

        # Position-wise feed-forward network (4x expansion is the GPT convention).
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """A minimal GPT decoder conditioned on a continuous latent vector ``z``.

    The forward pass prepends a single conditioning token -- a linear projection of
    ``z`` -- to the (optional) token embeddings, then runs a causal Transformer. The
    logits aligned with the prefix predict the first action, and so on, giving an
    autoregressive ``p(sequence | z)`` factorisation.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config
        self.block_size: int = config.block_size

        # --- Latent conditioning: project z into the Transformer's embedding space. ---
        # The result is treated as the first ("prefix") token of the sequence.
        self.z_proj: nn.Linear = nn.Linear(config.decoder_input_size, config.n_embd)

        # Token and (learned) positional embeddings. ``+ 1`` position accounts for the
        # prefix conditioning token sitting in front of the real sequence.
        self.tok_emb: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb: nn.Parameter = nn.Parameter(
            torch.zeros(1, config.block_size + 1, config.n_embd)
        )
        self.drop: nn.Dropout = nn.Dropout(config.embd_pdrop)

        # Transformer trunk.
        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.n_embd)

        # Language-model head: project hidden states back to vocabulary logits.
        self.head: nn.Linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Loudly announce (once, at construction) that the minGPT Transformer decoder is
        # in use -- so a run can be confirmed to be using this file and not the old MLP.
        self._announce()

    def _announce(self) -> None:
        """Print a one-time banner and, if a wandb run is active, tag its config.

        This is purely a diagnostic so you can verify from the logs / wandb that the
        Transformer decoder (this file) is the one being instantiated.
        """
        cfg = self.config
        n_params = sum(p.numel() for p in self.parameters())
        print(
            "\n==================== [GPS] DECODER = minGPT GPT ====================\n"
            f"  mingpt_decoder.GPT active  |  trainable params = {n_params:,}\n"
            f"  n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}\n"
            f"  vocab_size={cfg.vocab_size}  block_size={cfg.block_size}  "
            f"decoder_input_size={cfg.decoder_input_size}\n"
            "====================================================================\n",
            flush=True,
        )

        # Best-effort: record in wandb if (and only if) a run is already active.
        # Never let logging issues break model construction.
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "decoder_type": "mingpt_gpt",
                        "decoder_n_params": n_params,
                        "decoder_n_layer": cfg.n_layer,
                        "decoder_n_head": cfg.n_head,
                        "decoder_n_embd": cfg.n_embd,
                        "decoder_vocab_size": cfg.vocab_size,
                        "decoder_block_size": cfg.block_size,
                        "decoder_input_size": cfg.decoder_input_size,
                    },
                    allow_val_change=True,
                )
                print("[GPS] tagged wandb run config with decoder_type=mingpt_gpt", flush=True)
        except Exception as exc:  # pragma: no cover - diagnostics must never crash training
            print(f"[GPS] (wandb tagging skipped: {exc})", flush=True)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-style weight initialisation."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        z: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the conditional decoder.

        Args:
            z: Latent conditioning vector of shape ``(B, decoder_input_size)``.
            idx: Token indices of shape ``(B, T)`` with ``T <= block_size``. These are
                the (teacher-forced) input tokens. If ``None``, the model runs on the
                prefix token alone -- useful as the first step of free-running
                generation.
            targets: Optional ground-truth token indices of shape ``(B, T)`` for which a
                cross-entropy loss is computed and returned.

        Returns:
            A tuple ``(logits, loss)`` where ``logits`` has shape
            ``(B, T_total, vocab_size)`` with ``T_total = (T or 0) + 1`` (the leading
            position corresponds to the prefix conditioning token), and ``loss`` is the
            cross-entropy if ``targets`` was provided, otherwise ``None``.
        """
        B = z.size(0)

        # Project the latent into a single prefix token: (B, 1, n_embd).
        prefix = self.z_proj(z).unsqueeze(1)

        if idx is not None:
            T = idx.size(1)
            assert T <= self.block_size, (
                f"Sequence length {T} exceeds block size {self.block_size}"
            )
            token_embeddings = self.tok_emb(idx)  # (B, T, n_embd)
            # Concatenate prefix in front of the real tokens.
            x = torch.cat([prefix, token_embeddings], dim=1)  # (B, T + 1, n_embd)
        else:
            x = prefix  # (B, 1, n_embd)

        t_total = x.size(1)
        position_embeddings = self.pos_emb[:, :t_total, :]  # (1, T + 1, n_embd)
        x = self.drop(x + position_embeddings)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T + 1, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # The logit at position i predicts token i. Position 0 is fed by the prefix
            # (predicts the first action); positions 1..T are fed by tokens 0..T-1. We
            # therefore align the first T logits with the T targets.
            logits_for_loss = logits[:, : targets.size(1), :]
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample a token sequence conditioned on ``z``.

        Starting from the prefix token alone, repeatedly predict the next token and feed
        it back in until ``max_new_tokens`` have been produced.

        Args:
            z: Latent conditioning vector of shape ``(B, decoder_input_size)``.
            max_new_tokens: Number of tokens to generate. Defaults to ``block_size``.
            temperature: Softmax temperature; lower is more deterministic.
            do_sample: If ``True`` sample from the distribution, else take the argmax.
            top_k: If set, restrict sampling to the ``top_k`` most likely tokens.

        Returns:
            Generated token indices of shape ``(B, max_new_tokens)``.
        """
        if max_new_tokens is None:
            max_new_tokens = self.block_size

        B = z.size(0)
        idx = torch.empty((B, 0), dtype=torch.long, device=z.device)

        for _ in range(max_new_tokens):
            # Never exceed the model's context window.
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > 0 else None
            logits, _ = self.forward(z, idx_cond)

            # Take the logits at the final position to predict the next token.
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
