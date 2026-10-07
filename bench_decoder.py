"""Microbenchmark: how much does the minGPT decoder cost vs the old MLP decoder?

The VAE decoder went from a single ``nn.Sequential`` forward pass (one shot, fully
batched) to an autoregressive Transformer that runs one trunk pass per generated token.
At our sizes (n_embd=32, vocab=5, seq=10) neither is remotely FLOP-bound -- both are
dominated by CUDA kernel launch latency -- so the thing that actually sets wall-clock time
is *how many kernels get launched per call*, which went up by roughly 50x.

This script measures that directly, for the three shapes the training loop actually uses:

  * ``fwd b=1``    -- action selection during rollout, and every eval decision
  * ``fwd b=256``  -- the target-Q computation, which runs on *every* training step
  * ``fwd+bwd b=256`` -- the actor update, which backprops through the whole decode loop

Run it on a GPU node; it takes well under a minute and needs no dataset:

    srun --partition=main --gpus=1 --mem=8G --time=0-00:15:00 \
        poetry run python bench_decoder.py

Compare the ``per-call`` column across rows. The MLP row is the 9-hour-run baseline.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch
import torch.nn as nn

from generative.models.dense_autoencoder.decoder_utils import ActionGen
from generative.models.dense_autoencoder.dense_var_auto_encoder_vin_1 import DenseVAE

SEQ_LEN = 10
N_WORDS = 5
LATENT = 16


class LegacyMlpDecoder(nn.Module):
    """The pre-Transformer decoder trunk, reproduced from commit 47d6da4^.

    Kept here purely as a timing reference: this is the decoder the 9-hour run used.
    """

    def __init__(self, decoder_input_size: int = LATENT, input_length: int = SEQ_LEN,
                 n_words: int = N_WORDS) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size),
            nn.InstanceNorm1d(decoder_input_size),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_input_size, decoder_input_size * 2),
            nn.InstanceNorm1d(decoder_input_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_input_size * 2, input_length * n_words),
            nn.InstanceNorm1d(input_length * n_words),
        )
        self.n_action_seq_length = input_length
        self.n_words = n_words

    def gen_action_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Mirror the old ActionGen.gen_action_seq one-hot path."""
        logits = self.trunk(z).reshape(-1, self.n_action_seq_length, self.n_words)
        return torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)


def timeit(fn: Callable[[], torch.Tensor], device: torch.device,
           n_warmup: int = 20, n_iter: int = 100) -> float:
    """Return mean seconds per call, with the GPU properly synchronised."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_iter


def make_gpt_decoder(device: torch.device, n_layer: int, n_head: int, n_embd: int,
                     dropout: float) -> ActionGen:
    vae = DenseVAE(
        input_length=SEQ_LEN, n_words=N_WORDS, device=device,
        decoder_input_size=LATENT,
        decoder_n_layer=n_layer, decoder_n_head=n_head, decoder_n_embd=n_embd,
        decoder_dropout=dropout,
    ).to(device)
    return ActionGen(pretrained_decoder=vae, n_act_seq_len=SEQ_LEN, device=device)


def bench_row(label: str, device: torch.device, n_params: int,
              fwd1: Callable[[], torch.Tensor],
              fwd256: Callable[[], torch.Tensor],
              fwdbwd256: Callable[[], torch.Tensor]) -> None:
    with torch.no_grad():
        t1 = timeit(fwd1, device)
        t256 = timeit(fwd256, device)
    tbwd = timeit(fwdbwd256, device, n_warmup=10, n_iter=50)
    print(f"{label:<28} {n_params:>8,}  {t1 * 1e3:>9.3f}  {t256 * 1e3:>11.3f}  {tbwd * 1e3:>13.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=256,
                        help="batch size of the training-loop decoder calls")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}\n")

    z1 = torch.randn(1, LATENT, device=device)
    zb = torch.randn(args.batch, LATENT, device=device)

    header = (f"{'decoder':<28} {'params':>8}  {'fwd b=1':>9}  "
              f"{'fwd b=' + str(args.batch):>11}  {'fwd+bwd b=' + str(args.batch):>13}")
    print(header)
    print("-" * len(header))
    print(f"{'(all times are ms per call)':<28}")

    # --- Reference: the decoder the 9-hour run used -----------------------------------
    mlp = LegacyMlpDecoder().to(device)
    bench_row(
        "MLP (legacy, 1 pass)", device,
        sum(p.numel() for p in mlp.parameters()),
        lambda: mlp.gen_action_seq(z1),
        lambda: mlp.gen_action_seq(zb),
        lambda: mlp.gen_action_seq(zb).sum().backward(),
    )

    # --- The Transformer decoder, at a few trunk sizes --------------------------------
    for n_layer, n_head, n_embd in [(2, 4, 32), (1, 4, 32), (1, 2, 16)]:
        gen = make_gpt_decoder(device, n_layer, n_head, n_embd, dropout=0.0)
        gen.model.eval()
        label = f"GPT n_layer={n_layer} n_embd={n_embd}"
        bench_row(
            label, device,
            sum(p.numel() for p in gen.model.decoder.parameters()),
            lambda g=gen: g.gen_action_seq(z1, get_actions_as_one_hot=True)[0],
            lambda g=gen: g.gen_action_seq(zb, get_actions_as_one_hot=True)[0],
            lambda g=gen: g.gen_action_seq(zb, get_actions_as_one_hot=True)[0].sum().backward(),
        )

    print()
    print("Projected cost over a 1,000,000-step run, decoder time only:")
    print("  training loop  ~= total_timesteps * (1 fwd b=256)          [target-Q, every step]")
    print("               + ~= total_timesteps/4 * (1 fwd+bwd b=256)    [actor update]")
    print("  evaluation     ~= n_evals * n_episodes * gens_per_episode * (1 fwd b=1)")
    print("  gens_per_episode is ~8 with healthy sequences and up to 75 (max_episode_steps)")
    print("  when the decoder degenerates to length-1 sequences -- which is the term that")
    print("  dominates, so watch 'mean num valid actions in sequences' in the training log.")


if __name__ == "__main__":
    main()
