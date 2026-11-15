#!/usr/bin/env python
"""Offline pretraining for the Transformer (minGPT) VAE decoder.

Why this exists
---------------
GPS's sample efficiency does not come from the actor. It comes from the decoder already
knowing what an action sequence looks like before RL starts: the paper's own sweep config
(``hypersearch-grid-gps.py``) runs ``initialize_decoder_from_pretrained=True`` and
``train_decoder_end_to_end=False``, i.e. a pretrained, frozen decoder.

Every run in this repository so far has used ``--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`` for *both* arms, because the only checkpoint on
disk is the MLP VAE (``denseAE_generic_2act_seq_VAE_v7_...pt``) and ``DenseVAE`` on this
branch is a Transformer -- its ``state_dict`` does not fit. The comparison was therefore
fair but ran GPS with its sample-efficiency mechanism switched off. This script produces
the missing Transformer checkpoint so the paper protocol can be run.

The hypothesis it is meant to test (EVIDENCE.md: actor-side repertoire collapse) is that
training decoder and actor together lets the decoder shrink its latent space toward
whatever narrow region the actor occupies -- consistent with the observed small ``z``
magnitude and the plan length pinned to exactly 3.000 on every map and every state. A
pretrained frozen decoder removes that degree of freedom. The corpus holds sequences of
length 1-10, so a decoder fitted to it starts out with a *distribution* over plan lengths
rather than a single RL-induced constant.

Usage
-----
    poetry run python pretrain_decoder.py --epochs 20000

    # quick smoke test before committing a long run
    poetry run python pretrain_decoder.py --epochs 20 --eval_every 5 --n_diversity 256

CPU is fine -- the shipped MLP checkpoint was trained on CPU and this model is 26.7k
parameters. No GPU slot required.

Then point an RL run at the result:

    python gps_simplegrid_levels.py ... \
        --decoder_model_path <--out path> \
        --initialize_decoder_from_pretrained --no-train_decoder_end_to_end

Shape flags must match the RL run exactly or ``load_state_dict`` will reject the file.
The defaults here mirror ``Config`` in ``gps_simplegrid_levels.py`` (``n_actions_in_seq=10``,
``maze_n_actions=4`` -> ``n_words=5``, 2 layers, 4 heads, ``n_embd=32``, latent 16).

Deviations from the stale ``__main__`` block in ``dense_var_auto_encoder_vin_1.py``
-----------------------------------------------------------------------------------
That block predates the Transformer refactor and no longer runs at all (it unpacks four
values from a three-value ``forward``, passes ``desired_var`` twice to ``loss_function``,
and feeds flattened one-hot where the model now wants token indices). Rewritten here, with
three deliberate fixes:

1. **Batch/length misalignment.** It sliced sequences out of the *unshuffled* ``train_iter``
   while taking lengths from the *shuffled* ``c_train_seq_len``, so sequences were paired
   with other sequences' lengths. Here both come from the same shuffled list.
2. **Fixed shuffle.** It called ``random.Random(4).shuffle(...)`` every epoch, producing an
   identical ordering each time, so batch composition never varied. Here the RNG advances.
3. **No held-out data.** It merged train/val/test into train. Here validation is held out by
   default so overfitting is visible; pass ``--merge_all_into_train`` for the old behaviour.

The short/long interleaving (``combine_lists_iteratively`` over sequences <=5 and >5) is
deliberate curriculum design in the original and is preserved.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from generative.models.dense_autoencoder.dense_var_auto_encoder_vin_1 import (
    DenseVAE,
    MazeDataLoaderV2,
    combine_lists_iteratively,
    loss_function,
)

DEFAULT_DB = "action_seq_db_1_to_10_2_act_in_seq_v1.pickle"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def reconstruction_metrics(model, seqs, lens, device, var, batch_size=512):
    """Teacher-forced reconstruction quality.

    ``exact`` is the fraction of sequences reproduced with zero token errors, including
    the padding/EOS tail -- the strictest measure and the one comparable to the original
    script's "# wrong sequences". ``eff_acc`` ignores padding and scores only the real
    actions, which is what the RL agent actually executes.
    """
    was_training = model.training
    model.eval()
    exact = tok_ok = tok_n = eff_ok = eff_n = 0
    total_loss = 0.0
    n = 0
    for i in range(0, len(seqs), batch_size):
        batch = torch.stack(seqs[i : i + batch_size]).to(device)
        blens = lens[i : i + batch_size]
        logits, mean, logvar = model(batch)
        T = batch.size(1)
        pred = logits[:, :T, :].argmax(-1)
        match = pred == batch
        exact += int(match.all(dim=1).sum())
        tok_ok += int(match.sum())
        tok_n += batch.numel()
        for b, L in enumerate(blens):
            L = int(L)
            eff_ok += int(match[b, :L].sum())
            eff_n += L
        total_loss += float(loss_function(logits, batch, mean, logvar, desired_var=var))
        n += batch.size(0)
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(n, 1),
        "exact": exact / max(n, 1),
        "tok_acc": tok_ok / max(tok_n, 1),
        "eff_acc": eff_ok / max(eff_n, 1),
        "n": n,
    }


@torch.no_grad()
def plan_diversity(model, n_samples: int, device, var: float):
    """Unique greedy plans decoded from random z -- the decoder-health metric.

    Directly comparable to ``check_z_influence.py`` and RESULTS.md Table 6, which report
    64/256 for the end-to-end-trained decoder on 24x24_empty and 60/256 on room_v1.
    Also reports the plan-length distribution, which is the thing end-to-end training
    collapses to a constant.
    """
    was_training = model.training
    model.eval()
    z = torch.randn(n_samples, model.decoder_input_size, device=device) * (var ** 0.5)
    seqs = model.decoder.generate(z, do_sample=False)
    if was_training:
        model.train()
    rows = seqs.tolist()
    unique = len({tuple(r) for r in rows})
    # Effective length = tokens before the first padding/EOS symbol (n_words - 1).
    pad = model.n_words - 1
    lengths = [(r.index(pad) if pad in r else len(r)) for r in rows]
    return {
        "unique": unique,
        "n": n_samples,
        "len_mean": float(np.mean(lengths)),
        "len_std": float(np.std(lengths)),
        "len_unique": len(set(lengths)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shape -- must match the RL run.
    p.add_argument("--input_length", type=int, default=10, help="Config.n_actions_in_seq")
    p.add_argument("--maze_n_actions", type=int, default=4, help="n_words = this + 1")
    p.add_argument("--decoder_input_size", type=int, default=16, help="latent dim")
    p.add_argument("--decoder_n_layer", type=int, default=2)
    p.add_argument("--decoder_n_head", type=int, default=4)
    p.add_argument("--decoder_n_embd", type=int, default=32)
    p.add_argument("--decoder_z_all_positions", action="store_true",
                   help="branch z-every-position; must match the RL run")
    p.add_argument("--var", type=float, default=1.0, help="VAE prior variance (Config var_for_sample)")
    # Optimisation.
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Data.
    p.add_argument("--db", type=str, default=DEFAULT_DB)
    p.add_argument("--merge_all_into_train", action="store_true",
                   help="reproduce the original script: no held-out validation")
    # Reporting / output.
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000, help="0 disables periodic checkpoints")
    p.add_argument("--n_diversity", type=int, default=256, help="random z samples; 256 matches Table 6")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    n_words = args.maze_n_actions + 1

    out = Path(args.out) if args.out else Path(
        "generative/models/dense_autoencoder"
    ) / (
        f"gptVAE_l{args.decoder_n_layer}_h{args.decoder_n_head}_e{args.decoder_n_embd}"
        f"_in{args.input_length}_lat{args.decoder_input_size}_var{int(args.var)}"
        f"_ep{args.epochs}_seed{args.seed}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    mzl = MazeDataLoaderV2(
        f_names_list=[args.db],
        augment_more_data=False,
        merge_all_into_train=args.merge_all_into_train,
    )
    train_seqs, val_seqs, _, train_lens, val_lens, _ = mzl.get_train_val_test_data_eos_version(
        reserve_two_words=False
    )

    model = DenseVAE(
        input_length=args.input_length,
        n_words=n_words,
        device=device,
        variance_for_sample=args.var,
        decoder_input_size=args.decoder_input_size,
        decoder_n_layer=args.decoder_n_layer,
        decoder_n_head=args.decoder_n_head,
        decoder_n_embd=args.decoder_n_embd,
        decoder_z_all_positions=args.decoder_z_all_positions,
    ).to(device)
    model.train()

    n_params = sum(q.numel() for q in model.parameters())
    n_dec = sum(q.numel() for q in model.decoder.parameters())
    print(
        f"=== PRETRAIN: layers={args.decoder_n_layer} heads={args.decoder_n_head} "
        f"n_embd={args.decoder_n_embd} latent={args.decoder_input_size} "
        f"seq_len={args.input_length} n_words={n_words} var={args.var} "
        f"z_all_positions={int(args.decoder_z_all_positions)} epochs={args.epochs} "
        f"lr={args.lr} bs={args.batch_size} seed={args.seed} device={args.device} ==="
    )
    print(f"params: {n_params:,} total, {n_dec:,} in the Transformer decoder")
    print(f"train sequences: {len(train_seqs):,}"
          + (f" | val: {len(val_seqs):,}" if val_seqs else " | val: merged into train"))
    print(f"output: {out}")

    d0 = plan_diversity(model, args.n_diversity, device, args.var)
    print(f"at init: unique plans {d0['unique']}/{d0['n']}, "
          f"plan length {d0['len_mean']:.2f}+/-{d0['len_std']:.2f} "
          f"({d0['len_unique']} distinct lengths)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pairs = [(s, int(L)) for s, L in zip(train_seqs, train_lens.tolist())]
    short = [q for q in pairs if q[1] <= 5]
    long_ = [q for q in pairs if q[1] > 5]
    rng = random.Random(args.seed)

    eval_lens = train_lens.tolist()
    val_pack = None
    if val_seqs:
        val_pack = (val_seqs, val_lens.tolist())

    best_key = -1.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(args.epochs):
        rng.shuffle(short)
        rng.shuffle(long_)
        order = combine_lists_iteratively(short, long_, args.batch_size // 2)
        seqs_ep = [q[0] for q in order]

        running = 0.0
        n_batches = 0
        for i in range(0, len(seqs_ep), args.batch_size):
            # NOTE: sequences and their lengths both come from `order`, unlike the
            # original script -- see the module docstring, deviation 1.
            batch = torch.stack(seqs_ep[i : i + args.batch_size]).to(device)
            optimizer.zero_grad()
            logits, mean, logvar = model(batch)
            loss = loss_function(
                logits, batch, mean, logvar,
                desired_var=args.var, label_smoothing=args.label_smoothing,
            )
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            tr = reconstruction_metrics(model, train_seqs, eval_lens, device, args.var)
            div = plan_diversity(model, args.n_diversity, device, args.var)
            msg = (
                f"epoch {epoch}/{args.epochs}  loss {running / max(n_batches, 1):.3f}  "
                f"train exact {tr['exact']:.3f} eff_acc {tr['eff_acc']:.3f}  "
                f"unique {div['unique']}/{div['n']}  "
                f"len {div['len_mean']:.2f}+/-{div['len_std']:.2f}"
            )
            key = tr["exact"]
            if val_pack is not None:
                va = reconstruction_metrics(model, val_pack[0], val_pack[1], device, args.var)
                msg += f"  | val exact {va['exact']:.3f} eff_acc {va['eff_acc']:.3f}"
                key = va["exact"]
            print(msg + f"  [{time.time() - t0:.0f}s]", flush=True)

            if key > best_key:
                best_key, best_epoch = key, epoch
                torch.save(model.state_dict(), out.with_suffix(".best.pt"))

        if args.save_every and epoch and epoch % args.save_every == 0:
            torch.save(model.state_dict(), out.with_suffix(".mid.pt"))

    torch.save(model.state_dict(), out)

    tr = reconstruction_metrics(model, train_seqs, eval_lens, device, args.var)
    div = plan_diversity(model, args.n_diversity, device, args.var)
    print("\n=== FINAL ===")
    print(f"train: exact {tr['exact']:.4f}  token acc {tr['tok_acc']:.4f}  "
          f"action-only acc {tr['eff_acc']:.4f}")
    if val_pack is not None:
        va = reconstruction_metrics(model, val_pack[0], val_pack[1], device, args.var)
        print(f"val:   exact {va['exact']:.4f}  token acc {va['tok_acc']:.4f}  "
              f"action-only acc {va['eff_acc']:.4f}")
        print(f"best val exact {best_key:.4f} at epoch {best_epoch} -> {out.with_suffix('.best.pt')}")
    print(f"unique plans from random z: {div['unique']}/{div['n']}  "
          f"(end-to-end RL decoder scored 64/256 on 24x24_empty, 60/256 on room_v1)")
    print(f"plan length from random z: {div['len_mean']:.2f}+/-{div['len_std']:.2f}, "
          f"{div['len_unique']} distinct values  "
          f"(end-to-end RL decoder emits exactly 3.000 on every map and state)")
    print(f"saved: {out}")
    print(f"WALL_CLOCK_HOURS {(time.time() - t0) / 3600:.4f}")


if __name__ == "__main__":
    main()
