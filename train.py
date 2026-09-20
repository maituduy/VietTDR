# -*- coding: utf-8 -*-
"""Train VietTDR on cropped word images.

Baseline (no decomposition benefit switches):
    python train.py --data data/vintext_words --charset charset_vintext.txt \
        --out runs/tdr_base

Ablations:
    --no-tone-prior         disable the tone-prior attention branch
    --lam-c 0.0             disable the composition-validity loss
    --synth-jsonl/--synth-dir  add synthetic pretraining data
"""
import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from viettdr.vietchar import Charset
from viettdr.tokenizer import TripleTokenizer, FlatTokenizer
from viettdr.model import VietTDR, TripleLoss
from viettdr.dataset import WordDataset, collate
from viettdr.metrics import Scoreboard


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="dir with train/, val/, train_gt.jsonl, val_gt.jsonl")
    p.add_argument("--charset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--synth-jsonl", default=None)
    p.add_argument("--synth-dir", default=None)
    p.add_argument("--synth-aug", default="light",
                   choices=["full", "light", "none"],
                   help="augmentation level for synthetic images (they are "
                        "already degraded when generated)")
    p.add_argument("--resume", default=None,
                   help="continue an interrupted run (restores optimizer/epoch)")
    p.add_argument("--init-from", default=None,
                   help="load ONLY model weights and start a fresh schedule; "
                        "this is stage-2 fine-tuning after synthetic pretraining")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=192)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--max-len", type=int, default=25)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--enc-depth", type=int, default=8)
    p.add_argument("--dec-depth", type=int, default=2)
    p.add_argument("--lam-c", type=float, default=0.2)
    p.add_argument("--no-tone-prior", action="store_true")
    p.add_argument("--no-decompose", action="store_true",
                   help="ablation: single 229-way head, no triple decomposition")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (on by default on CUDA)")
    p.add_argument("--eval-every", type=int, default=1,
                   help="run validation every N epochs")
    p.add_argument("--val-subset", type=int, default=2000,
                   help="validate on this many val samples (0 = all)")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    charset = Charset.from_file(args.charset)
    if args.no_decompose:
        tok = FlatTokenizer(charset, max_len=args.max_len)
        # degenerate modifier/tone heads must not contribute to the objective
        args.lam_c = 0.0
        args.no_tone_prior = True
        print(f"[ablation] no decomposition: flat vocabulary of "
              f"{tok.num_bases} classes")
    else:
        tok = TripleTokenizer(charset, max_len=args.max_len)

    train_jsonl = [os.path.join(args.data, "train_gt.jsonl")]
    train_dirs = [os.path.join(args.data, "train")]
    levels = ["full"]
    if args.synth_jsonl:
        train_jsonl.append(args.synth_jsonl)
        train_dirs.append(args.synth_dir)
        # synthetic images are already degraded at generation time
        levels.append(args.synth_aug)
    train_ds = WordDataset(train_jsonl, train_dirs, tok, train=True,
                           levels=levels)
    val_ds = WordDataset(os.path.join(args.data, "val_gt.jsonl"),
                         os.path.join(args.data, "val"), tok, train=False)
    train_ld = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers > 0 else None)
    # Validation uses autoregressive greedy decoding, which costs ~26 decoder
    # passes per batch; on a small GPU that dominates epoch time. Evaluate on
    # an evenly-strided subset during training and keep the full set for eval.py.
    if 0 < args.val_subset < len(val_ds):
        stride = len(val_ds) // args.val_subset
        val_eval = Subset(val_ds, list(range(0, len(val_ds), stride)))
    else:
        val_eval = val_ds
    val_ld = DataLoader(val_eval, args.bs, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    model = VietTDR(tok.num_bases, max_len=args.max_len, dim=args.dim,
                    enc_depth=args.enc_depth, dec_depth=args.dec_depth,
                    use_tone_prior=not args.no_tone_prior).to(device)
    lam_mt = 0.0 if args.no_decompose else 1.0
    crit = TripleLoss(tok.valid_mask, tok.pad_id, lam_m=lam_mt, lam_t=lam_mt,
                      lam_c=args.lam_c).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M params | device={device} | "
          f"train={len(train_ds)} val={len(val_ds)}")

    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_ld))
    warm = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm else
        0.5 * (1 + torch.cos(torch.tensor(
            (s - warm) / max(total - warm, 1) * 3.141592653589793)).item()))

    start_ep, best = 0, 0.0
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
        if missing:
            print(f"[init] {len(missing)} params kept at init: {missing[:3]}")
        print(f"[init] weights from {args.init_from} "
              f"(pretrained {ck.get('epoch', '?')} epochs, "
              f"val {ck.get('best', float('nan')):.4f}); optimizer reset")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best = ck["epoch"] + 1, ck.get("best", 0.0)
        print(f"[resume] epoch {start_ep}, best {best:.4f}")

    log_path = os.path.join(args.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "base", "mod", "tone", "comp",
                 "val_acc", "val_1_ned", "val_heavy_acc", "lr", "sec"])

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "base": 0.0, "mod": 0.0, "tone": 0.0, "comp": 0.0}
        for imgs, ib, im_, it, tb, tm, tt, _ in train_ld:
            imgs = imgs.to(device, non_blocking=True)
            ib, im_, it = ib.to(device), im_.to(device), it.to(device)
            tb, tm, tt = tb.to(device), tm.to(device), tt.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(imgs, ib, im_, it)
            # the composition loss multiplies three softmaxes; keep it in fp32
            loss, parts = crit(tuple(l.float() for l in logits), (tb, tm, tt))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg["loss"] += loss.item()
            for k in parts:
                agg[k] += parts[k]
        for k in agg:
            agg[k] /= steps_per_epoch

        # ---- validation
        do_eval = ((ep + 1) % args.eval_every == 0) or (ep == args.epochs - 1)
        if do_eval:
            model.eval()
            board = Scoreboard()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=use_amp):
                for imgs, *_rest, texts in val_ld:
                    b, m, t = model.greedy_decode(imgs.to(device), tok)
                    for i, gt in enumerate(texts):
                        board.add(tok.decode(b[i], m[i], t[i]), gt)
            s = board.summary()
        else:
            s = {"word_acc": float("nan"), "heavy_acc": float("nan"),
                 "one_minus_ned": float("nan")}
        lr_now = sched.get_last_lr()[0]
        dt = time.time() - t0
        print(f"ep {ep:3d} | loss {agg['loss']:.4f} "
              f"(b {agg['base']:.3f} m {agg['mod']:.3f} t {agg['tone']:.3f} "
              f"c {agg['comp']:.3f}) | word {s['word_acc']:.4f} "
              f"1-NED {s['one_minus_ned']:.4f} heavy {s['heavy_acc']:.4f}"
              f" | lr {lr_now:.2e} | {dt:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, f"{agg['loss']:.4f}", f"{agg['base']:.4f}",
                 f"{agg['mod']:.4f}", f"{agg['tone']:.4f}",
                 f"{agg['comp']:.4f}", f"{s['word_acc']:.4f}",
                 f"{s['one_minus_ned']:.4f}", f"{s['heavy_acc']:.4f}",
                 f"{lr_now:.3e}", f"{dt:.0f}"])

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": ep, "best": best,
                 "args": vars(args)}
        torch.save(state, os.path.join(args.out, "last.pth"))
        # ">=" so best.pth exists after the FIRST eval even at 0.0 word acc —
        # early-stage runs (e.g. a short pretrain) must still produce a
        # checkpoint for --init-from to consume.
        if do_eval and s["word_acc"] >= best:
            best = s["word_acc"]
            state["best"] = best
            torch.save(state, os.path.join(args.out, "best.pth"))
            print(f"  ** new best {best:.4f}")

    print(f"done. best val word acc = {best:.4f}")


if __name__ == "__main__":
    main()
