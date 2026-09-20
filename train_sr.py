# -*- coding: utf-8 -*-
"""Train the diacritic-aware SR module.

Losses (each one is an ablation axis for the paper):
    (a) Charbonnier on all pixels                       -- always on
    (b) + diacritic-mask-weighted Charbonnier           -- --lam-mask
    (c) + recognition-component loss from a FROZEN      -- --rec-ckpt
        decomposition recognizer: the SR image must let the recognizer
        reproduce the ground-truth (base, modifier, tone) sequences.
        This is where the triple decomposition earns its keep: the tone
        head supplies gradient aimed exactly at the mark pixels.

    python train_sr.py --data data/sr_pairs --out runs/sr_full \
        --rec-ckpt runs/full/best.pth --charset charset_vintext.txt
"""
import argparse
import csv
import json
import os
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from viettdr.sr_model import VietSR, SRLoss, psnr, psnr_masked
from viettdr.vietchar import Charset
from viettdr.tokenizer import TripleTokenizer, FlatTokenizer
from viettdr.model import VietTDR, TripleLoss

import numpy as np


class SRPairs(Dataset):
    def __init__(self, root, tokenizer=None, max_len=25):
        self.root = root
        self.recs = [json.loads(l) for l in
                     open(os.path.join(root, "pairs_gt.jsonl"),
                          encoding="utf-8")]
        if tokenizer is not None:
            self.recs = [r for r in self.recs
                         if tokenizer.can_encode(r["text"])]
        self.tok = tokenizer

    def __len__(self):
        return len(self.recs)

    def _img(self, sub, rel):
        p = os.path.join(self.root, sub, rel)
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)

    def __getitem__(self, i):
        r = self.recs[i]
        lr = self._img("lr", r["file"])
        hr = self._img("hr", r["file"])
        m = np.asarray(Image.open(os.path.join(
            self.root, "mask", r["file"])).convert("L"), np.float32) / 255.0
        mask = torch.from_numpy(m)[None]
        if self.tok is None:
            return lr, hr, mask, r["text"]
        (ib, im_, it), (tb, tm, tt) = self.tok.encode(r["text"])
        return lr, hr, mask, r["text"], ib, im_, it, tb, tm, tt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--lam-mask", type=float, default=2.0)
    p.add_argument("--rec-ckpt", default=None,
                   help="frozen recognizer checkpoint for component loss")
    p.add_argument("--charset", default=None)
    p.add_argument("--lam-rec", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # frozen recognizer (optional)
    rec = tok = rec_crit = None
    if args.rec_ckpt:
        assert args.charset, "--rec-ckpt requires --charset"
        ck = torch.load(args.rec_ckpt, map_location="cpu", weights_only=False)
        targs = ck.get("args", {})
        cs = Charset.from_file(args.charset)
        Tok = FlatTokenizer if targs.get("no_decompose") else TripleTokenizer
        tok = Tok(cs, max_len=targs.get("max_len", 25))
        rec = VietTDR(tok.num_bases, max_len=tok.max_len,
                      dim=targs.get("dim", 384),
                      enc_depth=targs.get("enc_depth", 8),
                      dec_depth=targs.get("dec_depth", 2),
                      use_tone_prior=not targs.get("no_tone_prior", False))
        rec.load_state_dict(ck["model"])
        rec.to(device).eval()
        for q in rec.parameters():
            q.requires_grad_(False)
        rec_crit = TripleLoss(tok.valid_mask, tok.pad_id, lam_c=0.0).to(device)
        print(f"[rec] frozen recognizer from {args.rec_ckpt} "
              f"(val {ck.get('best', float('nan')):.4f})")

    ds = SRPairs(args.data, tokenizer=tok)
    n_val = max(64, int(len(ds) * args.val_frac))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(ds), generator=g).tolist()
    val_ds = Subset(ds, perm[:n_val])
    tr_ds = Subset(ds, perm[n_val:])
    tr = DataLoader(tr_ds, args.bs, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)
    va = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers)
    print(f"[data] train {len(tr_ds)} | val {n_val}")

    model = VietSR(ch=args.ch, blocks=args.blocks).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"[model] VietSR {n_par / 1e6:.2f}M params | device {device}")
    crit = SRLoss(lam_mask=args.lam_mask)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * max(1, len(tr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    start_ep, best = 0, -1e9
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"[resume] epoch {start_ep}")

    logp = os.path.join(args.out, "log.csv")
    if not os.path.exists(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "pix", "mark", "rec",
                                    "psnr", "dpsnr", "sec"])

    REC_MEAN = 0.5  # recognizer expects (x-0.5)/0.5

    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "pix": 0.0, "mark": 0.0, "rec": 0.0}
        nb = 0
        for batch in tr:
            if tok is None:
                lr_img, hr, mask, _ = batch
                rb = None
            else:
                lr_img, hr, mask, _, ib, im_, it, tb, tm, tt = batch
                rb = tuple(x.to(device) for x in (ib, im_, it, tb, tm, tt))
            lr_img = lr_img.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                sr = model(lr_img)
            loss, parts = crit(sr.float(), hr.float(), mask.float())
            if rec is not None:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    logits = rec((sr - REC_MEAN) / REC_MEAN,
                                 rb[0], rb[1], rb[2])
                rl, _ = rec_crit(tuple(l.float() for l in logits),
                                 (rb[3], rb[4], rb[5]))
                loss = loss + args.lam_rec * rl
                agg["rec"] += rl.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg["loss"] += loss.item()
            agg["pix"] += parts["pix"]
            agg["mark"] += parts["mark"]
            nb += 1
        for k in agg:
            agg[k] /= max(nb, 1)

        # ---- validation: PSNR + Diacritic-PSNR
        model.eval()
        ps, dps = [], []
        with torch.no_grad():
            for batch in va:
                lr_img, hr, mask = batch[0].to(device), batch[1].to(device), \
                    batch[2].to(device)
                sr = model(lr_img)
                ps.append(psnr(sr, hr))
                d = psnr_masked(sr, hr, mask)
                if d == d:
                    dps.append(d)
        mps = sum(ps) / max(len(ps), 1)
        mdps = sum(dps) / max(len(dps), 1)
        dt = time.time() - t0
        print(f"ep {ep:3d} | loss {agg['loss']:.4f} (pix {agg['pix']:.4f} "
              f"mark {agg['mark']:.4f} rec {agg['rec']:.3f}) | "
              f"PSNR {mps:.2f} D-PSNR {mdps:.2f} | {dt:.0f}s")
        with open(logp, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{agg['loss']:.4f}",
                                    f"{agg['pix']:.4f}", f"{agg['mark']:.4f}",
                                    f"{agg['rec']:.4f}", f"{mps:.2f}",
                                    f"{mdps:.2f}", f"{dt:.0f}"])
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": ep, "best": best,
                 "args": vars(args)}
        torch.save(state, os.path.join(args.out, "last.pth"))
        score = mdps if dps else mps
        if score >= best:
            best = score
            state["best"] = best
            torch.save(state, os.path.join(args.out, "best.pth"))
            print(f"  ** new best D-PSNR {best:.2f}")

    print(f"done. best D-PSNR = {best:.2f}")


if __name__ == "__main__":
    main()
