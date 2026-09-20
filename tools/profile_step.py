# -*- coding: utf-8 -*-
"""Separate the GPU cost of one training step from the data-loading cost.

Round 1 and round 2 both landed at ~0.38 s/step despite round 2 using 5.2x
cheaper augmentation, which says augmentation is not the bottleneck. This
script settles what is: it runs the identical forward/backward on tensors
that are already resident on the GPU (no DataLoader, no JPEG decode, no
augmentation), then runs the real DataLoader for comparison.

    python tools/profile_step.py --charset charset_vintext.txt \
        --data data/vintext_words --synth-jsonl data/synth/synth_gt.jsonl \
        --synth-dir data/synth --bs 192

Read the output as:
  gpu_only ~= observed  -> GPU-bound; caching images buys nothing, look at
                           batch size / torch.compile / attention fast path
  gpu_only << observed  -> input-bound; pre-resizing into a memmap pays off
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from viettdr.vietchar import Charset                    # noqa: E402
from viettdr.tokenizer import TripleTokenizer           # noqa: E402
from viettdr.model import VietTDR, TripleLoss           # noqa: E402
from viettdr.dataset import WordDataset, collate        # noqa: E402


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--synth-jsonl", default=None)
    ap.add_argument("--synth-dir", default=None)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--enc-depth", type=int, default=8)
    ap.add_argument("--dec-depth", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          f"{' - ' + torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    tok = TripleTokenizer(Charset.from_file(args.charset), max_len=args.max_len)
    model = VietTDR(tok.num_bases, max_len=args.max_len, dim=args.dim,
                    enc_depth=args.enc_depth, dec_depth=args.dec_depth).to(device)
    crit = TripleLoss(tok.valid_mask, tok.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par / 1e6:.2f}M params | batch {args.bs}")

    T = args.max_len + 1
    B = args.bs
    imgs = torch.randn(B, 3, 32, 128, device=device)
    ib = torch.randint(0, tok.charset.num_bases, (B, T), device=device)
    im_ = torch.randint(0, 5, (B, T), device=device)
    it = torch.randint(0, 6, (B, T), device=device)
    tb, tm, tt = ib.clone(), im_.clone(), it.clone()

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(imgs, ib, im_, it)
        loss, _ = crit(tuple(l.float() for l in logits), (tb, tm, tt))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    gpu_only = bench(step)
    print(f"\n  GPU-only (tensor san tren GPU) : {gpu_only * 1000:7.1f} ms/buoc "
          f"-> {B / gpu_only:6.0f} anh/s")

    def fwd_only():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=use_amp):
            model(imgs, ib, im_, it)
    fo = bench(fwd_only)
    print(f"  chi forward                    : {fo * 1000:7.1f} ms/buoc")

    if args.synth_jsonl and os.path.exists(args.synth_jsonl):
        ds = WordDataset([args.synth_jsonl], [args.synth_dir], tok,
                         train=True, levels=["light"])
        ld = DataLoader(ds, B, shuffle=True, drop_last=True,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)
        it_ld = iter(ld)
        next(it_ld)                                  # warm the workers
        t0 = time.time()
        n = 0
        for _ in range(20):
            try:
                batch = next(it_ld)
            except StopIteration:
                break
            n += 1
        load_only = (time.time() - t0) / max(n, 1)
        print(f"  chi nap du lieu (synth, light) : {load_only * 1000:7.1f} ms/buoc "
              f"-> {B / load_only:6.0f} anh/s")
        print(f"\n  ket luan: {'GPU-BOUND - cache anh vo ich' if gpu_only > load_only else 'INPUT-BOUND - cache anh se giup'}")
    else:
        print("\n  (bo qua do nap du lieu: khong co --synth-jsonl)")


if __name__ == "__main__":
    main()
