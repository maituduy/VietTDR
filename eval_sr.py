# -*- coding: utf-8 -*-
"""Evaluate the SR module.

Two protocols:

1. synth  — held-out synthetic pairs: PSNR / SSIM / Diacritic-PSNR for
   bicubic vs SR, plus recognizer accuracy on HR / bicubic / SR when a
   recognizer checkpoint is given (oracle / lower bound / ours).

2. real   — the protocol that needs no paired ground truth: take REAL
   VinText crops no taller than --max-h pixels (the slice where 75% of
   remaining recognition errors live), run the recognizer on the plain
   crop vs the SR-enhanced crop, and compare word accuracy. The delta is
   the downstream, deployment-relevant number.

    python eval_sr.py synth --data data/sr_pairs --ckpt runs/sr_full/best.pth \
        --rec-ckpt runs/full/best.pth --charset charset_vintext.txt
    python eval_sr.py real --data data/vintext_words --split test --max-h 16 \
        --ckpt runs/sr_full/best.pth --rec-ckpt runs/full/best.pth \
        --charset charset_vintext.txt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from viettdr.sr_model import VietSR, psnr, psnr_masked
from viettdr.vietchar import Charset
from viettdr.tokenizer import TripleTokenizer, FlatTokenizer
from viettdr.model import VietTDR
from viettdr.metrics import Scoreboard

LR_W, LR_H = 64, 16
HR_W, HR_H = 128, 32


def load_sr(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    m = VietSR(ch=a.get("ch", 64), blocks=a.get("blocks", 8))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def load_rec(path, charset_path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    cs = Charset.from_file(charset_path)
    Tok = FlatTokenizer if targs.get("no_decompose") else TripleTokenizer
    tok = Tok(cs, max_len=targs.get("max_len", 25))
    rec = VietTDR(tok.num_bases, max_len=tok.max_len,
                  dim=targs.get("dim", 384),
                  enc_depth=targs.get("enc_depth", 8),
                  dec_depth=targs.get("dec_depth", 2),
                  use_tone_prior=not targs.get("no_tone_prior", False))
    rec.load_state_dict(ck["model"])
    return rec.to(device).eval(), tok


def to_tensor(img):
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def rec_accuracy(rec, tok, imgs01, texts, device, bs=256):
    """imgs01: list of 3xHxW tensors in [0,1] already at 32x128."""
    board = Scoreboard()
    with torch.no_grad():
        for i in range(0, len(imgs01), bs):
            x = torch.stack(imgs01[i:i + bs]).to(device)
            x = (x - 0.5) / 0.5
            b, m, t = rec.greedy_decode(x, tok)
            for j, gt in enumerate(texts[i:i + bs]):
                board.add(tok.decode(b[j], m[j], t[j]), gt)
    return board.summary()


def ssim(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    """Single-scale SSIM on batched tensors, 7x7 uniform window."""
    w = torch.ones(3, 1, 7, 7, device=a.device) / 49.0
    mu_a = F.conv2d(a, w, padding=3, groups=3)
    mu_b = F.conv2d(b, w, padding=3, groups=3)
    sa = F.conv2d(a * a, w, padding=3, groups=3) - mu_a ** 2
    sb = F.conv2d(b * b, w, padding=3, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=3, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return s.mean().item()


def eval_synth(args, device):
    sr_net = load_sr(args.ckpt, device)
    rec = tok = None
    if args.rec_ckpt:
        rec, tok = load_rec(args.rec_ckpt, args.charset, device)

    recs = [json.loads(l) for l in
            open(os.path.join(args.data, "pairs_gt.jsonl"), encoding="utf-8")]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(recs), generator=g).tolist()
    recs = [recs[i] for i in perm[:args.limit]]

    stats = {"bicubic": {"psnr": [], "dpsnr": [], "ssim": []},
             "sr": {"psnr": [], "dpsnr": [], "ssim": []}}
    imgs = {"hr": [], "bicubic": [], "sr": []}
    texts = []
    for i in range(0, len(recs), args.bs):
        chunk = recs[i:i + args.bs]
        lr = torch.stack([to_tensor(Image.open(
            os.path.join(args.data, "lr", r["file"]))) for r in chunk]).to(device)
        hr = torch.stack([to_tensor(Image.open(
            os.path.join(args.data, "hr", r["file"]))) for r in chunk]).to(device)
        mk = torch.stack([to_tensor(Image.open(
            os.path.join(args.data, "mask", r["file"])))[:1] for r in chunk]).to(device)
        with torch.no_grad():
            up = F.interpolate(lr, size=(HR_H, HR_W), mode="bicubic",
                               align_corners=False).clamp(0, 1)
            out = sr_net(lr)
        for name, im in [("bicubic", up), ("sr", out)]:
            stats[name]["psnr"].append(psnr(im, hr))
            d = psnr_masked(im, hr, mk)
            if d == d:
                stats[name]["dpsnr"].append(d)
            stats[name]["ssim"].append(ssim(im, hr))
        if rec is not None:
            for r in chunk:
                texts.append(r["text"])
            for name, im in [("hr", hr), ("bicubic", up), ("sr", out)]:
                imgs[name].extend([t.cpu() for t in im])

    print(f"synth eval ({len(recs)} pairs):")
    print(f"{'':10} {'PSNR':>7} {'SSIM':>7} {'D-PSNR':>8}")
    for name in ["bicubic", "sr"]:
        s = stats[name]
        print(f"  {name:8} {np.mean(s['psnr']):7.2f} {np.mean(s['ssim']):7.4f} "
              f"{np.mean(s['dpsnr']):8.2f}")
    if rec is not None:
        print("\nrecognizer word acc:")
        for name in ["hr", "bicubic", "sr"]:
            s = rec_accuracy(rec, tok, imgs[name], texts, device)
            print(f"  {name:8} word {s['word_acc']:.4f} | "
                  f"co dau {s['any_diac_acc']:.4f} | hoi/nga "
                  f"{s['confusable_acc']:.4f}")


def eval_real(args, device):
    sr_net = load_sr(args.ckpt, device)
    rec, tok = load_rec(args.rec_ckpt, args.charset, device)

    gt_path = os.path.join(args.data, f"{args.split}_gt.jsonl")
    img_dir = os.path.join(args.data, args.split)
    small, texts = [], []
    n_all = 0
    for line in open(gt_path, encoding="utf-8"):
        r = json.loads(line)
        n_all += 1
        p = os.path.join(img_dir, r["file"])
        with Image.open(p) as im:
            if im.height <= args.max_h and tok.can_encode(r["text"]):
                small.append(im.convert("RGB").copy())
                texts.append(r["text"])
    print(f"real eval: {len(small)}/{n_all} crops with height <= {args.max_h}px")

    plain, enhanced = [], []
    for im in small:
        # baseline: recognizer's own preprocessing (resize whatever -> 32x128)
        plain.append(to_tensor(im.resize((HR_W, HR_H), Image.BICUBIC)))
        if args.aspect:
            # aspect-preserving: only the height is normalised to the LR
            # grid; the network is fully convolutional so width can vary.
            # Squeezing a 15x110 crop to 16x64 first (the fixed protocol)
            # throws away ~2x horizontal detail BEFORE the SR net ever
            # sees it -- an unfair handicap the baseline does not pay.
            w = max(16, round(im.width * LR_H / im.height))
            w += w % 2                      # even width for pixelshuffle
            lr = to_tensor(im.resize((w, LR_H), Image.BICUBIC))
        else:
            lr = to_tensor(im.resize((LR_W, LR_H), Image.BICUBIC))
        with torch.no_grad():
            out = sr_net(lr[None].to(device))[0].cpu()
        # recognizer input must be 32x128 regardless of SR output width
        out = F.interpolate(out[None], size=(HR_H, HR_W), mode="bicubic",
                            align_corners=False)[0].clamp(0, 1)
        enhanced.append(out)

    for name, imgs in [("goc (bicubic)", plain), ("qua SR", enhanced)]:
        s = rec_accuracy(rec, tok, imgs, texts, device)
        print(f"  {name:14} word {s['word_acc']:.4f} | 1-NED "
              f"{s['one_minus_ned']:.4f} | co dau {s['any_diac_acc']:.4f} | "
              f"hoi/nga {s['confusable_acc']:.4f} (n={s['confusable_n']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["synth", "real"])
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rec-ckpt", default=None)
    p.add_argument("--charset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max-h", type=int, default=16)
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--aspect", action="store_true",
                   help="real mode: keep aspect ratio of the LR input (fully-conv SR)")
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "synth":
        eval_synth(args, device)
    else:
        assert args.rec_ckpt and args.charset, "real mode requires recognizer"
        eval_real(args, device)


if __name__ == "__main__":
    main()
