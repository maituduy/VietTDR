# -*- coding: utf-8 -*-
"""Self-paired REAL training data for the SR module.

Round 3 measured the problem precisely: SR trained purely on synthetic
pairs improves PSNR by +3 dB yet costs 6-7 word-accuracy points on real
low-res crops — a synthetic-to-real domain gap. Vietnamese has no TextZoom,
but it does not need one: about half of the VinText crops are >=28 px tall.
Those become HR targets, and the same degradation family used for the
synthetic pairs turns them into LR inputs. Image statistics are now real;
only the degradation stays synthetic.

Diacritic masks cannot be derived for real images, so the mask channel is
zero and these samples simply contribute nothing to the mask-weighted term
(the loss normalises by mask area over the batch).

    python tools/make_real_pairs.py --crops data/vintext_words --split train \
        --out data/sr_mix --min-h 28 --append
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen_sr_pairs import degrade, HR_W, HR_H, has_marks   # noqa: E402

SHARD = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="vintext_words root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-h", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing pairs_gt.jsonl (mix with "
                         "synthetic pairs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    gt_in = os.path.join(args.crops, f"{args.split}_gt.jsonl")
    img_dir = os.path.join(args.crops, args.split)
    recs = [json.loads(l) for l in open(gt_in, encoding="utf-8")]

    mode = "a" if args.append else "w"
    gt_out = os.path.join(args.out, "pairs_gt.jsonl")
    os.makedirs(args.out, exist_ok=True)
    zero_mask = Image.new("L", (HR_W, HR_H), 0)

    n = 0
    with open(gt_out, mode, encoding="utf-8") as f:
        for r in recs:
            p = os.path.join(img_dir, r["file"])
            try:
                im = Image.open(p).convert("RGB")
            except Exception:                  # noqa: BLE001
                continue
            if im.height < args.min_h:
                continue
            hr = im.resize((HR_W, HR_H), Image.LANCZOS)
            lr = degrade(hr, random)
            rel = f"real{n // SHARD:04d}/real_{n:08d}.png"
            for sub, img in [("hr", hr), ("lr", lr), ("mask", zero_mask)]:
                q = os.path.join(args.out, sub, rel)
                os.makedirs(os.path.dirname(q), exist_ok=True)
                img.save(q)
            f.write(json.dumps({"file": rel, "text": r["text"],
                                "has_marks": has_marks(r["text"]),
                                "real": True}, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"real pairs: {n} (tu crop >= {args.min_h}px cua {args.split})"
          f" -> {args.out} ({'append' if args.append else 'new'})")


if __name__ == "__main__":
    main()
