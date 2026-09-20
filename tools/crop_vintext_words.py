# -*- coding: utf-8 -*-
"""Crop word images out of VinText for recognition training.

VinText layout (https://github.com/VinAIResearch/dict-guided):
    labels/gt_N.txt   : x1,y1,x2,y2,x3,y3,x4,y4,transcript   (### = ignore)
    train_images/imNNNN.jpg  N in [1, 1200]
    val_images/            N in [1201, 1500]
    test_images/           N in [1501, 2000]

Each quadrilateral is perspective-rectified to a horizontal strip using
PIL's QUAD transform. Output:

    <out>/<split>/word_xxxxxx.jpg
    <out>/<split>_gt.jsonl      {"file", "text", "src", "quad"}

Usage:
    python tools/crop_vintext_words.py \
        --vintext ../DeepSolo/DeepSolo-main/DeepSolo/datasets/vintext \
        --out data/vintext_words
"""
import argparse
import json
import math
import os
import unicodedata

from PIL import Image

SPLITS = {"train": (1, 1200, "train_images"),
          "val": (1201, 1500, "val_images"),
          "test": (1501, 2000, "test_images")}
MIN_SIDE = 8          # skip crops whose rectified height is tiny
MAX_AR = 40.0         # skip absurd aspect ratios (annotation errors)


def parse_gt(path):
    items = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            text = ",".join(parts[8:])
            items.append((coords, text))
    return items


def quad_size(q):
    """Average width/height of quad q = [x1..y4] in TL,TR,BR,BL order."""
    d = lambda ax, ay, bx, by: math.hypot(ax - bx, ay - by)
    w = (d(q[0], q[1], q[2], q[3]) + d(q[6], q[7], q[4], q[5])) / 2
    h = (d(q[0], q[1], q[6], q[7]) + d(q[2], q[3], q[4], q[5])) / 2
    return w, h


def crop_quad(img, q):
    w, h = quad_size(q)
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    # PIL QUAD wants NW, SW, SE, NE order
    quad = (q[0], q[1], q[6], q[7], q[4], q[5], q[2], q[3])
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BILINEAR), w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintext", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    stats = {}
    for split in args.splits:
        lo, hi, img_dir = SPLITS[split]
        out_img_dir = os.path.join(args.out, split)
        os.makedirs(out_img_dir, exist_ok=True)
        gt_path = os.path.join(args.out, f"{split}_gt.jsonl")
        n_kept = n_ignored = n_skipped = 0
        with open(gt_path, "w", encoding="utf-8") as gt_f:
            for n in range(lo, hi + 1):
                label = os.path.join(args.vintext, "labels", f"gt_{n}.txt")
                image = os.path.join(args.vintext, img_dir, f"im{n:04d}.jpg")
                if not (os.path.exists(label) and os.path.exists(image)):
                    continue
                img = Image.open(image).convert("RGB")
                for i, (coords, text) in enumerate(parse_gt(label)):
                    text = unicodedata.normalize("NFC", text.strip())
                    if not text or text == "###":
                        n_ignored += 1
                        continue
                    crop, w, h = crop_quad(img, coords)
                    if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > MAX_AR:
                        n_skipped += 1
                        continue
                    name = f"word_{n:04d}_{i:03d}.jpg"
                    crop.save(os.path.join(out_img_dir, name), quality=95)
                    gt_f.write(json.dumps(
                        {"file": name, "text": text,
                         "src": f"im{n:04d}.jpg", "quad": coords},
                        ensure_ascii=False) + "\n")
                    n_kept += 1
        stats[split] = (n_kept, n_ignored, n_skipped)
        print(f"[{split}] kept {n_kept}, ignored(###) {n_ignored}, "
              f"skipped(size) {n_skipped} -> {out_img_dir}")
    print("done:", stats)


if __name__ == "__main__":
    main()
