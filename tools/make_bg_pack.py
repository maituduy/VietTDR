# -*- coding: utf-8 -*-
"""Build a small pack of real background images for the synthetic generator.

Kaggle only has the word crops, not VinText's full scene images, so without
this pack gen_synth.py silently falls back to 100% artificial backgrounds
and loses the ~35% real-background mix. The generator itself downscales
backgrounds to <=900 px and crops windows of roughly 300x90 px at most, so
shipping the full-resolution images (hundreds of MB) is pointless — 640 px
JPEGs carry all the texture the crops will ever see.

    python tools/make_bg_pack.py \
        --src ../DeepSolo/DeepSolo-main/DeepSolo/datasets/vintext/train_images \
        --out bg_pack --count 500
"""
import argparse
import os
import random

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="VinText train_images dir")
    ap.add_argument("--out", default="bg_pack")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for f in sorted(os.listdir(args.src))
             if f.lower().endswith((".jpg", ".png"))]
    random.Random(args.seed).shuffle(files)
    os.makedirs(args.out, exist_ok=True)

    n = 0
    for f in files:
        if n >= args.count:
            break
        try:
            im = Image.open(os.path.join(args.src, f))
            im.draft("RGB", (args.max_side, args.max_side))
            im = im.convert("RGB")
            if max(im.size) > args.max_side:
                sc = args.max_side / max(im.size)
                im = im.resize((max(1, int(im.width * sc)),
                                max(1, int(im.height * sc))), Image.BILINEAR)
            if min(im.size) < 120:      # too small to crop windows from
                continue
            im.save(os.path.join(args.out, f"bg_{n:04d}.jpg"),
                    quality=args.quality)
            n += 1
        except Exception as e:          # noqa: BLE001
            print(f"  [skip] {f}: {e}")
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"{n} backgrounds -> {args.out}/ ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
