# -*- coding: utf-8 -*-
"""Generate HR/LR training pairs WITH diacritic masks for super-resolution.

The trick that makes this Vietnamese-specific: every word is rendered
TWICE with identical font, position, colours and background — once with
full diacritics, once stripped to bare base letters. The pixel difference
of the two renders is an exact, free segmentation mask of the diacritic
marks. No annotation, no heuristics. The mask then drives a weighted SR
loss (and the Diacritic-PSNR metric) so the network is graded precisely on
the few pixels that decide ỏ vs õ vs o.

Output layout:
    <out>/hr/0000/pair_00000000.png    32x128, clean target
    <out>/lr/0000/pair_00000000.png    16x64, degraded input
    <out>/mask/0000/pair_00000000.png  32x128, binary diacritic mask
    <out>/pairs_gt.jsonl               {"file", "text", "has_marks"}

    python tools/gen_sr_pairs.py --out data/sr_pairs --count 120000 \
        --fonts fonts --corpus data/vintext_words/train_gt.jsonl \
        assets/general_dict.txt --workers 8 --tone-balance
"""
import argparse
import json
import os
import random
import sys
import unicodedata
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viettdr.vietchar import (decompose_char, MOD_NONE, TONE_LEVEL)  # noqa: E402
import gen_synth as G                                                # noqa: E402

HR_W, HR_H = 128, 32
LR_W, LR_H = 64, 16
SHARD = 2000
_W = {}


def strip_marks(text):
    """ệ -> e, ắ -> a, đ -> d: bare base letters, same advance widths."""
    return "".join(decompose_char(c)[0] for c in text)


def has_marks(text):
    return any(decompose_char(c)[1] != MOD_NONE or
               decompose_char(c)[2] != TONE_LEVEL for c in text)


# ---------------------------------------------------------------- rendering
def render_pair(text, rng):
    """Return (hr_full, hr_stripped) rendered identically, or None."""
    fonts = _W["fonts"]
    bg_pool = _W["bg"]
    size = rng.randint(24, 44)
    font = ImageFont.truetype(rng.choice(fonts), size)

    pad_x, pad_y = rng.randint(4, 10), rng.randint(3, 8)
    l, t, r, b = font.getbbox(text)          # bbox of the FULL text
    tw, th = max(1, r - l), max(1, b - t)
    W, H = tw + 2 * pad_x, th + 2 * pad_y
    if W > 1600 or H > 220:
        return None

    img, base = G.make_background(rng, W, H, bg_pool)
    img2 = img.copy()                        # identical background
    fg = G.contrasting(rng, base or (128, 128, 128))
    xy = (pad_x - l, pad_y - t)
    ImageDraw.Draw(img).text(xy, text, font=font, fill=fg)
    ImageDraw.Draw(img2).text(xy, strip_marks(text), font=font, fill=fg)

    if rng.random() < 0.3:                   # same rotation on both
        ang = rng.uniform(-3, 3)
        fill = base or (128, 128, 128)
        img = img.rotate(ang, resample=Image.BILINEAR, expand=True,
                         fillcolor=fill)
        img2 = img2.rotate(ang, resample=Image.BILINEAR, expand=True,
                           fillcolor=fill)
    return img, img2


def degrade(hr, rng):
    """HR 32x128 -> realistic LR 16x64. Matches what small VinText crops
    actually look like: optical blur, sensor noise, jpeg blocking."""
    img = hr
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.8)))
    # sometimes fall below LR size first (very small originals) then back up
    if rng.random() < 0.4:
        h2 = rng.randint(8, LR_H)
        w2 = max(8, int(LR_W * h2 / LR_H))
        img = img.resize((w2, h2), Image.BILINEAR)
    img = img.resize((LR_W, LR_H),
                     rng.choice([Image.BILINEAR, Image.BICUBIC, Image.BOX]))
    if rng.random() < 0.5:
        a = np.asarray(img, np.int16) + np.random.randint(
            -14, 14, (LR_H, LR_W, 3), dtype=np.int16)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(35, 80))
        img = Image.open(buf).convert("RGB")
    return img


def make_sample(job):
    idx, out_dir, thresh = job
    rng = random
    for _ in range(4):
        text = G.pick_text(rng) if hasattr(G, "pick_text") else None
        if text is None:
            text = rng.choice(_W["corpus"])
        if rng.random() < 0.25:
            text = rng.choice([text.upper(), text.lower(), text.title()])
        try:
            pair = render_pair(text, rng)
        except Exception:                    # noqa: BLE001
            pair = None
        if pair is not None:
            break
    else:
        return None
    full, stripped = pair

    # mask from the double render, BEFORE resizing (marks are crisp here)
    d = np.abs(np.asarray(full, np.int16) - np.asarray(stripped, np.int16))
    m = (d.max(axis=2) > thresh).astype(np.uint8) * 255
    mask = Image.fromarray(m).filter(ImageFilter.MaxFilter(3))

    hr = full.resize((HR_W, HR_H), Image.LANCZOS)
    mask = mask.resize((HR_W, HR_H), Image.BILINEAR).point(
        lambda v: 255 if v > 64 else 0)
    lr = degrade(hr, rng)

    rel = f"{idx // SHARD:04d}/pair_{idx:08d}.png"
    for sub, im in [("hr", hr), ("lr", lr), ("mask", mask)]:
        p = os.path.join(out_dir, sub, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        im.save(p)
    return json.dumps({"file": rel, "text": text,
                       "has_marks": has_marks(text)}, ensure_ascii=False)


def init_worker(fonts, bg_paths, corpus, buckets, balance, seed, n_bg_cache):
    G.init_worker(fonts, bg_paths, corpus, buckets, balance, seed, n_bg_cache)
    _W["fonts"] = fonts
    _W["bg"] = G._G["bg"]
    _W["corpus"] = corpus
    random.seed(seed + os.getpid())
    np.random.seed((seed + os.getpid()) % (2 ** 32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=120000)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--bg-cache", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--mask-thresh", type=int, default=16)
    ap.add_argument("--tone-balance", action="store_true")
    ap.add_argument("--balance-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fonts = [os.path.join(args.fonts, f) for f in sorted(os.listdir(args.fonts))
             if f.lower().endswith((".ttf", ".otf"))]
    assert fonts, f"no fonts in {args.fonts}"
    rng = random.Random(args.seed)
    words = G.build_corpus(args.corpus, args.max_len)
    words += G.synthetic_strings(max(1, len(words) // 5), rng)
    buckets = G.tone_buckets(words) if args.tone_balance else {}

    bg = []
    if args.bg_dir and os.path.isdir(args.bg_dir):
        bg = [os.path.join(args.bg_dir, f)
              for f in sorted(os.listdir(args.bg_dir))[:3000]
              if f.lower().endswith((".jpg", ".png"))]
    print(f"fonts {len(fonts)} | corpus {len(words)} | backgrounds {len(bg)} "
          f"| tone-balance {args.tone_balance}")

    os.makedirs(args.out, exist_ok=True)
    gt = os.path.join(args.out, "pairs_gt.jsonl")
    jobs = [(i, args.out, args.mask_thresh) for i in range(args.count)]
    n_ok = n_marks = 0
    with open(gt, "w", encoding="utf-8") as f, \
            Pool(args.workers, initializer=init_worker,
                 initargs=(fonts, bg, words, buckets,
                           args.balance_ratio if args.tone_balance else 0.0,
                           args.seed, args.bg_cache)) as pool:
        for i, line in enumerate(pool.imap_unordered(make_sample, jobs, 128)):
            if line:
                f.write(line + "\n")
                n_ok += 1
                n_marks += json.loads(line)["has_marks"]
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{args.count}")
    print(f"done: {n_ok} pairs ({n_marks} with diacritics = "
          f"{100 * n_marks / max(n_ok, 1):.0f}%) -> {args.out}")


if __name__ == "__main__":
    main()
