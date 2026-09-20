# -*- coding: utf-8 -*-
"""Fast synthetic Vietnamese word-image generator for recognizer pretraining.

Why not SynthTIGER: it renders a few images per second per core, so a
million samples costs more CPU-hours than a Kaggle session has. This
renderer does the same job for scene-text pretraining at ~1-3 ms/image, so
800k images take minutes on 4 cores.

The one thing it does that a generic renderer cannot: **tone rebalancing**.
On VinText the tone distribution is brutally skewed (level 79550, acute
3726, grave 3163, dot 3153, hook 1618, tilde 760). Because our labels are
decomposed, we can sample words so every tone is seen equally often —
directly attacking the hỏi/ngã confusion that the whole method targets.
That option is `--tone-balance`.

    python tools/gen_synth.py --out data/synth --count 800000 \
        --fonts fonts --corpus corpus_vi.txt --workers 4 --tone-balance
"""
import argparse
import json
import os
import random
import unicodedata
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from viettdr.vietchar import decompose_char, TONES, TONE_LEVEL  # noqa: E402

SHARD = 2000            # files per subdirectory; avoids 800k entries in one dir
_G = {}                 # per-worker globals (fonts are not picklable)


# --------------------------------------------------------------- corpus
def build_corpus(paths, max_len):
    """Read plain word lists and/or *_gt.jsonl label files.

    Caution: VinText ships two dictionaries and only one is usable here —
    `vn_dictionary.txt` is pure ASCII (no diacritics at all), so training on
    it would teach the model that Vietnamese has no tones. Use
    `general_dict.txt` and/or the real transcriptions in train_gt.jsonl.
    """
    words = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    words.append(json.loads(line)["text"])
        else:
            with open(p, encoding="utf-8") as f:
                words.extend(f.read().split())
    out = []
    for w in words:
        w = unicodedata.normalize("NFC", w.strip()).strip("()[]{}\"'")
        if 0 < len(w) <= max_len:
            out.append(w)
    return list(dict.fromkeys(out))


def synthetic_strings(n, rng):
    """Signage-flavoured strings a dictionary will not contain."""
    out = []
    for _ in range(n):
        k = rng.random()
        if k < 0.3:
            out.append(f"{rng.randint(1, 999)}.{rng.randint(0, 999):03d}Đ")
        elif k < 0.5:
            out.append(f"{rng.randint(1, 999)}{rng.choice('ABCDEGHKLMNPST')}"
                       f"-{rng.randint(1000, 99999)}")
        elif k < 0.7:
            out.append(f"{rng.randint(1, 300)}/{rng.randint(1, 99)}")
        elif k < 0.85:
            out.append(f"0{rng.randint(300000000, 999999999)}")
        else:
            out.append(str(rng.randint(1, 100000)))
    return out


def tone_buckets(words):
    """word -> which tones it contains; bucket per tone for rebalancing."""
    buckets = {t: [] for t in range(len(TONES))}
    for w in words:
        tones = {decompose_char(c)[2] for c in w}
        if tones == {TONE_LEVEL}:
            buckets[TONE_LEVEL].append(w)
        else:
            for t in tones - {TONE_LEVEL}:
                buckets[t].append(w)
    return {t: v for t, v in buckets.items() if v}


# --------------------------------------------------------------- rendering
def rand_color(rng):
    return tuple(rng.randint(0, 255) for _ in range(3))


def contrasting(rng, bg, min_gap=90):
    """Foreground colour with enough luminance distance from the background."""
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    for _ in range(12):
        fg = rand_color(rng)
        if abs(lum(fg) - lum(bg)) >= min_gap:
            return fg
    return (255, 255, 255) if lum(bg) < 128 else (0, 0, 0)


def make_background(rng, w, h, bg_pool):
    if bg_pool and rng.random() < 0.35:
        src = rng.choice(bg_pool)          # already-decoded PIL image in RAM
        if src.width > w and src.height > h:
            x = rng.randint(0, src.width - w)
            y = rng.randint(0, src.height - h)
            crop = src.crop((x, y, x + w, y + h))
            # report the crop's mean colour so the caller can still pick a
            # foreground that contrasts with it; without this, text drawn on
            # photo backgrounds comes out unreadable
            mean = tuple(int(v) for v in np.array(crop, np.float32)
                         .reshape(-1, 3).mean(0))
            return crop, mean
    base = rand_color(rng)
    img = Image.new("RGB", (w, h), base)
    r = rng.random()
    if r < 0.35:                                   # linear gradient
        other = rand_color(rng)
        arr = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        a = np.array(base, np.float32)[None, None, :]
        b = np.array(other, np.float32)[None, None, :]
        img = Image.fromarray(
            np.repeat(a * (1 - arr) + b * arr, h, axis=0).astype(np.uint8))
    elif r < 0.5:                                  # mild texture
        noise = np.random.randint(-18, 18, (h, w, 3), dtype=np.int16)
        img = Image.fromarray(
            np.clip(np.array(img, np.int16) + noise, 0, 255).astype(np.uint8))
    return img, base


def render_one(text, rng):
    """Render one word image.

    Deliberately lean: PIL already antialiases TrueType text, so there is no
    supersampling pass, and every resize is BILINEAR. Rendering at 2x and
    downsampling with LANCZOS looked marginally nicer but cost ~4x the pixels
    through rotate/blur/noise, which dominated generation time.
    """
    fonts = _G["fonts"]
    bg_pool = _G["bg"]
    size = rng.randint(14, 40)
    font = ImageFont.truetype(rng.choice(fonts), size)

    pad_x, pad_y = rng.randint(3, 12), rng.randint(2, 9)
    l, t, r, b = font.getbbox(text)
    tw, th = max(1, r - l), max(1, b - t)
    W, H = tw + 2 * pad_x, th + 2 * pad_y
    if W > 2000 or H > 400:
        return None

    img, base = make_background(rng, W, H, bg_pool)
    fg = contrasting(rng, base or (128, 128, 128))
    d = ImageDraw.Draw(img)
    x, y = pad_x - l, pad_y - t
    if rng.random() < 0.2:                          # drop shadow
        off = max(1, size // 22)
        d.text((x + off, y + off), text, font=font, fill=rand_color(rng))
    if rng.random() < 0.15:                         # outlined text
        d.text((x, y), text, font=font, fill=fg,
               stroke_width=max(1, size // 22), stroke_fill=rand_color(rng))
    else:
        d.text((x, y), text, font=font, fill=fg)

    if rng.random() < 0.4:
        img = img.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR,
                         expand=True, fillcolor=base or (128, 128, 128))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if rng.random() < 0.15:                         # extra softening
        w2 = max(8, int(img.width * rng.uniform(0.5, 0.85)))
        h2 = max(8, int(img.height * rng.uniform(0.5, 0.85)))
        img = img.resize((w2, h2), Image.BILINEAR).resize(
            img.size, Image.BILINEAR)

    # Match the real resolution distribution. Measured on VinText crops the
    # height spans p10=13, p50=29, p90=108 px, while a fixed font-size range
    # produces a narrow 25-50 px band: the model would never meet the tiny,
    # blurred crops that dominate the hard cases. Resampling to a height drawn
    # from the real empirical distribution fixes scale and sharpness at once
    # (width-per-character relative to height already matches, 0.51 vs 0.52).
    heights = _G.get("heights")
    if heights:
        target = rng.choice(heights)
        scale = target / max(1, img.height)
        img = img.resize((max(8, int(img.width * scale)), max(8, target)),
                         Image.LANCZOS if scale < 1 else Image.BILINEAR)
    if rng.random() < 0.25:
        a = np.asarray(img, np.int16) + np.random.randint(
            -12, 12, (img.height, img.width, 3), dtype=np.int16)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


# --------------------------------------------------------------- workers
def init_worker(fonts, bg_paths, corpus, buckets, balance, seed, n_bg_cache,
                heights=None):
    _G["fonts"] = fonts
    _G["heights"] = heights
    # Decode background images once per worker instead of per sample —
    # re-opening a JPEG for every image made generation ~4x slower.
    # They MUST be downscaled first: VinText holds images up to 16 megapixel
    # (~48 MB decoded), so caching them raw exhausts RAM and thrashes swap.
    # Only ~100x50 windows are ever cropped, so a 900 px cap loses nothing.
    bg = []
    if bg_paths:
        random.seed(seed + os.getpid())
        for p in random.sample(bg_paths, min(n_bg_cache, len(bg_paths))):
            try:
                im = Image.open(p)
                im.draft("RGB", (900, 900))       # cheap JPEG-level downscale
                im = im.convert("RGB")
                if max(im.size) > 900:
                    sc = 900 / max(im.size)
                    im = im.resize((max(1, int(im.width * sc)),
                                    max(1, int(im.height * sc))),
                                   Image.BILINEAR)
                im.load()
                bg.append(im)
            except Exception:              # noqa: BLE001
                pass
    _G["bg"] = bg
    _G["corpus"] = corpus
    _G["buckets"] = buckets
    _G["balance"] = balance
    _G["keys"] = sorted(buckets) if buckets else []
    random.seed(seed + os.getpid())
    np.random.seed((seed + os.getpid()) % (2 ** 32))


def pick_text(rng):
    """Mix tone-balanced and natural sampling.

    Pure tone-balancing distorts more than the tone histogram: it also
    starves Latin-only strings, because every brand name, abbreviation and
    phone number sits in the level-tone bucket. Real VinText labels are
    38.7% diacritic-free, and sampling buckets uniformly drops that to ~12%.
    Drawing only `balance` of the samples from buckets keeps the rare-tone
    boost while leaving the rest of the distribution close to the real data.
    """
    if _G["balance"] and _G["keys"] and rng.random() < _G["balance"]:
        return rng.choice(_G["buckets"][rng.choice(_G["keys"])])
    return rng.choice(_G["corpus"])


def make_sample(job):
    idx, out_dir, quality = job
    rng = random
    for _ in range(4):                              # retry on pathological font
        text = pick_text(rng)
        if rng.random() < 0.25:
            text = rng.choice([text.upper(), text.lower(), text.title()])
        try:
            img = render_one(text, rng)
        except Exception:                            # noqa: BLE001
            img = None
        if img is not None:
            break
    else:
        return None
    rel = f"{idx // SHARD:04d}/synth_{idx:08d}.jpg"
    dst = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, quality=quality)
    return json.dumps({"file": rel, "text": text}, ensure_ascii=False)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=800000)
    ap.add_argument("--fonts", required=True, help="directory of .ttf/.otf")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--bg-dir", default=None,
                    help="optional real images to crop backgrounds from")
    ap.add_argument("--height-ref", default=None,
                    help="a real *_gt.jsonl; rendered images are resampled "
                         "to heights drawn from these crops")
    ap.add_argument("--height-ref-dir", default=None)
    ap.add_argument("--height-samples", type=int, default=3000)
    ap.add_argument("--bg-cache", type=int, default=80,
                    help="background images decoded into RAM per worker")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--tone-balance", action="store_true",
                    help="oversample rare tones (hoi/nga) via per-tone buckets")
    ap.add_argument("--balance-ratio", type=float, default=0.5,
                    help="fraction of samples drawn tone-balanced; the rest "
                         "follow the corpus distribution (1.0 = fully balanced)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fonts = [os.path.join(args.fonts, f) for f in sorted(os.listdir(args.fonts))
             if f.lower().endswith((".ttf", ".otf"))]
    assert fonts, f"no fonts in {args.fonts}"
    rng = random.Random(args.seed)

    words = build_corpus(args.corpus, args.max_len)
    words += synthetic_strings(max(1, len(words) // 5), rng)
    assert words, "empty corpus"
    buckets = tone_buckets(words) if args.tone_balance else {}

    bg = []
    if args.bg_dir and os.path.isdir(args.bg_dir):
        bg = [os.path.join(args.bg_dir, f)
              for f in sorted(os.listdir(args.bg_dir))[:3000]
              if f.lower().endswith((".jpg", ".png"))]

    heights = []
    if args.height_ref and os.path.exists(args.height_ref):
        ref_dir = args.height_ref_dir or os.path.join(
            os.path.dirname(args.height_ref),
            os.path.basename(args.height_ref).replace("_gt.jsonl", ""))
        recs = [json.loads(l) for l in open(args.height_ref, encoding="utf-8")]
        rng.shuffle(recs)
        for r in recs[: args.height_samples]:
            try:
                with Image.open(os.path.join(ref_dir, r["file"])) as im:
                    heights.append(im.height)
            except Exception:                       # noqa: BLE001
                pass
        if heights:
            q = lambda p: sorted(heights)[int(p * len(heights))]
            print(f"height ref: {len(heights)} crops | "
                  f"p10={q(.1)} p50={q(.5)} p90={q(.9)}")

    print(f"fonts {len(fonts)} | corpus {len(words)} words | "
          f"backgrounds {len(bg)} | tone-balance {args.tone_balance}"
          f" (ratio {args.balance_ratio if args.tone_balance else 0.0})")
    if buckets:
        print("  tone buckets: " +
              ", ".join(f"{TONES[t]}={len(v)}" for t, v in sorted(buckets.items())))

    os.makedirs(args.out, exist_ok=True)
    gt_path = os.path.join(args.out, "synth_gt.jsonl")
    jobs = [(i, args.out, args.quality) for i in range(args.count)]
    n_ok = 0
    with open(gt_path, "w", encoding="utf-8") as gt, \
            Pool(args.workers, initializer=init_worker,
                 initargs=(fonts, bg, words, buckets,
                           args.balance_ratio if args.tone_balance else 0.0,
                           args.seed, args.bg_cache, heights)) as pool:
        for i, line in enumerate(pool.imap_unordered(make_sample, jobs, 256)):
            if line:
                gt.write(line + "\n")
                n_ok += 1
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.count}")
    print(f"done: {n_ok} images -> {args.out} ({gt_path})")


if __name__ == "__main__":
    main()
