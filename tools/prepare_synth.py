# -*- coding: utf-8 -*-
"""Convert a SynthTIGER output directory into VietTDR jsonl format.

SynthTIGER (https://github.com/clovaai/synthtiger) writes:
    <root>/images/**/*.jpg  and  <root>/gt.txt  (lines: <relpath>\t<label>)

Usage:
    python tools/prepare_synth.py --synth /path/synthtiger_out \
        --out data/synth_words

The images are NOT copied; the jsonl stores paths relative to --synth and
train.py should be given  --synth-jsonl <out>/synth_gt.jsonl
                          --synth-dir  <synth>/images   (or <synth>).
See docs/synthetic_data.md for the full generation recipe.
"""
import argparse
import json
import os
import unicodedata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gt-name", default="gt.txt")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = os.path.join(args.synth, args.gt_name)
    dst = os.path.join(args.out, "synth_gt.jsonl")
    n = 0
    with open(src, encoding="utf-8") as f, \
            open(dst, "w", encoding="utf-8") as g:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            rel, label = line.split("\t", 1)
            label = unicodedata.normalize("NFC", label.strip())
            if not label:
                continue
            g.write(json.dumps({"file": rel, "text": label},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} entries -> {dst}")


if __name__ == "__main__":
    main()
