# -*- coding: utf-8 -*-
"""Class-distribution statistics before/after triple decomposition.

Produces the long-tail evidence for the paper: how sparse the 229-class
combined vocabulary is on VinText vs. the per-component distributions.

    python tools/class_stats.py --gt data/vintext_words/train_gt.jsonl \
        --charset charset_vintext.txt --out docs/class_stats.json
"""
import argparse
import json
import sys, os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from viettdr.vietchar import (Charset, decompose_char, MODIFIERS, TONES,
                              MOD_NONE, TONE_LEVEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--charset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cs = Charset.from_file(args.charset)
    full, base, mod, tone = Counter(), Counter(), Counter(), Counter()
    n_words = n_diac_words = 0
    with open(args.gt, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            n_words += 1
            has_diac = False
            for ch in text:
                b, m, t = decompose_char(ch)
                full[ch] += 1
                base[b] += 1
                mod[m] += 1
                tone[t] += 1
                if m != MOD_NONE or t != TONE_LEVEL:
                    has_diac = True
            n_diac_words += has_diac

    n_chars = sum(full.values())
    in_vocab = [c for c in cs.chars if c in full]
    missing = [c for c in cs.chars if c not in full]
    counts = sorted(full.values())

    def tail_stats(cnt, k):
        vals = sorted(cnt.values())
        return {"classes_seen": len(cnt), "min": vals[0],
                "median": vals[len(vals) // 2], "max": vals[-1],
                f"classes_with_lt_{k}": sum(v < k for v in vals)}

    report = {
        "n_words": n_words,
        "n_words_with_diacritics": n_diac_words,
        "n_chars": n_chars,
        "combined_vocab": {
            **tail_stats(full, 50),
            "vocab_size": len(cs.chars),
            "classes_never_seen": len(missing),
            "missing_examples": missing[:20],
        },
        "base": tail_stats(base, 50),
        "modifier": {MODIFIERS[m]: mod.get(m, 0) for m in range(5)},
        "tone": {TONES[t]: tone.get(t, 0) for t in range(6)},
        "rarest_combined_classes": [
            {"char": c, "count": n}
            for c, n in sorted(full.items(), key=lambda kv: kv[1])[:25]],
    }
    txt = json.dumps(report, ensure_ascii=False, indent=1)
    print(txt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
