# -*- coding: utf-8 -*-
"""Evaluate a VietTDR checkpoint on a cropped-word split.

    python eval.py --ckpt runs/tdr_base/best.pth \
        --data data/vintext_words --split test --charset charset_vintext.txt

Reports word accuracy (case-sensitive / insensitive), 1-NED, accuracy on
the diacritic-heavy subset (>2 diacritic characters, mirroring the thesis
protocol) and dumps a tone confusion matrix + per-sample predictions.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from viettdr.vietchar import Charset, TONES
from viettdr.tokenizer import TripleTokenizer, FlatTokenizer
from viettdr.model import VietTDR
from viettdr.dataset import WordDataset, collate
from viettdr.metrics import Scoreboard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--charset", required=True)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--out", default=None, help="json report path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    targs = ck.get("args", {})
    charset = Charset.from_file(args.charset)
    Tok = FlatTokenizer if targs.get("no_decompose") else TripleTokenizer
    tok = Tok(charset, max_len=targs.get("max_len", 25))
    model = VietTDR(tok.num_bases, max_len=tok.max_len,
                    dim=targs.get("dim", 384),
                    enc_depth=targs.get("enc_depth", 8),
                    dec_depth=targs.get("dec_depth", 2),
                    use_tone_prior=not targs.get("no_tone_prior", False))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    ds = WordDataset(os.path.join(args.data, f"{args.split}_gt.jsonl"),
                     os.path.join(args.data, args.split), tok, train=False)
    ld = DataLoader(ds, args.bs, shuffle=False, num_workers=4,
                    collate_fn=collate)

    board = Scoreboard()
    # every prediction is kept, not just the errors: subset metrics can then
    # be recomputed offline without re-running the model
    records = []
    with torch.no_grad():
        for imgs, *_rest, texts in ld:
            b, m, t = model.greedy_decode(imgs.to(device), tok)
            for i, gt in enumerate(texts):
                pred = tok.decode(b[i], m[i], t[i])
                board.add(pred, gt)
                records.append({"gt": gt, "pred": pred})

    s = board.summary()
    print(f"split={args.split}  n={s['n']}")
    print(f"word acc          : {s['word_acc']:.4f}")
    print(f"word acc (ci)     : {s['word_acc_ci']:.4f}")
    print(f"1 - NED           : {s['one_minus_ned']:.4f}")
    for name, label in [("any_diac", "co dau bat ky"),
                        ("multi_diac", ">=2 ky tu co dau"),
                        ("confusable", "chua hoi/nga"),
                        ("stacked", "co ky tu chong dau")]:
        print(f"  {label:<18}: {s[name + '_acc']:.4f}  (n={s[name + '_n']})")
    print("tone confusion (rows=gt, cols=pred):")
    print("        " + "  ".join(f"{t:>6}" for t in TONES))
    for gi, row in enumerate(s["tone_confusion"]):
        print(f"{TONES[gi]:>7} " + "  ".join(f"{v:6d}" for v in row))

    out = args.out or os.path.join(
        os.path.dirname(args.ckpt), f"eval_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": s, "errors": records}, f,
                  ensure_ascii=False, indent=1)
    print(f"report -> {out}  ({len(records)} errors listed)")


if __name__ == "__main__":
    main()
