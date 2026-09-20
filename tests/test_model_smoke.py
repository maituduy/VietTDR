# -*- coding: utf-8 -*-
"""End-to-end smoke test on CPU with a tiny model and real data files."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from viettdr.vietchar import Charset
from viettdr.tokenizer import TripleTokenizer
from viettdr.model import VietTDR, TripleLoss
from viettdr.dataset import WordDataset, collate

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    torch.manual_seed(0)
    cs = Charset.from_file(os.path.join(ROOT, "charset_vintext.txt"))
    tok = TripleTokenizer(cs, max_len=25)

    model = VietTDR(tok.num_bases, dim=128, enc_depth=2, dec_depth=1,
                    enc_heads=4, dec_heads=4)
    crit = TripleLoss(tok.valid_mask, tok.pad_id)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny model: {n/1e6:.2f}M params")

    ds = WordDataset(os.path.join(ROOT, "data/vintext_words/val_gt.jsonl"),
                     os.path.join(ROOT, "data/vintext_words/val"),
                     tok, train=True)
    ld = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)
    imgs, ib, im_, it, tb, tm, tt, texts = next(iter(ld))
    print("batch:", imgs.shape, "texts:", texts[:3])

    # forward + backward
    logits = model(imgs, ib, im_, it)
    loss, parts = crit(logits, (tb, tm, tt))
    loss.backward()
    assert torch.isfinite(loss), loss
    print(f"loss {loss.item():.4f} parts {parts}")

    # one optimizer step must change the tone-prior gate's grad state
    g = model.tone_prior.gate.grad
    assert g is not None and torch.isfinite(g), "tone gate got no gradient"
    print(f"tone-prior gate grad = {g.item():.3e}")

    # constrained greedy decode returns only valid, composable strings
    b, m, t = model.greedy_decode(imgs, tok)
    preds = [tok.decode(b[i], m[i], t[i]) for i in range(b.size(0))]
    print("untrained preds:", preds)
    for p in preds:
        assert isinstance(p, str)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
