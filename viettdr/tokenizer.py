# -*- coding: utf-8 -*-
"""Tokenizer bridging text and triple-id tensors for the AR decoder."""
import torch

from .vietchar import Charset, NUM_MODIFIERS, NUM_TONES


class TripleTokenizer:
    """Base-head vocabulary = charset bases + [EOS], [BOS], [PAD].

    Special tokens live on the base head only; their modifier/tone targets
    are always (none, level).
    """

    def __init__(self, charset: Charset, max_len=25):
        self.charset = charset
        self.max_len = max_len
        nb = charset.num_bases
        self.eos_id, self.bos_id, self.pad_id = nb, nb + 1, nb + 2
        self.num_bases = nb + 3

        vm = torch.zeros(self.num_bases, NUM_MODIFIERS, NUM_TONES,
                         dtype=torch.bool)
        for b in range(nb):
            for m in range(NUM_MODIFIERS):
                for t in range(NUM_TONES):
                    vm[b, m, t] = charset.valid[b][m][t]
        vm[self.eos_id, 0, 0] = True          # EOS is decodable
        self.valid_mask = vm

    def can_encode(self, text):
        return (len(text) <= self.max_len
                and None not in self.charset.encode(text))

    def encode(self, text):
        """Return (input_ids, target_ids), each a (b, m, t) tuple of
        LongTensors of length max_len + 1 (teacher forcing layout)."""
        triples = [tr for tr in self.charset.encode(text) if tr is not None]
        triples = triples[: self.max_len]
        L = self.max_len + 1

        def pack(seq, fill):
            out = list(seq) + [fill] * (L - len(seq))
            return out[:L]

        bos, eos, pad = ((self.bos_id, 0, 0), (self.eos_id, 0, 0),
                         (self.pad_id, 0, 0))
        inp = pack([bos] + triples, pad)
        tgt = pack(triples + [eos], pad)
        to_t = lambda seq: tuple(torch.tensor([s[i] for s in seq],
                                              dtype=torch.long)
                                 for i in range(3))
        return to_t(inp), to_t(tgt)

    def decode(self, b_ids, m_ids, t_ids):
        """Id tensors (single sequence) -> text, stopping at EOS."""
        triples = []
        for b, m, t in zip(b_ids.tolist(), m_ids.tolist(), t_ids.tolist()):
            if b == self.eos_id or b == self.pad_id:
                break
            if b >= self.charset.num_bases:
                continue
            triples.append((b, m, t))
        return self.charset.decode(triples)


class FlatTokenizer:
    """Ablation baseline: NO decomposition — one atomic class per combined
    character (the conventional 229-way vocabulary).

    It deliberately exposes the exact same interface as TripleTokenizer so
    the model, dataset, training loop and metrics are byte-for-byte shared.
    The trick: the combined character id is carried on the *base* channel
    while the modifier/tone channels are pinned to 0. Training then sets
    lam_m = lam_t = lam_c = 0 so those degenerate heads contribute nothing.
    This isolates the effect of decomposition alone, with identical
    backbone, optimizer and data.
    """

    def __init__(self, charset: Charset, max_len=25):
        self.charset = charset
        self.max_len = max_len
        self.chars = list(charset.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        n = len(self.chars)
        self.eos_id, self.bos_id, self.pad_id = n, n + 1, n + 2
        self.num_bases = n + 3            # = flat vocabulary size

        vm = torch.zeros(self.num_bases, NUM_MODIFIERS, NUM_TONES,
                         dtype=torch.bool)
        vm[:, 0, 0] = True                # only (char, none, level) decodable
        self.valid_mask = vm

    def can_encode(self, text):
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        return len(text) <= self.max_len and all(c in self.stoi for c in text)

    def encode(self, text):
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        ids = [self.stoi[c] for c in text if c in self.stoi][: self.max_len]
        L = self.max_len + 1
        inp = ([self.bos_id] + ids + [self.pad_id] * L)[:L]
        tgt = (ids + [self.eos_id] + [self.pad_id] * L)[:L]
        zeros = torch.zeros(L, dtype=torch.long)
        to_t = lambda seq: (torch.tensor(seq, dtype=torch.long),
                            zeros.clone(), zeros.clone())
        return to_t(inp), to_t(tgt)

    def decode(self, b_ids, m_ids=None, t_ids=None):
        out = []
        for b in b_ids.tolist():
            if b == self.eos_id or b == self.pad_id:
                break
            if b < len(self.chars):
                out.append(self.chars[b])
        return "".join(out)
