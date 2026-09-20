# -*- coding: utf-8 -*-
"""Evaluation metrics for Vietnamese scene-text recognition."""
import unicodedata

from .vietchar import (decompose_char, MOD_NONE, TONE_LEVEL,
                       TONE_HOOK, TONE_TILDE)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ned(pred, gt):
    """Normalised edit distance in [0, 1] (0 = identical)."""
    if not pred and not gt:
        return 0.0
    return levenshtein(pred, gt) / max(len(pred), len(gt))


def n_diacritic_chars(text):
    """Number of characters carrying a modifier and/or a tone mark."""
    n = 0
    for ch in unicodedata.normalize("NFC", text):
        _, m, t = decompose_char(ch)
        if m != MOD_NONE or t != TONE_LEVEL:
            n += 1
    return n


def has_confusable_tone(text):
    """Contains hỏi or ngã — the pair this method is built to separate."""
    return any(decompose_char(c)[2] in (TONE_HOOK, TONE_TILDE)
               for c in unicodedata.normalize("NFC", text))


def has_stacked_mark(text):
    """Contains a character carrying BOTH a vowel modifier and a tone
    (ế ệ ữ ằ ...) — the hardest visual case for centreline-style models."""
    return any(m != MOD_NONE and t != TONE_LEVEL
               for _, m, t in
               (decompose_char(c) for c in unicodedata.normalize("NFC", text)))


class Scoreboard:
    """Accumulates word accuracy (case-sensitive & insensitive), 1-NED,
    per-subset accuracies and a tone confusion matrix.

    Subset choice matters for Vietnamese. 99% of VinText labels are single
    syllables, so "more than 2 diacritic-carrying characters" — the obvious
    definition — selects only 0.6% of the test set (62 of 9789) and the
    resulting accuracy is quantised into steps of 1/62. These subsets are
    sized to be informative instead:

        any_diac      any diacritic at all            61.5% of test
        multi_diac    >= 2 diacritic characters        7.9%
        confusable    contains hỏi or ngã              9.2%
        stacked       char with modifier AND tone     23.7%   (ế ệ ữ ằ)
    """

    SUBSETS = {
        "any_diac": lambda t: n_diacritic_chars(t) >= 1,
        "multi_diac": lambda t: n_diacritic_chars(t) >= 2,
        "confusable": has_confusable_tone,
        "stacked": has_stacked_mark,
    }

    def __init__(self):
        self.n = 0
        self.correct = 0
        self.correct_ci = 0
        self.sum_ned = 0.0
        self.sub_n = {k: 0 for k in self.SUBSETS}
        self.sub_ok = {k: 0 for k in self.SUBSETS}
        # tone confusion over aligned equal-length pairs: gt_tone x pred_tone
        self.tone_conf = [[0] * 6 for _ in range(6)]

    def add(self, pred, gt):
        pred = unicodedata.normalize("NFC", pred)
        gt = unicodedata.normalize("NFC", gt)
        hit = int(pred == gt)
        self.n += 1
        self.correct += hit
        self.correct_ci += int(pred.lower() == gt.lower())
        self.sum_ned += 1.0 - ned(pred, gt)
        for name, belongs in self.SUBSETS.items():
            if belongs(gt):
                self.sub_n[name] += 1
                self.sub_ok[name] += hit
        if len(pred) == len(gt):
            for pc, gc in zip(pred, gt):
                pb, pm, pt = decompose_char(pc)
                gb, gm, gt_ = decompose_char(gc)
                if pb.lower() == gb.lower():
                    self.tone_conf[gt_][pt] += 1

    def summary(self):
        z = max(self.n, 1)
        out = {
            "n": self.n,
            "word_acc": self.correct / z,
            "word_acc_ci": self.correct_ci / z,
            "one_minus_ned": self.sum_ned / z,
            "tone_confusion": self.tone_conf,
        }
        for name in self.SUBSETS:
            out[f"{name}_n"] = self.sub_n[name]
            out[f"{name}_acc"] = self.sub_ok[name] / max(self.sub_n[name], 1)
        # kept under the old key so the training log keeps one headline number
        out["heavy_n"] = self.sub_n["multi_diac"]
        out["heavy_acc"] = out["multi_diac_acc"]
        return out
