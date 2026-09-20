# -*- coding: utf-8 -*-
"""Word-image dataset (jsonl produced by tools/crop_vintext_words.py or
tools/prepare_synth.py)."""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def build_transform(img_size=(32, 128), level="full"):
    """level: 'full' (real training images), 'light' or 'none'.

    Synthetic images already have blur, noise, rotation, perspective and
    JPEG artefacts baked in at generation time, so running the full
    torchvision pipeline over them again is redundant — and it is the
    training bottleneck, because these PIL ops are CPU-bound and Kaggle
    gives only 4 cores. 'light' keeps colour jitter alone.
    """
    if level == "full":
        aug = [
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.6),
            T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.2),
            T.RandomApply([T.RandomRotation(4, expand=False,
                                            fill=(127, 127, 127))], p=0.3),
            T.RandomPerspective(0.12, p=0.3, fill=(127, 127, 127)),
        ]
    elif level == "light":
        aug = [T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2, 0.03)], p=0.4)]
    else:
        aug = []
    return T.Compose(aug + [
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])


class WordDataset(Dataset):
    """`levels` gives the augmentation level per source, so synthetic data
    can use the cheap pipeline while real images keep the full one."""

    def __init__(self, jsonl_paths, img_dirs, tokenizer, img_size=(32, 128),
                 train=True, levels=None):
        if isinstance(jsonl_paths, str):
            jsonl_paths, img_dirs = [jsonl_paths], [img_dirs]
        if levels is None:
            levels = ["full" if train else "none"] * len(jsonl_paths)
        self.samples = []
        n_skip = 0
        for src, (jp, di) in enumerate(zip(jsonl_paths, img_dirs)):
            with open(jp, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if tokenizer.can_encode(rec["text"]):
                        self.samples.append(
                            (os.path.join(di, rec["file"]), rec["text"], src))
                    else:
                        n_skip += 1
        self.tokenizer = tokenizer
        self.tfs = [build_transform(img_size, lv) for lv in levels]
        if n_skip:
            print(f"[dataset] {len(self.samples)} samples "
                  f"({n_skip} skipped: OOV or > max_len) | aug {levels}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text, src = self.samples[idx]
        img = self.tfs[src](Image.open(path).convert("RGB"))
        (ib, im_, it), (tb, tm, tt) = self.tokenizer.encode(text)
        return img, ib, im_, it, tb, tm, tt, text


def collate(batch):
    imgs, ib, im_, it, tb, tm, tt, texts = zip(*batch)
    st = lambda xs: torch.stack(xs)
    return (st(imgs), st(ib), st(im_), st(it),
            st(tb), st(tm), st(tt), list(texts))
