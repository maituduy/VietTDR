# -*- coding: utf-8 -*-
"""Lightweight x2 super-resolution network for word crops (16x64 -> 32x128).

Small on purpose: the downstream recognizer runs at 32x128, inputs are tiny,
and the whole point is a cheap preprocessing module that could sit in front
of any recognizer. ~1.2M parameters, SRResNet-style trunk with a global
skip so the identity mapping is easy to learn.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.PReLU(ch)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class VietSR(nn.Module):
    def __init__(self, ch=64, blocks=8, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)],
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(ch))
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x):
        # global residual against bicubic upsampling: the net only has to
        # predict the missing detail, not the whole image
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                             align_corners=False)
        f = self.head(x)
        f = f + self.body(f)
        out = self.tail(self.up(f))
        return (base + out).clamp(0, 1)


def charbonnier(a, b, eps=1e-6):
    return torch.sqrt((a - b) ** 2 + eps).mean()


class SRLoss(nn.Module):
    """Charbonnier + diacritic-mask-weighted Charbonnier.

    Marks cover only ~1-3% of the pixels, so the plain term barely notices
    them; the masked term is normalised by mask area so its gradient scale
    does not depend on how much of the word carries marks.
    """

    def __init__(self, lam_mask=2.0):
        super().__init__()
        self.lam_mask = lam_mask

    def forward(self, sr, hr, mask):
        loss_pix = charbonnier(sr, hr)
        m = mask.expand_as(sr)
        denom = m.sum().clamp_min(1.0)
        loss_mark = (torch.sqrt((sr - hr) ** 2 + 1e-6) * m).sum() / denom
        total = loss_pix + self.lam_mask * loss_mark
        return total, {"pix": loss_pix.item(), "mark": loss_mark.item()}


def psnr(a, b, eps=1e-8):
    """a, b in [0,1], shape B,C,H,W -> mean PSNR over batch."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(eps)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def psnr_masked(a, b, mask, eps=1e-8):
    """PSNR computed only inside the diacritic mask (Diacritic-PSNR).
    Samples without any mask pixels are skipped."""
    m = mask.expand_as(a)
    per = []
    for i in range(a.size(0)):
        n = m[i].sum()
        if n < 1:
            continue
        mse = (((a[i] - b[i]) ** 2) * m[i]).sum() / n
        per.append(10 * torch.log10(1.0 / mse.clamp_min(eps)))
    if not per:
        return float("nan")
    return torch.stack(per).mean().item()
