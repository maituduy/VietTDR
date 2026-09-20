# -*- coding: utf-8 -*-
"""VietTDR: Vietnamese scene-text recognizer with triple-decomposition decoding.

Architecture
    ViT encoder (32x128 image -> 8x16 tokens)
      -> autoregressive Transformer decoder
      -> three parallel heads: base letter / vowel modifier / tone
    + tone-prior cross-attention: a light attention branch over encoder
      tokens whose logits carry a learnable vertical-position bias,
      initialised to favour the top/bottom bands where Vietnamese tone
      marks live.
    + composition-validity loss: only 229 of the 95x5x6 = 2850 triples are
      linguistically valid; the joint probability mass on invalid triples
      is penalised, and invalid triples are masked out at inference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vietchar import NUM_MODIFIERS, NUM_TONES


# --------------------------------------------------------------- encoder
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(32, 128), patch=(4, 8), dim=384):
        super().__init__()
        self.grid = (img_size[0] // patch[0], img_size[1] // patch[1])
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                       # B, C, gh, gw
        return x.flatten(2).transpose(1, 2)    # B, gh*gw, C


class ViTEncoder(nn.Module):
    def __init__(self, img_size=(32, 128), patch=(4, 8), dim=384,
                 depth=8, heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, dim)
        self.grid = self.patch_embed.grid
        n = self.grid[0] * self.grid[1]
        self.pos_emb = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, int(dim * mlp_ratio), dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_emb
        return self.norm(self.blocks(x))       # B, N, C


# ---------------------------------------------------- tone prior attention
class TonePriorAttention(nn.Module):
    """Single-head cross-attention with a learnable vertical-position bias.

    Tone marks sit above (sắc/huyền/hỏi/ngã) or below (nặng) the letter
    body, so encoder rows near the top/bottom of the crop are a priori more
    informative for the tone head. The bias is initialised bimodally and
    remains learnable.
    """

    def __init__(self, dim, grid):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        gh, gw = grid
        prior = torch.zeros(gh)
        prior[: max(1, gh // 4)] = 1.0          # top band: above-marks
        prior[-max(1, gh // 4):] = 0.5          # bottom band: dot-below
        prior[gh // 3: 2 * gh // 3] = -0.5      # letter body
        self.row_bias = nn.Parameter(prior.repeat_interleave(gw))  # N
        self.gate = nn.Parameter(torch.tensor(-2.0))  # progressive gate

    def forward(self, h, mem):
        # h: B,T,C decoder states; mem: B,N,C encoder tokens
        att = torch.einsum("btc,bnc->btn", self.q(h), self.k(mem)) * self.scale
        att = att + self.row_bias.view(1, 1, -1)
        z = torch.einsum("btn,bnc->btc", att.softmax(-1), self.v(mem))
        return h + torch.sigmoid(self.gate) * self.out(z)


# --------------------------------------------------------------- decoder
class TripleEmbedding(nn.Module):
    """Embed a character as the sum of its component embeddings."""

    def __init__(self, num_bases, dim):
        super().__init__()
        self.base = nn.Embedding(num_bases, dim)
        self.mod = nn.Embedding(NUM_MODIFIERS, dim)
        self.tone = nn.Embedding(NUM_TONES, dim)

    def forward(self, b, m, t):
        return self.base(b) + self.mod(m) + self.tone(t)


class VietTDR(nn.Module):
    """num_bases includes the specials appended by the tokenizer
    (EOS/BOS/PAD are base-head classes with modifier=none, tone=level)."""

    def __init__(self, num_bases, max_len=25, dim=384, enc_depth=8,
                 enc_heads=6, dec_depth=2, dec_heads=6, drop=0.1,
                 img_size=(32, 128), patch=(4, 8), use_tone_prior=True):
        super().__init__()
        self.max_len = max_len
        self.encoder = ViTEncoder(img_size, patch, dim, enc_depth,
                                  enc_heads, drop=drop)
        self.embed = TripleEmbedding(num_bases, dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            dim, dec_heads, int(dim * 4), dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_depth)
        self.norm = nn.LayerNorm(dim)

        self.tone_prior = (TonePriorAttention(dim, self.encoder.grid)
                           if use_tone_prior else None)
        self.head_base = nn.Linear(dim, num_bases)
        self.head_mod = nn.Linear(dim, NUM_MODIFIERS)
        self.head_tone = nn.Linear(dim, NUM_TONES)

    def forward(self, images, tgt_b, tgt_m, tgt_t):
        """Teacher-forced training pass.
        tgt_*: B,T token ids starting with BOS (inputs, not outputs)."""
        mem = self.encoder(images)
        T = tgt_b.size(1)
        h = self.embed(tgt_b, tgt_m, tgt_t) + self.dec_pos[:, :T]
        mask = torch.triu(torch.full((T, T), float("-inf"),
                                     device=h.device), diagonal=1)
        h = self.norm(self.decoder(h, mem, tgt_mask=mask))
        h_tone = self.tone_prior(h, mem) if self.tone_prior else h
        return (self.head_base(h), self.head_mod(h),
                self.head_tone(h_tone))

    @torch.no_grad()
    def greedy_decode(self, images, tokenizer):
        """Joint constrained greedy decoding.

        At each step the (base, mod, tone) triple with the highest joint
        log-probability among *linguistically valid* triples is selected.
        """
        self.eval()
        device = images.device
        B = images.size(0)
        mem = self.encoder(images)
        vmask = tokenizer.valid_mask.to(device)          # nb,nm,nt bool
        neg = torch.finfo(torch.float32).min

        b = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        m = torch.zeros_like(b)
        t = torch.zeros_like(b)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(self.max_len + 1):
            h = self.embed(b, m, t) + self.dec_pos[:, : b.size(1)]
            mask = torch.triu(torch.full((b.size(1), b.size(1)), float("-inf"),
                                         device=device), diagonal=1)
            h = self.norm(self.decoder(h, mem, tgt_mask=mask))
            h_tone = self.tone_prior(h, mem) if self.tone_prior else h
            lb = self.head_base(h[:, -1]).log_softmax(-1)     # B,nb
            lm = self.head_mod(h[:, -1]).log_softmax(-1)      # B,nm
            lt = self.head_tone(h_tone[:, -1]).log_softmax(-1)  # B,nt
            joint = (lb[:, :, None, None] + lm[:, None, :, None]
                     + lt[:, None, None, :])
            joint = joint.masked_fill(~vmask[None], neg)
            flat = joint.flatten(1).argmax(-1)
            nb_, nm_, nt_ = vmask.shape
            bi = flat // (nm_ * nt_)
            mi = (flat // nt_) % nm_
            ti = flat % nt_
            bi = torch.where(done, torch.full_like(bi, tokenizer.pad_id), bi)
            mi = torch.where(done, torch.zeros_like(mi), mi)
            ti = torch.where(done, torch.zeros_like(ti), ti)
            b = torch.cat([b, bi[:, None]], 1)
            m = torch.cat([m, mi[:, None]], 1)
            t = torch.cat([t, ti[:, None]], 1)
            done = done | (bi == tokenizer.eos_id)
            if bool(done.all()):
                break
        return b[:, 1:], m[:, 1:], t[:, 1:]


# ----------------------------------------------------------------- loss
class TripleLoss(nn.Module):
    """CE on the three heads + composition-validity penalty.

    L = CE_b + lam_m*CE_m + lam_t*CE_t + lam_c * L_comp
    L_comp = -log sum_{valid (b,m,t)} p_b(b) p_m(m) p_t(t)
    """

    def __init__(self, valid_mask, pad_id, lam_m=1.0, lam_t=1.0,
                 lam_c=0.2, smoothing=0.1):
        super().__init__()
        self.register_buffer("vmask", valid_mask.float())
        self.pad_id = pad_id
        self.lam_m, self.lam_t, self.lam_c = lam_m, lam_t, lam_c
        self.smoothing = smoothing

    def forward(self, logits, targets):
        (lb, lm, lt) = logits          # B,T,*
        (tb, tm, tt) = targets         # B,T
        keep = tb != self.pad_id
        ce = lambda lg, tg, nc: F.cross_entropy(
            lg[keep], tg[keep], label_smoothing=self.smoothing)
        loss_b = ce(lb, tb, lb.size(-1))
        loss_m = ce(lm, tm, lm.size(-1))
        loss_t = ce(lt, tt, lt.size(-1))

        pb = lb[keep].softmax(-1)
        pm = lm[keep].softmax(-1)
        pt = lt[keep].softmax(-1)
        p_valid = torch.einsum("nb,nm,nt,bmt->n", pb, pm, pt, self.vmask)
        loss_c = -torch.log(p_valid.clamp_min(1e-6)).mean()

        total = (loss_b + self.lam_m * loss_m + self.lam_t * loss_t
                 + self.lam_c * loss_c)
        return total, {"base": loss_b.item(), "mod": loss_m.item(),
                       "tone": loss_t.item(), "comp": loss_c.item()}
