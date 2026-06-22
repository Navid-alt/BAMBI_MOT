"""Learned cross-frame association on top of frozen RF-DETR embeddings.

Given the per-query embeddings + boxes of two frames, this module contextualises
them with a small attentional GNN (alternating self/cross attention, SuperGlue
style) and produces a soft assignment matrix with learnable "dustbin" rows/cols
that absorb track births and deaths. Trained with the negative log-likelihood of
the ground-truth correspondences derived from ``track_id``.

Only this module is trained; RF-DETR stays frozen, so it fits a 4070 easily.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class MLP(nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionLayer(nn.Module):
    """One multi-head attention block (self or cross) with a residual FFN."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP([dim, dim * 2, dim])

    def forward(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm1(x), self.norm1(src), self.norm1(src))
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


def log_optimal_transport(scores: torch.Tensor, bin_score: torch.Tensor, iters: int) -> torch.Tensor:
    """Sinkhorn in log-space with a learnable dustbin (SuperGlue)."""
    m, n = scores.shape
    one = scores.new_tensor(1.0)
    ms, ns = (m * one), (n * one)

    bins0 = bin_score.expand(m, 1)
    bins1 = bin_score.expand(1, n)
    alpha = bin_score.expand(1, 1)

    couplings = torch.cat([
        torch.cat([scores, bins0], dim=1),
        torch.cat([bins1, alpha], dim=1),
    ], dim=0)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(1), dim=0)
    return couplings + u.unsqueeze(1) + v.unsqueeze(0) - norm


class TrackAssociator(nn.Module):
    """Soft-assigns detections between two frames using embeddings + geometry."""

    def __init__(self, in_dim: int = 256, dim: int = 256, heads: int = 8,
                 layers: int = 4, sinkhorn_iters: int = 50) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim)
        self.pos_encoder = MLP([4, dim, dim])   # encodes cxcywh geometry
        # alternating self / cross attention layers
        self.self_layers = nn.ModuleList([AttentionLayer(dim, heads) for _ in range(layers)])
        self.cross_layers = nn.ModuleList([AttentionLayer(dim, heads) for _ in range(layers)])
        self.final = nn.Linear(dim, dim)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self.sinkhorn_iters = sinkhorn_iters
        self.dim = dim

    def _encode(self, embeds: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return self.input_proj(embeds) + self.pos_encoder(boxes)

    def forward(self, embeds_a, boxes_a, embeds_b, boxes_b) -> torch.Tensor:
        """Returns log-assignment matrix [Na+1, Nb+1] (last row/col = dustbin)."""
        xa = self._encode(embeds_a, boxes_a).unsqueeze(0)  # [1, Na, D]
        xb = self._encode(embeds_b, boxes_b).unsqueeze(0)
        for self_l, cross_l in zip(self.self_layers, self.cross_layers):
            xa = self_l(xa, xa)
            xb = self_l(xb, xb)
            xa2 = cross_l(xa, xb)
            xb2 = cross_l(xb, xa)
            xa, xb = xa2, xb2
        fa = self.final(xa).squeeze(0)  # [Na, D]
        fb = self.final(xb).squeeze(0)  # [Nb, D]
        scores = fa @ fb.t() / self.dim ** 0.5
        return log_optimal_transport(scores, self.bin_score, self.sinkhorn_iters)


def assignment_nll_loss(log_assign: torch.Tensor, matches: torch.Tensor,
                        unmatched_a: torch.Tensor, unmatched_b: torch.Tensor) -> torch.Tensor:
    """NLL of GT correspondences (SuperGlue loss).

    matches: [K, 2] index pairs (a_idx, b_idx) that share a track_id.
    unmatched_a/_b: indices that have no partner (assigned to the dustbin).
    """
    losses = []
    if matches.numel():
        losses.append(-log_assign[matches[:, 0], matches[:, 1]])
    if unmatched_a.numel():
        losses.append(-log_assign[unmatched_a, -1])
    if unmatched_b.numel():
        losses.append(-log_assign[-1, unmatched_b])
    if not losses:
        return log_assign.new_zeros(())
    return torch.cat(losses).mean()
