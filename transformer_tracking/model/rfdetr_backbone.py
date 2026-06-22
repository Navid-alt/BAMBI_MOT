"""Frozen RF-DETR feature extractor for the tracker.

Loads the project's trained RF-DETR-L checkpoint, freezes it, and exposes the
per-query decoder embeddings (``hs[-1]``, captured via a forward hook on the
transformer) alongside the usual boxes / class logits. These embeddings are the
object descriptors the trainable association module links across frames.

Preprocessing mirrors rfdetr.detr.predict exactly: to_tensor -> square resize to
``resolution`` -> ImageNet normalize. Because the resize is square, RF-DETR's
normalized cxcywh ``pred_boxes`` are directly comparable to GT boxes normalized
by the original (w, h).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "detection_models" / "rf-detr-l" / "output" / "checkpoint_best_ema.pth"

_MEANS = [0.485, 0.456, 0.406]
_STDS = [0.229, 0.224, 0.225]


def _find_lwdetr(obj) -> nn.Module:
    cur = obj
    for _ in range(4):
        if isinstance(cur, nn.Module) and hasattr(cur, "transformer"):
            return cur
        cur = getattr(cur, "model", None) or getattr(cur, "module", None)
        if cur is None:
            break
    raise RuntimeError("Could not locate LWDETR nn.Module on RF-DETR wrapper")


class FrozenRFDETR(nn.Module):
    """Frozen RF-DETR; returns per-query embeddings + boxes + logits per frame."""

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str = "cuda",
                 resolution: int | None = None) -> None:
        super().__init__()
        from rfdetr import RFDETRLarge

        rf = RFDETRLarge(pretrain_weights=str(ckpt))
        self.net = _find_lwdetr(rf.model)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.device = device
        self.resolution = resolution or getattr(rf.model_config, "resolution", 704)
        self.embed_dim = int(self.net.transformer.d_model) if hasattr(self.net.transformer, "d_model") else 256
        self.num_queries = int(getattr(self.net, "num_queries", 300))

        self._hs: torch.Tensor | None = None
        self.net.transformer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        self._hs = output[0] if isinstance(output, (tuple, list)) else output

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img.convert("RGB"))
        t = TF.resize(t, [self.resolution, self.resolution], antialias=True)
        t = TF.normalize(t, _MEANS, _STDS)
        return t

    @torch.no_grad()
    def extract(self, imgs: list[Image.Image]) -> list[dict]:
        """Run a batch of frames; return one dict per frame.

        Each dict: embeds [Q, D], boxes [Q, 4] norm cxcywh, logits [Q, C],
        scores [Q], labels [Q].
        """
        batch = torch.stack([self.preprocess(im) for im in imgs]).to(self.device)
        out = self.net(batch)
        hs = self._hs[-1]                     # [B, Q, D] last decoder layer
        logits = out["pred_logits"]           # [B, Q, C]
        boxes = out["pred_boxes"]             # [B, Q, 4] norm cxcywh
        probs = logits.sigmoid()
        scores, labels = probs.max(dim=-1)

        results = []
        for b in range(batch.shape[0]):
            results.append(dict(
                embeds=hs[b].detach(),
                boxes=boxes[b].detach(),
                logits=logits[b].detach(),
                scores=scores[b].detach(),
                labels=labels[b].detach(),
            ))
        return results
