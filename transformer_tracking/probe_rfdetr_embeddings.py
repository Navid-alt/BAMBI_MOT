"""Milestone 1 feasibility probe.

Loads the trained RF-DETR-L checkpoint and confirms that the decoder's
per-query hidden states (``hs``) — the embeddings we want to use as track
queries — are reachable via a forward hook on the transformer, alongside the
usual ``pred_logits`` / ``pred_boxes`` outputs.

Run from the detection_models venv:
    cd detection_models && uv run python ../transformer_tracking/probe_rfdetr_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "detection_models" / "rf-detr-l" / "output" / "checkpoint_best_ema.pth"


def _find_nn_module(obj, depth=0):
    """Walk a couple of wrapper levels to find the underlying LWDETR nn.Module."""
    import torch.nn as nn
    seen = []
    cur = obj
    for attr in ("model", "model", "module"):
        if isinstance(cur, nn.Module) and hasattr(cur, "transformer"):
            return cur
        seen.append(type(cur).__name__)
        cur = getattr(cur, attr, None)
        if cur is None:
            break
    if isinstance(cur, nn.Module) and hasattr(cur, "transformer"):
        return cur
    raise RuntimeError(f"Could not locate LWDETR module; walked: {seen}")


def main() -> None:
    if not CKPT.exists():
        sys.exit(f"checkpoint not found: {CKPT}")
    print(f"[probe] loading {CKPT}")

    from rfdetr import RFDETRLarge

    rf = RFDETRLarge(pretrain_weights=str(CKPT))
    net = _find_nn_module(rf.model)
    print(f"[probe] found nn.Module: {type(net).__name__}")
    print(f"[probe] num_queries = {getattr(net, 'num_queries', '?')}")

    captured = {}

    def hook(module, inputs, output):
        # transformer returns a tuple; output[0] == hs of shape [L, B, Q, D]
        hs = output[0] if isinstance(output, (tuple, list)) else output
        captured["hs"] = hs

    h = net.transformer.register_forward_hook(hook)

    net.eval()
    device = next(net.parameters()).device
    res = getattr(rf.model_config, "resolution", 704)
    print(f"[probe] device={device} resolution={res}")
    dummy = torch.randn(1, 3, res, res, device=device)

    with torch.no_grad():
        out = net(dummy)
    h.remove()

    print("\n=== RESULT ===")
    print("output keys     :", list(out.keys()))
    print("pred_logits     :", tuple(out["pred_logits"].shape))
    print("pred_boxes      :", tuple(out["pred_boxes"].shape))
    if "hs" in captured:
        hs = captured["hs"]
        print("captured hs     :", tuple(hs.shape), hs.dtype, "<-- track-query embeddings")
        print("hs[-1] (last)   :", tuple(hs[-1].shape), "= [B, num_queries, dim]")
        print("\nFEASIBLE: per-query embeddings ARE hook-accessible.")
    else:
        print("\nNOT captured via this hook — fall back to external-detector design.")


if __name__ == "__main__":
    main()
