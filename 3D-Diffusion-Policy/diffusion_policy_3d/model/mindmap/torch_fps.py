"""Pure-torch replacement for dgl.geometry.farthest_point_sampler (the only dgl
symbol the mindmap encoder uses), so training needs no DGL wheel."""

import torch


def farthest_point_sampler(pos: torch.Tensor, npoints: int, start_idx=None) -> torch.Tensor:
    """pos: (B, N, C). Returns (B, npoints) index tensor, dgl-compatible.
    Iterative max-min selection; with start_idx given the result is deterministic."""
    B, N, _ = pos.shape
    npoints = min(npoints, N)
    idxs = torch.zeros(B, npoints, dtype=torch.long, device=pos.device)
    if start_idx is None:
        cur = torch.randint(0, N, (B,), device=pos.device)
    else:
        cur = torch.full((B,), int(start_idx), dtype=torch.long, device=pos.device)
    min_d = torch.full((B, N), float("inf"), device=pos.device)
    bidx = torch.arange(B, device=pos.device)
    for i in range(npoints):
        idxs[:, i] = cur
        d = (pos - pos[bidx, cur].unsqueeze(1)).pow(2).sum(-1)
        min_d = torch.minimum(min_d, d)
        cur = min_d.argmax(-1)
    return idxs
