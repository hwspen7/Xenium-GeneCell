"""Detached, per-center diagnostics for sparse one-hop spatial attention.

Diagnostics use PRE-dropout attention probabilities, not gated message
magnitudes. They describe the learned neighbor selection, not biological
causality or reconstruction performance. No graph tensors are transferred
between devices and no gradients pass through these summaries.
"""

from __future__ import annotations

import torch


def summarize_spatial_attention(
    weights: torch.Tensor,
    edge_index: torch.Tensor,
    edge_distances_um: torch.Tensor,
    center_index: torch.Tensor,
    *,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    """Return degree, per-head entropy, effective neighbors and distance.

    ``edge_index`` uses [receiver local node, sender local node]. The returned
    center axis follows ``center_index`` order. Isolated centers have degree 0
    and zero-valued diagnostics, including effective neighbor count.
    """
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("weights must be floating [E,H].")
    e, heads = weights.shape
    if heads == 0:
        raise ValueError("At least one attention head is required.")
    if (edge_index.shape != (2, e) or edge_index.dtype != torch.long
            or edge_distances_um.shape != (e,)
            or center_index.ndim != 1 or center_index.dtype != torch.long):
        raise ValueError("Graph tensors have incompatible shapes or dtypes.")
    if num_nodes < 1 or not 0 < center_index.numel() <= num_nodes:
        raise ValueError("Invalid graph/center dimensions.")
    if any(t.device != weights.device for t in (edge_index, edge_distances_um, center_index)):
        raise ValueError("Graph tensors must share the weights device.")
    c = center_index.numel()
    with torch.no_grad():
        prob = weights.detach().float()
        mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=weights.device)
        mapping[center_index] = torch.arange(c, device=weights.device)
        receiver = mapping[edge_index[0]]
        degree = torch.bincount(receiver, minlength=c)
        entropy = prob.new_zeros((c, heads))
        attended_distance = prob.new_zeros((c, heads))
        if e:
            terms = torch.where(prob > 0, -prob * prob.clamp_min(1e-30).log(), 0.0)
            entropy.index_add_(0, receiver, terms)
            attended_distance.index_add_(0, receiver, prob * edge_distances_um.float().unsqueeze(1))
        active = degree > 0
        effective = torch.where(active[:, None], entropy.exp(), torch.zeros_like(entropy))
        normalized = torch.zeros_like(entropy)
        eligible = degree > 1
        if eligible.any():
            normalized[eligible] = entropy[eligible] / degree[eligible, None].float().log()
        return {
            "degree": degree,
            "entropy": entropy,
            "normalized_entropy": normalized,
            "effective_neighbors": effective,
            "attended_distance_um": attended_distance,
            "active_center_mask": active,
        }
