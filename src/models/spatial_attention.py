"""Sparse one-hop distance-aware graph attention for SG-MAE V2.0-B.

This module transforms *already masked* intrinsic cell embeddings. It does
not read raw expression, gene reconstruction targets or cell-type labels.
The receiver of each edge is a center node; the sender is a neighbor.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DistanceAwareSpatialAttention(nn.Module):
    """One-hop multi-head attention with a monotonically decreasing distance bias.

    For edge neighbor j -> center i and attention head h:
        score(i,j,h) = dot(q_i,h, k_j,h) / sqrt(head_dim)
                       - softplus(decay_h) * (distance_ij / scale_um)

    Softmax is performed over the incoming neighbors of each center. Isolated
    centers receive *exactly zero* spatial context, rather than self-messages.
    No dense [N,N] attention matrix is constructed.

    Optional exponential message gating suppresses far-neighbor messages even
    when the center has only one neighbor. The default preserves V2-B behavior.
    """

    def __init__(
        self,
        cell_dim: int = 128,
        num_heads: int = 4,
        max_distance_um: float = 50.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        distance_gate: str = "none",
        max_neighbors: int | None = None,
    ) -> None:
        super().__init__()
        if type(cell_dim) is not int or type(num_heads) is not int or cell_dim < 1 or num_heads < 1:
            raise ValueError("cell_dim and num_heads must be positive integers.")
        if cell_dim % num_heads:
            raise ValueError("cell_dim must be divisible by num_heads.")
        if isinstance(max_distance_um, bool) or not isinstance(max_distance_um, (int, float)):
            raise ValueError("max_distance_um must be positive and finite.")
        if not math.isfinite(max_distance_um) or max_distance_um <= 0:
            raise ValueError("max_distance_um must be positive and finite.")
        for name, value in (("attention_dropout", attention_dropout), ("projection_dropout", projection_dropout)):
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1).")
        if distance_gate not in ("none", "exponential"):
            raise ValueError("distance_gate must be 'none' or 'exponential'.")
        if max_neighbors is not None and (type(max_neighbors) is not int or max_neighbors < 1):
            raise ValueError("max_neighbors must be None or a positive integer.")
        self.distance_gate = distance_gate
        self.max_neighbors = max_neighbors
        self.cell_dim = cell_dim
        self.num_heads = num_heads
        self.head_dim = cell_dim // num_heads
        self.max_distance_um = float(max_distance_um)
        self.attention_dropout = float(attention_dropout)
        self.projection_dropout = nn.Dropout(float(projection_dropout))
        self.input_norm = nn.LayerNorm(cell_dim)
        self.query = nn.Linear(cell_dim, cell_dim, bias=False)
        self.key = nn.Linear(cell_dim, cell_dim, bias=False)
        self.value = nn.Linear(cell_dim, cell_dim, bias=False)
        self.output = nn.Linear(cell_dim, cell_dim, bias=False)
        self.raw_distance_decay = nn.Parameter(
            torch.full((num_heads,), math.log(math.expm1(1.0)))
        )
        if distance_gate == "exponential":
            self.raw_distance_gate_decay = nn.Parameter(
                torch.full((num_heads,), math.log(math.expm1(1.0)))
            )
        else:
            self.register_parameter("raw_distance_gate_decay", None)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances_um: torch.Tensor,
        center_index: torch.Tensor,
        *,
        return_attention: bool = False,
        validate_topology: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return spatial context [C,D], optionally edge attention weights [E,H].

        ``node_embeddings``: [N,D] V1 embeddings built from visible genes.
        ``center_index``: [C] distinct local indices for receiving cells.
        ``edge_index``: [2,E], receiver center first, source neighbor second.
        ``edge_distances_um``: [E] nonnegative distances in micrometers.
        All edge receivers must appear in ``center_index``. Self-edges are
        forbidden. The caller must use a validated, partition-local graph.
        ``validate_topology=False`` skips costly content scans and is intended
        only for already checked, unchanged graph tensors in a hot loop.
        """
        if not isinstance(node_embeddings, torch.Tensor) or node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [N,D].")
        n, d = node_embeddings.shape
        if n < 1 or d != self.cell_dim or not node_embeddings.is_floating_point():
            raise ValueError("Nonempty floating embeddings must match cell_dim.")
        if node_embeddings.device != self.query.weight.device:
            raise ValueError("Model and embeddings must be on the same device.")
        if not isinstance(validate_topology, bool):
            raise TypeError("validate_topology must be bool.")
        if not isinstance(center_index, torch.Tensor) or center_index.ndim != 1 or center_index.dtype != torch.long:
            raise TypeError("center_index must be a 1D torch.long tensor.")
        if center_index.numel() == 0 or center_index.numel() > n:
            raise ValueError("At least one distinct center is required.")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise TypeError("edge_index must be a [2,E] torch.long tensor.")
        e = edge_index.shape[1]
        if not isinstance(edge_distances_um, torch.Tensor) or edge_distances_um.ndim != 1 or edge_distances_um.numel() != e or not edge_distances_um.is_floating_point():
            raise TypeError("edge_distances_um must be floating [E].")
        if any(item.device != node_embeddings.device for item in (center_index, edge_index, edge_distances_um)):
            raise ValueError("Graph tensors and features must share one device.")
        if validate_topology:
            if not torch.isfinite(node_embeddings).all().item():
                raise ValueError("Nonfinite node embeddings are not allowed.")
            if (center_index < 0).any().item() or (center_index >= n).any().item():
                raise IndexError("A center row lies outside the embedding batch.")
            if torch.unique(center_index).numel() != center_index.numel():
                raise ValueError("Duplicate center nodes are forbidden.")
            if not torch.isfinite(edge_distances_um).all().item() or (edge_distances_um < 0).any().item():
                raise ValueError("Distances must be finite and nonnegative.")
            if (edge_distances_um > self.max_distance_um + 1e-5).any().item():
                raise ValueError("Edge distance exceeds the configured graph radius.")
            if e:
                if (edge_index < 0).any().item() or (edge_index >= n).any().item():
                    raise IndexError("An edge endpoint lies outside the embedding batch.")
                if (edge_index[0] == edge_index[1]).any().item():
                    raise ValueError("Self-edges are forbidden in the spatial graph.")
                if torch.unique(edge_index.T, dim=0).shape[0] != e:
                    raise ValueError("Duplicate spatial edges are forbidden.")

        c = center_index.numel()
        center_of = torch.full((n,), -1, device=node_embeddings.device, dtype=torch.long)
        center_of[center_index] = torch.arange(c, device=node_embeddings.device)
        receiver = center_of[edge_index[0]]
        if validate_topology:
            if (receiver < 0).any().item():
                raise ValueError("Every edge receiver must be listed in center_index.")
            if self.max_neighbors is not None and e:
                degree = torch.bincount(receiver, minlength=c)
                if (degree > self.max_neighbors).any().item():
                    raise ValueError("A center has more than max_neighbors incoming edges.")

        normalized = self.input_norm(node_embeddings)
        queries = self.query(normalized[center_index]).reshape(c, self.num_heads, self.head_dim)
        keys = self.key(normalized).reshape(n, self.num_heads, self.head_dim)
        values = self.value(normalized).reshape(n, self.num_heads, self.head_dim)

        if not e:
            # Keep a differentiable zero even when an isolated minibatch has no edges.
            context = self.output(values.new_zeros((c, d)))
            context = context + (queries.sum() + keys.sum() + values.sum()) * 0.0
            if return_attention:
                return context, node_embeddings.new_zeros((0, self.num_heads))
            return context

        src = edge_index[1]
        scores = (queries[receiver].float() * keys[src].float()).sum(-1)
        scores = scores * (self.head_dim ** -0.5)
        decay = F.softplus(self.raw_distance_decay.float())
        scores = scores - (
            edge_distances_um.float() / self.max_distance_um
        ).unsqueeze(1) * decay.unsqueeze(0)

        # Segment softmax over incoming edges for each center and head.
        head_index = receiver.unsqueeze(1).expand(e, self.num_heads)
        maximum = torch.full(
            (c, self.num_heads), -torch.inf, device=scores.device, dtype=scores.dtype
        )
        maximum.scatter_reduce_(0, head_index, scores, reduce="amax", include_self=True)
        numerator = torch.exp(scores - maximum[receiver])
        normalizer = torch.zeros_like(maximum)
        normalizer.scatter_add_(0, head_index, numerator)
        weights = numerator / normalizer[receiver].clamp_min(torch.finfo(scores.dtype).tiny)
        used_weights = F.dropout(weights, p=self.attention_dropout, training=self.training)

        if self.distance_gate == "exponential":
            gate_decay = F.softplus(self.raw_distance_gate_decay.float())
            gate = torch.exp(
                -(edge_distances_um.float() / self.max_distance_um).unsqueeze(1)
                * gate_decay.unsqueeze(0)
            )
            used_weights = used_weights * gate
        messages = used_weights.unsqueeze(-1) * values[src].float()
        aggregate = torch.zeros(
            (c, self.num_heads, self.head_dim),
            device=scores.device, dtype=scores.dtype,
        )
        aggregate.index_add_(0, receiver, messages)
        context = self.output(aggregate.reshape(c, d).to(node_embeddings.dtype))
        context = self.projection_dropout(context)
        if return_attention:
            return context, weights
        return context
