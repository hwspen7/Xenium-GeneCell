"""V1-compatible intrinsic encoder plus V2.0-B spatial context extraction.

This module never changes the V1 reconstruction decoder. A spatially fused
reconstruction head will be introduced and evaluated separately in V2.0-C.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.gene_aware_mae import GeneAwareMAE
from src.models.spatial_attention import DistanceAwareSpatialAttention


class SpatialContextBackbone(nn.Module):
    """Compute V1 intrinsic and one-hop spatial embeddings on center cells.

    ``expression``, ``gene_ids`` and masks must be ordered using the
    ``h5_row_indices`` of the SpatialOneHopBatch. All nodes, including halo
    neighbors, must pass through the same V1 masking protocol. Only center
    rows are returned. V1 logits are explicitly named ``v1_*`` to prevent
    confusion with spatially decoded predictions (not implemented in V2-B).
    """

    def __init__(
        self,
        v1_model: GeneAwareMAE,
        attention: DistanceAwareSpatialAttention,
    ) -> None:
        super().__init__()
        if not isinstance(v1_model, GeneAwareMAE):
            raise TypeError("v1_model must be an existing GeneAwareMAE.")
        if not isinstance(attention, DistanceAwareSpatialAttention):
            raise TypeError("attention must be DistanceAwareSpatialAttention.")
        if v1_model.cell_projection.out_features != attention.cell_dim:
            raise ValueError("The V1 cell dimension and spatial attention dimension differ.")
        self.v1_model = v1_model
        self.attention = attention

    def forward(
        self,
        *,
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
        measured_mask: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        center_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances_um: torch.Tensor,
        target_gene_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        result = self.v1_model(
            expression=expression,
            gene_ids=gene_ids,
            measured_mask=measured_mask,
            reconstruction_mask=reconstruction_mask,
            target_gene_ids=target_gene_ids,
        )
        spatial = self.attention(
            result["cell_embeddings"],
            edge_index=edge_index,
            edge_distances_um=edge_distances_um,
            center_index=center_index,
            return_attention=return_attention,
        )
        if return_attention:
            spatial_context, weights = spatial
        else:
            spatial_context = spatial

        # Only center rows are prediction targets; halo nodes are context.
        c = center_index
        outputs = {
            "intrinsic_cell_embeddings": result["cell_embeddings"].index_select(0, c),
            "spatial_context_embeddings": spatial_context,
            "v1_presence_logits": result["presence_logits"].index_select(0, c),
            "v1_positive_intensity": result["positive_intensity"].index_select(0, c),
            "target_counts": result["target_counts"].index_select(0, c),
            "target_measured_mask": result["target_measured_mask"].index_select(0, c),
            "target_reconstruction_mask": result["target_reconstruction_mask"].index_select(0, c),
            "target_valid_mask": result["target_valid_mask"].index_select(0, c),
        }
        if return_attention:
            outputs["spatial_attention_weights"] = weights
        return outputs
