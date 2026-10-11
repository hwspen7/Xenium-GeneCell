"""Scalable Gene-Aware Masked Autoencoder (SG-MAE), V1.0."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTokenBlock(nn.Module):
    """Shared, position-wise feed-forward block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class LatentAttentionBlock(nn.Module):
    """Self-attention and feed-forward mixing in latent space."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = ResidualTokenBlock(dim, dropout)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm(latents)
        update, _ = self.attention(x, x, x, need_weights=False)
        return self.ffn(latents + update)


class GeneAwareMAE(nn.Module):
    """
    Variable-length Gene-Token MAE with a gene-conditioned dual-head decoder.

    gene_ids: [G] or [B, G], with -1 marking padding.
    measured_mask: [B, G], true only if the gene was measured.
    reconstruction_mask: [B, G], true only for deliberately hidden measurements.
    expression: [B, G], nonnegative raw counts at measured entries.
    target_gene_ids: [T] or [B, T], optional gene IDs to decode.

    Predictions use only visible measured counts. Labels and supervision masks
    are returned separately and aligned to the requested target gene IDs.
    """

    def __init__(
        self,
        vocab_size: int,
        token_dim: int = 128,
        cell_dim: int = 128,
        num_latents: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(vocab_size, token_dim, cell_dim, num_latents, num_heads) < 1:
            raise ValueError("All model dimensions must be positive.")
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.gene_embeddings = nn.Embedding(vocab_size, token_dim)
        self.expression_projection = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.detection_embeddings = nn.Embedding(2, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_encoder = nn.Sequential(
            ResidualTokenBlock(token_dim, dropout),
            ResidualTokenBlock(token_dim, dropout),
        )

        self.latent_queries = nn.Parameter(torch.empty(num_latents, token_dim))
        self.empty_cell_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.cross_query_norm = nn.LayerNorm(token_dim)
        self.cross_key_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.latent_block = LatentAttentionBlock(token_dim, num_heads, dropout)
        self.cell_norm = nn.LayerNorm(token_dim)
        self.cell_projection = nn.Linear(token_dim, cell_dim)

        self.presence_projection = nn.Linear(cell_dim, token_dim)
        self.intensity_projection = nn.Linear(cell_dim, token_dim)
        self.presence_bias = nn.Embedding(vocab_size, 1)
        self.intensity_bias = nn.Embedding(vocab_size, 1)
        self.decoder_scale = math.sqrt(token_dim)

        nn.init.normal_(self.gene_embeddings.weight, std=0.02)
        nn.init.normal_(self.detection_embeddings.weight, std=0.02)
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.empty_cell_token, std=0.02)
        nn.init.zeros_(self.presence_bias.weight)
        nn.init.zeros_(self.intensity_bias.weight)

    def expand_vocabulary(self, new_vocab_size: int) -> None:
        """
        Expand gene IDs while preserving existing learned gene parameters.

        Call before constructing the optimizer, or reconstruct the optimizer
        after expanding because embedding Parameter objects are replaced.
        New genes are initialized and require training.
        """
        if new_vocab_size <= self.vocab_size:
            raise ValueError("new_vocab_size must exceed the current vocabulary.")

        device = self.gene_embeddings.weight.device
        dtype = self.gene_embeddings.weight.dtype
        old_size = self.vocab_size

        new_genes = nn.Embedding(new_vocab_size, self.token_dim, device=device, dtype=dtype)
        new_presence_bias = nn.Embedding(new_vocab_size, 1, device=device, dtype=dtype)
        new_intensity_bias = nn.Embedding(new_vocab_size, 1, device=device, dtype=dtype)

        nn.init.normal_(new_genes.weight, std=0.02)
        nn.init.zeros_(new_presence_bias.weight)
        nn.init.zeros_(new_intensity_bias.weight)
        with torch.no_grad():
            new_genes.weight[:old_size].copy_(self.gene_embeddings.weight)
            new_presence_bias.weight[:old_size].copy_(self.presence_bias.weight)
            new_intensity_bias.weight[:old_size].copy_(self.intensity_bias.weight)

        self.gene_embeddings = new_genes
        self.presence_bias = new_presence_bias
        self.intensity_bias = new_intensity_bias
        self.vocab_size = new_vocab_size

    def _broadcast_ids(
        self, ids: torch.Tensor, batch_size: int, label: str
    ) -> torch.Tensor:
        if ids.dtype != torch.long:
            raise TypeError(f"{label} must have dtype torch.long.")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0).expand(batch_size, -1)
        elif ids.ndim != 2 or ids.shape[0] != batch_size:
            raise ValueError(f"{label} must have shape [G] or [B, G].")
        if ids.shape[1] == 0:
            raise ValueError(f"{label} cannot be empty.")
        if ((ids < -1) | (ids >= self.vocab_size)).any().item():
            raise ValueError(f"{label} contains invalid gene IDs.")

        # Sorting checks uniqueness in O(G log G) instead of O(G^2).
        sorted_ids = torch.sort(ids.masked_fill(ids == -1, self.vocab_size), dim=1).values
        duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (
            sorted_ids[:, 1:] != self.vocab_size
        )
        if duplicate.any().item():
            raise ValueError(f"{label} contains duplicate valid gene IDs.")
        return ids

    @staticmethod
    def _pack_visible(
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact visible entries and pad only up to the longest visible row."""
        batch_size = visible.shape[0]
        visible_counts = visible.sum(dim=1)
        max_visible = int(visible_counts.max().item())
        packed_ids = torch.zeros(
            batch_size, max_visible, dtype=torch.long, device=expression.device
        )
        packed_counts = expression.new_zeros(batch_size, max_visible)
        if max_visible:
            row, col = torch.where(visible)
            packed_col = visible.long().cumsum(dim=1)[row, col] - 1
            packed_ids[row, packed_col] = gene_ids[row, col]
            packed_counts[row, packed_col] = expression[row, col]
        packed_padding = (
            torch.arange(max_visible, device=expression.device).unsqueeze(0)
            >= visible_counts.unsqueeze(1)
        )
        return packed_ids, packed_counts, packed_padding, visible_counts

    def _aligned_targets(
        self,
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
        measured_mask: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Match target IDs to columns without assuming gene order."""
        safe_input_ids = gene_ids.masked_fill(gene_ids == -1, self.vocab_size)
        sorted_ids, sorted_positions = torch.sort(safe_input_ids, dim=1)

        found_index = torch.searchsorted(
            sorted_ids.contiguous(), target_ids.contiguous()
        )
        found_index = found_index.clamp_max(gene_ids.shape[1] - 1)
        original_columns = sorted_positions.gather(1, found_index)
        matched = (
            (target_ids >= 0)
            & (sorted_ids.gather(1, found_index) == target_ids)
        )
        measured = matched & measured_mask.gather(1, original_columns)
        supervise = measured & reconstruction_mask.gather(1, original_columns)
        raw_counts = expression.gather(1, original_columns)
        raw_counts = torch.where(
            measured, raw_counts, torch.zeros_like(raw_counts)
        )
        return {
            "target_counts": raw_counts,
            "target_measured_mask": measured,
            "target_reconstruction_mask": supervise,
            "target_valid_mask": target_ids >= 0,
        }

    def forward(
        self,
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
        measured_mask: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        target_gene_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if expression.ndim != 2 or not expression.is_floating_point():
            raise ValueError("expression must be floating point [B, G].")
        batch_size, gene_count = expression.shape
        if batch_size < 1 or gene_count < 1:
            raise ValueError("Input dimensions must be nonempty.")
        if measured_mask.shape != expression.shape or reconstruction_mask.shape != expression.shape:
            raise ValueError("Mask shapes must equal expression shape.")
        if measured_mask.dtype != torch.bool or reconstruction_mask.dtype != torch.bool:
            raise TypeError("Both masks must be boolean.")
        tensors = [expression, gene_ids, measured_mask, reconstruction_mask]
        if target_gene_ids is not None:
            tensors.append(target_gene_ids)
        if any(t.device != self.gene_embeddings.weight.device for t in tensors):
            raise ValueError("All tensors and the model must be on the same device.")
        input_ids = self._broadcast_ids(gene_ids, batch_size, "gene_ids")
        if input_ids.shape[1] != gene_count:
            raise ValueError("gene_ids length must match expression.")
        if (measured_mask & (input_ids == -1)).any().item():
            raise ValueError("Padding positions cannot be measured.")
        if (reconstruction_mask & ~measured_mask).any().item():
            raise ValueError("Only measured genes may be reconstruction targets.")
        if (reconstruction_mask & (input_ids == -1)).any().item():
            raise ValueError("Padding cannot be a reconstruction target.")
        measured_values = expression[measured_mask]
        if not torch.isfinite(measured_values).all().item() or (measured_values < 0).any().item():
            raise ValueError("Measured counts must be finite and nonnegative.")

        visible = measured_mask & ~reconstruction_mask
        ids, counts, padding, visible_counts = self._pack_visible(
            expression, input_ids, visible
        )

        tokens = self.gene_embeddings(ids) + self.expression_projection(
            torch.log1p(counts).unsqueeze(-1)
        ) + self.detection_embeddings((counts > 0).long())
        tokens = self.token_encoder(self.token_norm(tokens))

        # A learned empty-cell token avoids an all-masked attention key sequence.
        null_token = self.empty_cell_token.expand(batch_size, -1, -1)
        tokens = torch.cat([tokens, null_token], dim=1)
        padding = torch.cat([padding, (visible_counts > 0).unsqueeze(1)], dim=1)

        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        cross_output, _ = self.cross_attention(
            self.cross_query_norm(latents),
            self.cross_key_norm(tokens),
            self.cross_key_norm(tokens),
            key_padding_mask=padding,
            need_weights=False,
        )
        latents = self.latent_block(latents + cross_output)
        cell_embeddings = self.cell_projection(self.cell_norm(latents.mean(dim=1)))

        if target_gene_ids is None:
            target_gene_ids = gene_ids
        target_ids = self._broadcast_ids(target_gene_ids, batch_size, "target_gene_ids")
        aligned = self._aligned_targets(
            expression, input_ids, measured_mask, reconstruction_mask, target_ids
        )

        safe_targets = target_ids.clamp_min(0)
        gene_vectors = self.gene_embeddings(safe_targets)
        presence_features = self.presence_projection(cell_embeddings)
        intensity_features = self.intensity_projection(cell_embeddings)
        presence_logits = (
            (gene_vectors * presence_features.unsqueeze(1)).sum(-1)
            / self.decoder_scale
            + self.presence_bias(safe_targets).squeeze(-1)
        )
        intensity_logits = (
            (gene_vectors * intensity_features.unsqueeze(1)).sum(-1)
            / self.decoder_scale
            + self.intensity_bias(safe_targets).squeeze(-1)
        )
        valid = aligned["target_valid_mask"]
        return {
            "cell_embeddings": cell_embeddings,
            "gene_embeddings": self.gene_embeddings.weight,
            "presence_logits": presence_logits.masked_fill(~valid, 0.0),
            "positive_intensity": F.softplus(intensity_logits).masked_fill(~valid, 0.0),
            **aligned,
        }
