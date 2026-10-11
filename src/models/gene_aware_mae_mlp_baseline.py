
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual feed-forward block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class GeneExpressionEncoder(nn.Module):
    """Encode masked gene expression into cell embeddings."""

    def __init__(
        self,
        num_genes: int = 300,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_genes = num_genes

        self.input_projection = nn.Sequential(
            nn.Linear(num_genes * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.latent_projection = nn.Linear(
            256, latent_dim
        )

    def forward(
        self,
        expression: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        if expression.ndim != 2:
            raise ValueError("Expression must be [B, G].")

        if expression.shape[1] != self.num_genes:
            raise ValueError("Incorrect number of genes.")

        if mask.shape != expression.shape:
            raise ValueError("Mask shape must match expression.")

        if not torch.isfinite(expression).all():
            raise ValueError("Expression contains non-finite values.")

        if (expression < 0).any():
            raise ValueError("Expression counts must be non-negative.")

        mask = mask.bool()

        # True indicates a hidden expression entry.
        masked_expression = expression.masked_fill(mask, 0.0)

        # Apply log1p after masking; no hidden-count normalization.
        masked_expression = torch.log1p(masked_expression)

        x = torch.cat(
            [masked_expression, mask.float()],
            dim=-1,
        )

        x = self.input_projection(x)
        x = self.residual_blocks(x)
        x = self.feature_projection(x)

        return self.latent_projection(x)


class GeneConditionedDecoder(nn.Module):
    """Predict gene detection and positive expression."""

    def __init__(
        self,
        num_genes: int = 300,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.gene_embeddings = nn.Embedding(
            num_genes, latent_dim
        )

        self.presence_projection = nn.Linear(
            latent_dim, latent_dim
        )

        self.intensity_projection = nn.Linear(
            latent_dim, latent_dim
        )

        self.presence_bias = nn.Parameter(
            torch.zeros(num_genes)
        )

        self.intensity_bias = nn.Parameter(
            torch.zeros(num_genes)
        )

        nn.init.normal_(
            self.gene_embeddings.weight,
            mean=0.0,
            std=0.02,
        )

        self.scale = math.sqrt(latent_dim)

    def forward(
        self,
        cell_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gene_embeddings = self.gene_embeddings.weight

        presence_features = self.presence_projection(
            cell_embeddings
        )

        intensity_features = self.intensity_projection(
            cell_embeddings
        )

        presence_logits = (
            presence_features @ gene_embeddings.T
        ) / self.scale + self.presence_bias

        intensity_logits = (
            intensity_features @ gene_embeddings.T
        ) / self.scale + self.intensity_bias

        # Positive expression is modeled on the log1p scale.
        positive_intensity = F.softplus(intensity_logits)

        return presence_logits, positive_intensity


class GeneAwareMAE(nn.Module):
    """Gene-aware masked autoencoder for Xenium data."""

    def __init__(
        self,
        num_genes: int = 300,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_genes = num_genes

        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.decoder = GeneConditionedDecoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
        )

    def forward(
        self,
        expression: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        cell_embeddings = self.encoder(
            expression, mask
        )

        presence_logits, positive_intensity = self.decoder(
            cell_embeddings
        )

        return {
            "cell_embeddings": cell_embeddings,
            "gene_embeddings": self.decoder.gene_embeddings.weight,
            "presence_logits": presence_logits,
            "positive_intensity": positive_intensity,
        }
