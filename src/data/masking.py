"""Measured-gene-only random masking for SG-MAE V1.0.

A reconstruction mask marks measured counts deliberately hidden from the
encoder. It never includes padding or unmeasured gene positions.
"""

import math
from numbers import Integral, Real

import torch


def _validate_mask_ratio(mask_ratio: float) -> float:
    """Require a finite masking fraction in [0, 1]."""
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
        raise TypeError("mask_ratio must be a real number.")
    ratio = float(mask_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("mask_ratio must be finite and in [0, 1].")
    return ratio


def _validate_measured_mask(measured_mask: torch.Tensor) -> None:
    """Validate a batch-first mask without inferring measurement from counts."""
    if not isinstance(measured_mask, torch.Tensor):
        raise TypeError("measured_mask must be a torch.Tensor.")
    if measured_mask.ndim != 2 or min(measured_mask.shape) < 1:
        raise ValueError("measured_mask must have nonempty shape [B, G].")
    if measured_mask.dtype != torch.bool:
        raise TypeError("measured_mask must have dtype torch.bool.")


def _validate_gene_ids(
    gene_ids: torch.Tensor,
    measured_mask: torch.Tensor,
) -> None:
    """Optionally check that padded or duplicate IDs cannot enter masking."""
    if not isinstance(gene_ids, torch.Tensor):
        raise TypeError("gene_ids must be a torch.Tensor.")
    if gene_ids.dtype != torch.long:
        raise TypeError("gene_ids must have dtype torch.long.")
    if gene_ids.device != measured_mask.device:
        raise ValueError("gene_ids and measured_mask must use the same device.")
    batch_size, width = measured_mask.shape
    if gene_ids.ndim == 1 and gene_ids.shape[0] == width:
        ids = gene_ids.unsqueeze(0).expand(batch_size, -1)
    elif gene_ids.ndim == 2 and gene_ids.shape == (batch_size, width):
        ids = gene_ids
    else:
        raise ValueError("gene_ids must have shape [G] or [B, G].")
    if (ids < -1).any().item():
        raise ValueError("gene_ids contains values below the padding ID -1.")
    if (measured_mask & (ids == -1)).any().item():
        raise ValueError("A padding position cannot be measured.")

    # Each real gene ID must occur at most once per cell, including unmeasured IDs.
    sorted_ids = torch.sort(ids.masked_fill(ids == -1, torch.iinfo(torch.long).max), dim=1).values
    duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (
        sorted_ids[:, 1:] != torch.iinfo(torch.long).max
    )
    if duplicate.any().item():
        raise ValueError("Duplicate valid gene IDs in a cell.")


def generate_reconstruction_mask(
    measured_mask: torch.Tensor,
    mask_ratio: float = 0.2,
    *,
    gene_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    ensure_visible: bool = True,
) -> torch.Tensor:
    """Uniformly mask measured gene positions independently of their counts.

    Args:
        measured_mask: Boolean [B, G]; includes measured zeros as True.
        mask_ratio: Target fraction of measured genes to hide per cell.
        gene_ids: Optional [G] or [B, G] for padding and ID validation.
        generator: Optional CPU torch.Generator for reproducible sampling.
        ensure_visible: For cells with at least two measured genes, keep at
            least one visible. The sole measured gene can still be hidden.

    Returns:
        Boolean [B, G]. The number of masked entries per row is
        round_half_up(mask_ratio * measured_count), with at least one when
        0 < mask_ratio and measured_count > 0. If ensure_visible=True,
        cap this at measured_count - 1 for rows with >= 2 measured genes.
        Zero measured genes or mask_ratio=0 always produce zero targets.

    Random draws use CPU even when tensors reside on a GPU. This makes a
    seeded CPU generator usable in any training environment. For identical
    masks across validation epochs, reset the generator seed and iterate
    cells in the same batch order and batch sizes.
    """
    _validate_measured_mask(measured_mask)
    ratio = _validate_mask_ratio(mask_ratio)
    if not isinstance(ensure_visible, bool):
        raise TypeError("ensure_visible must be a bool.")
    if gene_ids is not None:
        _validate_gene_ids(gene_ids, measured_mask)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None.")
        if generator.device.type != "cpu":
            raise ValueError("generator must be a CPU torch.Generator.")

    if ratio == 0.0:
        return torch.zeros_like(measured_mask)

    # Sample on CPU to support CPU-seeded reproducibility on CUDA devices.
    measured_cpu = measured_mask.to(device="cpu")
    batch_size, width = measured_cpu.shape
    measured_counts = measured_cpu.sum(dim=1)

    # Use explicit half-up rounding to avoid ambiguous banker's rounding.
    hide_counts = torch.floor(measured_counts.to(torch.float64) * ratio + 0.5).long()
    hide_counts = torch.where(
        measured_counts > 0,
        hide_counts.clamp_min(1),
        torch.zeros_like(hide_counts),
    )
    if ensure_visible:
        maximum = torch.where(
            measured_counts >= 2,
            measured_counts - 1,
            measured_counts,
        )
        hide_counts = torch.minimum(hide_counts, maximum)

    # Rank only measured positions; unmeasured and padding get score -1.
    scores = torch.rand(
        (batch_size, width),
        generator=generator,
        device="cpu",
    )
    scores.masked_fill_(~measured_cpu, -1.0)
    selected_order = torch.argsort(scores, dim=1, descending=True)
    selected_rank = torch.empty_like(selected_order)
    selected_rank.scatter_(
        1,
        selected_order,
        torch.arange(width).unsqueeze(0).expand(batch_size, -1),
    )
    mask_cpu = measured_cpu & (selected_rank < hide_counts.unsqueeze(1))
    return mask_cpu.to(device=measured_mask.device)


def generate_random_mask(
    expression: torch.Tensor,
    mask_ratio: float = 0.2,
    generator: torch.Generator | None = None,
    measured_mask: torch.Tensor | None = None,
    *,
    gene_ids: torch.Tensor | None = None,
    ensure_visible: bool = True,
) -> torch.Tensor:
    """Backward-compatible wrapper for the legacy masking interface.

    For new SG-MAE training, pass measured_mask (or call
    generate_reconstruction_mask directly). When omitted, every expression
    position is assumed measured, which is only valid for a full panel
    without padding. Expression values are deliberately ignored.
    """
    if not isinstance(expression, torch.Tensor):
        raise TypeError("expression must be a torch.Tensor.")
    if expression.ndim != 2 or min(expression.shape) < 1:
        raise ValueError("expression must have nonempty shape [B, G].")
    if measured_mask is None:
        measured_mask = torch.ones_like(expression, dtype=torch.bool)
    elif measured_mask.shape != expression.shape or measured_mask.device != expression.device:
        raise ValueError("measured_mask must match expression shape and device.")

    return generate_reconstruction_mask(
        measured_mask,
        mask_ratio=mask_ratio,
        gene_ids=gene_ids,
        generator=generator,
        ensure_visible=ensure_visible,
    )
