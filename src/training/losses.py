"""Masked dual-head reconstruction loss for SG-MAE V1.0.

The model aligns requested target gene IDs with original expression columns.
Only deliberately masked, measured genes contribute to reconstruction loss.
"""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


class MaskedReconstructionLoss(nn.Module):
    """Compute masked presence BCE and positive-count log1p Huber loss.

    Presence: BCEWithLogits on all masked, measured target genes, including
    measured zeros.

    Intensity: Smooth L1 (Huber, beta=1.0 by default) only on masked,
    measured target genes whose raw count is strictly positive.

    An output dictionary from GeneAwareMAE.forward is required. It must
    contain aligned target_counts, target_measured_mask,
    target_reconstruction_mask, and target_valid_mask in addition to the two
    prediction heads.

    If no relevant targets exist, the corresponding loss is a differentiable
    zero. No unmeasured, unmasked, or padded genes contribute to either loss.
    """

    REQUIRED_KEYS = (
        "presence_logits",
        "positive_intensity",
        "target_counts",
        "target_measured_mask",
        "target_reconstruction_mask",
        "target_valid_mask",
    )

    def __init__(
        self,
        lambda_presence: float = 1.0,
        lambda_intensity: float = 1.0,
        huber_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_presence = self._check_weight(lambda_presence, "lambda_presence")
        self.lambda_intensity = self._check_weight(lambda_intensity, "lambda_intensity")
        if self.lambda_presence == 0 and self.lambda_intensity == 0:
            raise ValueError("At least one loss weight must be positive.")
        if isinstance(huber_beta, bool) or not isinstance(huber_beta, Real):
            raise TypeError("huber_beta must be a positive real number.")
        self.huber_beta = float(huber_beta)
        if not math.isfinite(self.huber_beta) or self.huber_beta <= 0:
            raise ValueError("huber_beta must be finite and positive.")

    @staticmethod
    def _check_weight(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return weight

    def forward(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return total/individual losses and their exact normalization counts."""
        if not isinstance(outputs, dict):
            raise TypeError("Model outputs must be a dictionary.")
        missing = [name for name in self.REQUIRED_KEYS if name not in outputs]
        if missing:
            raise KeyError(f"Missing model outputs: {', '.join(missing)}")

        logits = outputs["presence_logits"]
        intensities = outputs["positive_intensity"]
        counts = outputs["target_counts"]
        measured = outputs["target_measured_mask"]
        reconstruction = outputs["target_reconstruction_mask"]
        valid = outputs["target_valid_mask"]

        if not all(isinstance(tensor, torch.Tensor) for tensor in (
            logits, intensities, counts, measured, reconstruction, valid
        )):
            raise TypeError("All required model outputs must be torch.Tensors.")
        if logits.ndim != 2 or min(logits.shape) < 1:
            raise ValueError("Expected nonempty [B, T] prediction tensors.")
        if any(t.shape != logits.shape for t in (
            intensities, counts, measured, reconstruction, valid
        )):
            raise ValueError("All target tensors must have matching [B, T] shapes.")
        if not all(t.device == logits.device for t in (
            intensities, counts, measured, reconstruction, valid
        )):
            raise ValueError("All target tensors must use the same device.")
        if not all(t.is_floating_point() for t in (logits, intensities, counts)):
            raise TypeError("Predictions and target counts must be floating point.")
        if any(t.dtype != torch.bool for t in (measured, reconstruction, valid)):
            raise TypeError("Target masks must have dtype torch.bool.")
        if (measured & ~valid).any().item():
            raise ValueError("Measured targets must have valid Gene IDs.")
        if (reconstruction & ~measured).any().item():
            raise ValueError("Reconstruction targets must be measured.")

        # Labels for unmeasured genes are ignored and may be nonfinite.
        measured_counts = counts[measured]
        if not torch.isfinite(measured_counts).all().item():
            raise ValueError("Measured target counts must be finite.")
        if (measured_counts < 0).any().item():
            raise ValueError("Measured target counts must be nonnegative.")

        supervised = reconstruction & measured & valid
        positive = supervised & (counts > 0)
        num_masked = supervised.sum()
        num_positive = positive.sum()

        # Index before computing either loss to exclude non-target values.
        selected_logits = logits[supervised]
        selected_intensities = intensities[positive]
        selected_counts = counts[supervised]
        positive_counts = counts[positive]

        if not torch.isfinite(selected_logits).all().item():
            raise ValueError("Supervised presence logits must be finite.")
        if not torch.isfinite(selected_intensities).all().item():
            raise ValueError("Supervised intensity estimates must be finite.")
        if (selected_intensities < 0).any().item():
            raise ValueError("Positive intensity estimates must be nonnegative.")

        if selected_logits.numel():
            presence_loss_sum = F.binary_cross_entropy_with_logits(
                selected_logits,
                (selected_counts > 0).to(dtype=selected_logits.dtype),
                reduction="sum",
            )
            presence_loss = presence_loss_sum / num_masked
        else:
            presence_loss_sum = selected_logits.sum()
            presence_loss = presence_loss_sum

        if selected_intensities.numel():
            positive_target = torch.log1p(positive_counts).to(
                dtype=selected_intensities.dtype
            )
            intensity_loss_sum = F.smooth_l1_loss(
                selected_intensities,
                positive_target,
                reduction="sum",
                beta=self.huber_beta,
            )
            intensity_loss = intensity_loss_sum / num_positive
        else:
            intensity_loss_sum = selected_intensities.sum()
            intensity_loss = intensity_loss_sum

        total_loss = (
            self.lambda_presence * presence_loss
            + self.lambda_intensity * intensity_loss
        )

        return {
            "total_loss": total_loss,
            "presence_loss": presence_loss,
            "intensity_loss": intensity_loss,
            "presence_loss_sum": presence_loss_sum,
            "intensity_loss_sum": intensity_loss_sum,
            "num_masked": num_masked,
            "num_positive": num_positive,
        }
