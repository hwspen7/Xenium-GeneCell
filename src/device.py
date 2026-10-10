"""Cross-platform PyTorch device selection."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, MPS or CPU; fail explicitly for an unavailable requested backend."""
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {requested!r}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)
