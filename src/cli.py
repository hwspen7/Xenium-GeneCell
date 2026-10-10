"""CLI foundation. Actual MAE training/inference will be added in later phases."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import sys

import torch

from .device import select_device


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def smoke_test(device: torch.device) -> float:
    """Run a synthetic forward/backward/optimizer step, not a research-model train."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(300, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 300),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(64, 300, device=device)
    mask = torch.rand(64, 300, device=device) < 0.2
    corrupted = features.masked_fill(mask, 0.0)
    prediction = model(corrupted)
    loss = (prediction[mask] - features[mask]).abs().mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Smoke-test loss is not finite.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return float(loss.item())


def _doctor(args: argparse.Namespace) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    for package in ("torch", "numpy", "scanpy", "anndata"):
        print(f"{package}: {_version(package)}")
    print(f"CUDA runtime in PyTorch: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    device = select_device(args.device)
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.compute_test:
        loss = smoke_test(device)
        print(f"Compute test passed, synthetic loss: {loss:.6f}")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    loss = smoke_test(device)
    print(f"Synthetic training smoke test passed ({device}), loss: {loss:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cellmae", description="Xenium-GeneCell CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect Python, packages and compute backend")
    doctor.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    doctor.add_argument("--compute-test", action="store_true", help="Run a synthetic compute step")
    doctor.set_defaults(func=_doctor)

    smoke = subparsers.add_parser("smoke-test", help="Run synthetic forward/backward step")
    smoke.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    smoke.set_defaults(func=_smoke)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
