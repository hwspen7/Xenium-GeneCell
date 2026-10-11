"""Reproducible ShamL-only trainer for the SG-MAE V1.0 model.

The training entrypoint verifies the frozen spatial split before creating
loaders. This module never accesses ShamR, Hour12R, or Day2L.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.data.masking import generate_reconstruction_mask
from src.dataset import SGMAECellDataset, collate_sgmae_batch


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class SchedulerConfig:
    """Validated ReduceLROnPlateau options for a new experiment."""

    name: str = "reduce_on_plateau"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: str = "abs"
    cooldown: int = 1
    min_lr: float = 1e-5

    def __post_init__(self):
        if self.name != "reduce_on_plateau":
            raise ValueError("Only reduce_on_plateau scheduler is supported.")
        for name in ("patience", "cooldown"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"scheduler {name} must be a nonnegative integer.")
        for name in ("factor", "threshold", "min_lr"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"scheduler {name} must be numeric.")
            if not math.isfinite(value):
                raise ValueError(f"scheduler {name} must be finite.")
        if not 0 < self.factor < 1:
            raise ValueError("scheduler factor must be in (0, 1).")
        if self.threshold < 0 or self.min_lr < 0:
            raise ValueError("scheduler threshold and min_lr must be nonnegative.")
        if self.threshold_mode not in ("abs", "rel"):
            raise ValueError("scheduler threshold_mode must be abs or rel.")


@dataclass(frozen=True)
class TrainerConfig:
    """Training options. max_epochs may be extended when resuming."""

    max_epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    mask_ratio: float = 0.20
    grad_clip_norm: float = 1.0
    patience: int = 12
    min_delta: float = 0.0
    seed: int = 42
    validation_mask_seed: int = 314159
    scheduler: SchedulerConfig | dict | None = None

    def __post_init__(self):
        if isinstance(self.scheduler, dict):
            object.__setattr__(self, "scheduler", SchedulerConfig(**self.scheduler))
        elif self.scheduler is not None and not isinstance(self.scheduler, SchedulerConfig):
            raise TypeError("scheduler must be a mapping, SchedulerConfig, or None.")
        if self.scheduler is not None and self.scheduler.min_lr > self.learning_rate:
            raise ValueError("scheduler min_lr cannot exceed initial learning_rate.")
        for field in ("max_epochs", "batch_size", "patience"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer.")
        for field in ("seed", "validation_mask_seed"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer.")
        for field in ("learning_rate", "mask_ratio", "grad_clip_norm"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{field} must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive.")
        if self.learning_rate <= 0 or not (0 < self.mask_ratio <= 1):
            raise ValueError("learning_rate must be positive and mask_ratio in (0, 1].")
        for field in ("weight_decay", "min_delta"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{field} must be numeric.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative.")

    def frozen_options(self) -> dict:
        """Options which must not change when continuing one experiment."""
        result = asdict(self)
        result.pop("max_epochs")
        # Preserve the metadata schema used by existing fixed-LR checkpoints.
        if result["scheduler"] is None:
            result.pop("scheduler")
        return result


def seed_training(seed: int) -> None:
    """Seed training initialization and disable nondeterministic CuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_shaml_loaders(dataset: SGMAECellDataset, split, config: TrainerConfig):
    """Create disjoint loaders from a previously verified frozen ShamL split."""
    if not isinstance(dataset, SGMAECellDataset):
        raise TypeError("Expected SGMAECellDataset, not a generic or mixed sample.")
    if not (dataset.sample_name == "ShamL" or dataset.sample_name.startswith("ShamL-")):
        raise ValueError("V1.0 training is restricted to ShamL.")
    if split.sample_name != "ShamL":
        raise ValueError("Frozen manifest must identify ShamL.")
    if tuple(dataset.cell_ids) != split.cell_ids:
        raise ValueError("Spatial split indices are not aligned with dataset cell order.")
    if sum(split.counts.values()) != len(dataset):
        raise ValueError("Spatial split does not cover every ShamL cell.")
    train_indices = split.train_indices.tolist()
    validation_indices = split.validation_indices.tolist()
    excluded_indices = split.excluded_indices.tolist()
    if not train_indices or not validation_indices:
        raise ValueError("Both training and validation sets must be nonempty.")
    if (
        len(set(train_indices + validation_indices + excluded_indices)) != len(dataset)
        or min(train_indices + validation_indices + excluded_indices) < 0
        or max(train_indices + validation_indices + excluded_indices) >= len(dataset)
    ):
        raise ValueError("Spatial split indices do not form a complete disjoint partition.")

    shuffle_generator = torch.Generator(device="cpu").manual_seed(config.seed + 17)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        collate_fn=collate_sgmae_batch,
        drop_last=False,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        generator=torch.Generator(device="cpu").manual_seed(config.validation_mask_seed + 37),
        num_workers=0,
        collate_fn=collate_sgmae_batch,
        drop_last=False,
    )
    return train_loader, validation_loader, shuffle_generator


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a complete checkpoint to a temporary file before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_torch_load(path: Path) -> dict:
    """Load only tensor and primitive checkpoint contents."""
    return torch.load(path, map_location="cpu", weights_only=True)


class SGMAETrainer:
    """SG-MAE trainer with exact epoch statistics and validated checkpoints."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        shuffle_generator: torch.Generator,
        config: TrainerConfig,
        model_config: dict,
        vocabulary_fingerprint: str,
        split_fingerprint: str,
        data_sha256: str,
        run_dir: str | Path,
        device: str | torch.device = "cpu",
        resume: bool = False,
    ):
        for field, value in (
            ("vocabulary_fingerprint", vocabulary_fingerprint),
            ("split_fingerprint", split_fingerprint),
            ("data_sha256", data_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 hexadecimal digest.")
        if model_config.get("vocab_size") != model.vocab_size:
            raise ValueError("Model configuration and instantiated vocabulary size disagree.")
        if not isinstance(config, TrainerConfig):
            raise TypeError("config must be TrainerConfig.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.shuffle_generator = shuffle_generator
        self.train_mask_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1009)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = None
        if config.scheduler is not None:
            options = config.scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=options.factor,
                patience=options.patience,
                threshold=options.threshold,
                threshold_mode=options.threshold_mode,
                cooldown=options.cooldown,
                min_lr=options.min_lr,
            )
        self.run_dir = Path(run_dir)
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"
        self.metadata = {
            "sample_name": "ShamL",
            "vocabulary_fingerprint": vocabulary_fingerprint,
            "split_fingerprint": split_fingerprint,
            "data_sha256": data_sha256,
            "model_config": model_config.copy(),
            "trainer_config": config.frozen_options(),
            "loss_config": {
                "lambda_presence": criterion.lambda_presence,
                "lambda_intensity": criterion.lambda_intensity,
                "huber_beta": criterion.huber_beta,
            },
        }
        self.history: list[dict] = []
        self.completed_epoch = 0
        self.best_epoch = 0
        self.best_validation = math.inf
        self.bad_epochs = 0

        if resume:
            self._resume()
        elif self.last_path.exists() or self.best_path.exists():
            raise FileExistsError(
                "Checkpoint already exists. Use resume=True, or a new run directory."
            )

    def _verify_metadata(self, saved: dict):
        if not isinstance(saved, dict) or saved.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint format.")
        if saved.get("metadata") != self.metadata:
            raise ValueError(
                "Checkpoint model, loss, dataset, vocabulary, split or training config differs."
            )

    def _resume(self):
        if not self.last_path.is_file():
            raise FileNotFoundError("Cannot resume: last.pt does not exist.")
        checkpoint = _safe_torch_load(self.last_path)
        self._verify_metadata(checkpoint)
        if checkpoint.get("device_type") != self.device.type:
            raise ValueError("Changing CPU/GPU device type during resume is not supported.")
        if checkpoint.get("kind") != "last":
            raise ValueError("Expected a last-epoch training checkpoint.")
        if checkpoint["completed_epoch"] < 1 or not isinstance(checkpoint["history"], list):
            raise ValueError("Checkpoint epoch/history fields are invalid.")
        if len(checkpoint["history"]) != checkpoint["completed_epoch"]:
            raise ValueError("Checkpoint history length is inconsistent.")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if self.scheduler is not None and "scheduler_state_dict" not in checkpoint:
            raise ValueError("Scheduler checkpoint state is missing; cannot resume safely.")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # The optimizer state is saved on CPU for portability; tensors must
        # match the current training device before the next optimization step.
        for state in self.optimizer.state.values():
            for name, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and name != "step":
                    state[name] = tensor.to(self.device)
        self.completed_epoch = int(checkpoint["completed_epoch"])
        self.history = checkpoint["history"]
        self.best_epoch = int(checkpoint["best_epoch"])
        self.best_validation = float(checkpoint["best_validation"])
        self.bad_epochs = int(checkpoint["bad_epochs"])
        self.shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
        self.train_mask_generator.set_state(checkpoint["train_mask_generator_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if self.device.type == "cuda":
            saved_cuda = checkpoint.get("torch_cuda_rng_states")
            if saved_cuda is None or len(saved_cuda) != torch.cuda.device_count():
                raise ValueError("GPU RNG state count differs; exact resume is unsupported.")
            torch.cuda.set_rng_state_all(saved_cuda)
        print(f"Resumed after epoch {self.completed_epoch} (best: {self.best_epoch}).")

    def _iterate(self, loader: DataLoader, *, training: bool) -> dict:
        """One pass with target-count-weighted epoch aggregation and live progress."""
        self.model.train(training)
        mask_generator = (
            self.train_mask_generator
            if training
            else torch.Generator(device="cpu").manual_seed(self.config.validation_mask_seed)
        )
        p_sum = 0.0
        i_sum = 0.0
        masked_count = 0
        positive_count = 0
        cell_count = 0
        batch_count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        stage = "Train" if training else "Valid"
        epoch = self.completed_epoch + 1

        # Disable animated bars in redirected logs and automated test runners.
        with tqdm(
            loader,
            desc=f"Epoch {epoch:03d}/{self.config.max_epochs:03d} {stage}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            disable=not sys.stderr.isatty(),
        ) as progress:
            with context:
                for batch in progress:
                    if batch.get("vocabulary_fingerprint") != self.metadata["vocabulary_fingerprint"]:
                        raise ValueError("Batch vocabulary fingerprint does not match training checkpoint.")
                    expression = batch["expression"].to(self.device)
                    gene_ids = batch["gene_ids"].to(self.device)
                    measured = batch["measured_mask"].to(self.device)
                    reconstruction = generate_reconstruction_mask(
                        measured,
                        mask_ratio=self.config.mask_ratio,
                        gene_ids=gene_ids,
                        generator=mask_generator,
                        ensure_visible=True,
                    )
                    outputs = self.model(
                        expression=expression,
                        gene_ids=gene_ids,
                        measured_mask=measured,
                        reconstruction_mask=reconstruction,
                    )
                    metrics = self.criterion(outputs)
                    loss = metrics["total_loss"]
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Non-finite loss encountered.")
                    if training:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.grad_clip_norm,
                            error_if_nonfinite=True,
                        )
                        if not torch.isfinite(norm).item():
                            raise FloatingPointError("Non-finite gradient norm.")
                        self.optimizer.step()
                    p_sum += metrics["presence_loss_sum"].detach().item()
                    i_sum += metrics["intensity_loss_sum"].detach().item()
                    masked_count += int(metrics["num_masked"].item())
                    positive_count += int(metrics["num_positive"].item())
                    cell_count += expression.shape[0]
                    batch_count += 1

                    # Show epoch-so-far loss, not the loss of a single batch.
                    if batch_count % 10 == 0 or batch_count == len(loader):
                        p_mean = p_sum / masked_count if masked_count else 0.0
                        i_mean = i_sum / positive_count if positive_count else 0.0
                        running_loss = (
                            self.criterion.lambda_presence * p_mean
                            + self.criterion.lambda_intensity * i_mean
                        )
                        progress.set_postfix(
                            loss=f"{running_loss:.4f}",
                            bce=f"{p_mean:.4f}",
                            huber=f"{i_mean:.4f}",
                            refresh=False,
                        )

        if masked_count == 0:
            raise ValueError("No measured reconstruction targets in this epoch.")
        presence = p_sum / masked_count
        intensity = i_sum / positive_count if positive_count else 0.0
        total = (
            self.criterion.lambda_presence * presence
            + self.criterion.lambda_intensity * intensity
        )
        if not math.isfinite(total):
            raise FloatingPointError("Non-finite epoch loss encountered.")
        return {
            "total_loss": total,
            "presence_loss": presence,
            "intensity_loss": intensity,
            "num_masked": masked_count,
            "num_positive": positive_count,
            "num_cells": cell_count,
            "num_batches": batch_count,
        }

    def evaluate_validation(self) -> dict:
        """Use identical validation masking and ordering on every call."""
        return self._iterate(self.validation_loader, training=False)

    def _checkpoint(self, kind: str) -> dict:
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "kind": kind,
            "metadata": self.metadata,
            "completed_epoch": self.completed_epoch,
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "model_state_dict": self.model.state_dict(),
        }
        if kind == "last":
            if self.scheduler is not None:
                payload["scheduler_state_dict"] = self.scheduler.state_dict()
            payload.update({
                "optimizer_state_dict": self.optimizer.state_dict(),
                "shuffle_generator_state": self.shuffle_generator.get_state(),
                "train_mask_generator_state": self.train_mask_generator.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
                ),
                "device_type": self.device.type,
            })
        return payload

    def fit(self) -> list[dict]:
        """Train up to max_epochs; stop on non-improving validation reconstruction."""
        if self.bad_epochs >= self.config.patience:
            print("Previous run already reached early stopping; no extra epochs will be trained.")
            return self.history
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.completed_epoch + 1, self.config.max_epochs + 1):
            train_stats = self._iterate(self.train_loader, training=True)
            val_stats = self.evaluate_validation()
            metric = val_stats["total_loss"]
            epoch_learning_rate = float(self.optimizer.param_groups[0]["lr"])
            improved = metric < self.best_validation - self.config.min_delta
            if improved:
                self.best_validation = metric
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            # The scheduler sees the exact aggregate validation loss.
            # Save the post-step state so interrupted and continuous runs agree.
            if self.scheduler is not None:
                self.scheduler.step(metric)
            next_learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.completed_epoch = epoch
            self.history.append({
                "epoch": epoch,
                "train": train_stats,
                "validation": val_stats,
                "best_epoch": self.best_epoch,
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": next_learning_rate,
            })
            if improved:
                _atomic_torch_save(self._checkpoint("best"), self.best_path)
            _atomic_torch_save(self._checkpoint("last"), self.last_path)
            print(
                f"Epoch {epoch:03d} | train={train_stats['total_loss']:.6f} "
                f"val={metric:.6f} | best={self.best_epoch:03d} "
                f"| lr={epoch_learning_rate:.2e} -> {next_learning_rate:.2e} "
                f"| masked={val_stats['num_masked']:,} "
                f"positive={val_stats['num_positive']:,}"
            )
            if self.bad_epochs >= self.config.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        return self.history


def load_best_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_metadata: dict,
) -> dict:
    """Load the best model only after strict provenance verification."""
    saved = _safe_torch_load(Path(checkpoint_path))
    if (
        saved.get("format_version") != CHECKPOINT_VERSION
        or saved.get("kind") != "best"
        or saved.get("metadata") != expected_metadata
    ):
        raise ValueError("Best checkpoint provenance mismatch.")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return saved
