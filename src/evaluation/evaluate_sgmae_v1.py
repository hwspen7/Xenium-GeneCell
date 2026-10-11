"""Read-only, masked-reconstruction evaluation for frozen SG-MAE V1.0.

This module never trains, modifies model weights, or changes a spatial split.
Scores are calculated only for measured, deliberately hidden gene targets.
The frequency/intensity baseline is fitted from ShamL TRAIN cells only.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from src.data.masking import generate_reconstruction_mask
from src.dataset import SGMAECellDataset, collate_sgmae_batch
from src.models.gene_aware_mae import GeneAwareMAE
from src.training.losses import MaskedReconstructionLoss


EVALUATION_SCHEMA_VERSION = 2


def sha256_file(path: str | Path) -> str:
    """Hash source bytes to bind evaluation to the training H5 file."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_frozen_best(
    checkpoint_path: str | Path,
    *,
    vocabulary_fingerprint: str,
    split_fingerprint: str,
    sham_h5_sha256: str,
    vocabulary_size: int,
    device: str | torch.device = "cpu",
) -> tuple[GeneAwareMAE, dict, dict]:
    """Load exactly one best.pt with strict experimental provenance checks."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported SG-MAE V1 checkpoint format.")
    if checkpoint.get("kind") != "best":
        raise ValueError("Evaluation requires best.pt, not last.pt.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("sample_name") != "ShamL":
        raise ValueError("Checkpoint was not trained on ShamL.")
    expected = {
        "vocabulary_fingerprint": vocabulary_fingerprint,
        "split_fingerprint": split_fingerprint,
        "data_sha256": sham_h5_sha256,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Checkpoint {field} does not match frozen experiment.")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("vocab_size") != vocabulary_size:
        raise ValueError("Checkpoint vocabulary size differs from current vocabulary.")
    config = metadata.get("trainer_config")
    loss_config = metadata.get("loss_config")
    if not isinstance(config, dict) or not isinstance(loss_config, dict):
        raise ValueError("Checkpoint is missing trainer/loss configuration.")
    for key in ("batch_size", "mask_ratio", "validation_mask_seed"):
        if key not in config:
            raise ValueError(f"Checkpoint is missing {key}.")
    if (
        type(config["batch_size"]) is not int
        or config["batch_size"] < 1
        or type(config["validation_mask_seed"]) is not int
        or config["validation_mask_seed"] < 0
        or not (0.0 < float(config["mask_ratio"]) <= 1.0)
    ):
        raise ValueError("Checkpoint evaluation protocol is invalid.")
    if checkpoint.get("best_epoch", 0) < 1 or not math.isfinite(
        float(checkpoint.get("best_validation", float("nan")))
    ):
        raise ValueError("Checkpoint best epoch or validation loss is invalid.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model weights.")

    model = GeneAwareMAE(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model, checkpoint, metadata


def fit_train_only_baseline(dataset: SGMAECellDataset, train_indices) -> dict:
    """Estimate gene-wise occurrence probability and positive log1p intensity.

    Laplace smoothing is used only for presence; zero-observation genes have
    positive-intensity default 0 and are identified via their positive counts.
    """
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("A nonempty ShamL train index vector is required.")
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError("Invalid ShamL training indices.")
    matrix = dataset.expression[indices].tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Invalid counts in training data.")
    if (matrix.data == 0).any():
        raise ValueError("Sparse matrix contains explicit zeros after cleanup.")
    n_cells = len(indices)
    positive_count = np.asarray(matrix.getnnz(axis=0), dtype=np.int64).reshape(-1)
    prevalence = (positive_count.astype(np.float64) + 1.0) / (n_cells + 2.0)
    log_counts = matrix.copy()
    log_counts.data = np.log1p(log_counts.data.astype(np.float64))
    log_sums = np.asarray(log_counts.sum(axis=0), dtype=np.float64).reshape(-1)
    intensity = np.divide(
        log_sums,
        positive_count,
        out=np.zeros_like(log_sums),
        where=positive_count > 0,
    )
    return {
        "gene_ids": dataset.gene_ids.numpy().copy(),
        "prevalence": prevalence,
        "positive_log_intensity": intensity,
        "positive_counts": positive_count,
        "num_train_cells": n_cells,
    }


def _baseline_by_id(baseline: dict, vocabulary_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resolve training-panel estimates by global Gene ID, not by column."""
    ids = np.asarray(baseline["gene_ids"], dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("Baseline contains duplicate gene IDs.")
    if ids.size == 0 or ids.min() < 0 or ids.max() >= vocabulary_size:
        raise ValueError("Baseline Gene ID outside vocabulary.")
    prevalence = np.full(vocabulary_size, np.nan, dtype=np.float64)
    intensity = np.full(vocabulary_size, np.nan, dtype=np.float64)
    prevalence[ids] = baseline["prevalence"]
    intensity[ids] = baseline["positive_log_intensity"]
    return prevalence, intensity


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return AP only when both positive and negative examples exist."""
    if np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, scores))


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predicted = scores >= 0.5
    return {
        "auroc": _safe_auc(labels, scores),
        "auprc": _safe_ap(labels, scores),
        "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, predicted, zero_division=0)),
        "brier": float(np.mean(np.square(scores - labels))),
        "positive_prevalence": float(labels.mean()),
    }


def _mean_or_none(numbers: list[float]) -> float | None:
    return float(np.mean(numbers)) if numbers else None


def _safe_pearson(true: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return undefined for fewer than three observations or constant arrays."""
    x = np.asarray(true, dtype=np.float64)
    y = np.asarray(predicted, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite values in correlation inputs.")
    x_center = x - x.mean()
    y_center = y - y.mean()
    sx = float(np.dot(x_center, x_center))
    sy = float(np.dot(y_center, y_center))
    if sx <= 1e-24 or sy <= 1e-24:
        return None
    coefficient = float(np.dot(x_center, y_center) / math.sqrt(sx * sy))
    return max(-1.0, min(1.0, coefficient))


def _safe_spearman(true: np.ndarray, predicted: np.ndarray) -> float | None:
    """Pearson correlation of average ranks, with explicit tie handling."""
    x = np.asarray(true, dtype=np.float64)
    y = np.asarray(predicted, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return None
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return _safe_pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def _correlation_pair(true: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "pearson": _safe_pearson(true, predicted),
        "spearman": _safe_spearman(true, predicted),
    }


def _sampled_pooled_correlation(
    true: np.ndarray,
    predicted: np.ndarray,
    *,
    max_spearman_targets: int = 100000,
    seed: int = 20261010,
) -> dict:
    """Use exact pooled Pearson and a bounded, deterministic pooled Spearman."""
    x = np.asarray(true, dtype=np.float64)
    y = np.asarray(predicted, dtype=np.float64)
    n = x.size
    if n > max_spearman_targets:
        sampled = np.random.default_rng(seed).choice(n, max_spearman_targets, replace=False)
        x_rank = x[sampled]
        y_rank = y[sampled]
    else:
        x_rank = x
        y_rank = y
    return {
        "pearson": _safe_pearson(x, y),
        "spearman": _safe_spearman(x_rank, y_rank),
        "spearman_sampled_targets": int(x_rank.size),
        "pearson_targets": int(n),
        "spearman_exact": bool(n <= max_spearman_targets),
    }


def _regression_error(true: np.ndarray, predicted: np.ndarray) -> dict:
    if true.size == 0:
        return {"mae": None, "rmse": None}
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _macro_summary(per_gene: list[dict], prefix: str) -> dict:
    values = {
        metric: [float(row[f"{prefix}_{metric}"]) for row in per_gene
                 if row[f"{prefix}_{metric}"] is not None]
        for metric in ("pearson", "spearman")
    }
    return {
        "macro_gene_pearson": _mean_or_none(values["pearson"]),
        "macro_gene_spearman": _mean_or_none(values["spearman"]),
        "genes_with_pearson": len(values["pearson"]),
        "genes_with_spearman": len(values["spearman"]),
    }


def _huber_sum(prediction: np.ndarray, target: np.ndarray, beta: float) -> float:
    error = np.abs(prediction - target)
    return float(np.where(error < beta, 0.5 * error * error / beta, error - 0.5 * beta).sum())


def summarize_predictions(
    *,
    gene_ids: np.ndarray,
    counts: np.ndarray,
    logits: np.ndarray,
    intensity: np.ndarray,
    baseline_presence: np.ndarray,
    baseline_intensity: np.ndarray,
    huber_beta: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Calculate pooled and gene-macro classification and positive intensity metrics."""
    arrays = [np.asarray(arr) for arr in (
        gene_ids, counts, logits, intensity, baseline_presence, baseline_intensity
    )]
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("At least one masked measured target is required.")
    if any(arr.shape != arrays[0].shape or arr.ndim != 1 for arr in arrays):
        raise ValueError("Prediction/label shapes differ.")
    gid, raw, raw_logits, predicted_log, base_prob, base_log = arrays
    if not np.issubdtype(gid.dtype, np.integer):
        raise TypeError("Gene IDs must be integers, not fractional values.")
    if (
        isinstance(huber_beta, (bool, np.bool_))
        or not isinstance(huber_beta, (int, float, np.integer, np.floating))
        or not math.isfinite(float(huber_beta))
        or huber_beta <= 0
    ):
        raise ValueError("huber_beta must be finite and positive.")
    if (
        not all(np.isfinite(arr).all() for arr in arrays[1:])
        or np.any(raw < 0) or np.any(predicted_log < 0)
        or np.any(base_prob <= 0) or np.any(base_prob >= 1)
        or np.any(base_log < 0)
        or np.any(gid < 0)
    ):
        raise ValueError("Invalid predictions or targets.")
    raw = raw.astype(np.float64)
    logits = raw_logits.astype(np.float64)
    # Stable sigmoid avoids numerical overflow for large logits.
    probabilities = np.exp(-np.logaddexp(0.0, -logits))
    base_prob = base_prob.astype(np.float64)
    labels = raw > 0
    targets = np.log1p(raw[labels])
    pos_predictions = predicted_log[labels].astype(np.float64)
    pos_baseline = base_log[labels].astype(np.float64)
    bce_model = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    bce_baseline = float(np.mean(-labels.astype(np.float64) * np.log(base_prob) - (~labels).astype(np.float64) * np.log1p(-base_prob)))
    positive_count = int(labels.sum())
    mae = float(np.mean(np.abs(pos_predictions - targets))) if positive_count else None
    rmse = float(np.sqrt(np.mean((pos_predictions - targets) ** 2))) if positive_count else None
    base_mae = float(np.mean(np.abs(pos_baseline - targets))) if positive_count else None
    base_rmse = float(np.sqrt(np.mean((pos_baseline - targets) ** 2))) if positive_count else None
    huber_model = _huber_sum(pos_predictions, targets, huber_beta) / positive_count if positive_count else 0.0
    huber_baseline = _huber_sum(pos_baseline, targets, huber_beta) / positive_count if positive_count else 0.0

    all_targets_log = np.log1p(raw)
    # This is an explicitly defined log-scale reconstruction heuristic, not E[raw count].
    full_model_log = probabilities * predicted_log.astype(np.float64)
    full_baseline_log = base_prob * base_log.astype(np.float64)
    positive_pooled_model = _sampled_pooled_correlation(targets, pos_predictions)
    positive_pooled_baseline = _sampled_pooled_correlation(targets, pos_baseline)
    full_pooled_model = _sampled_pooled_correlation(all_targets_log, full_model_log)
    full_pooled_baseline = _sampled_pooled_correlation(all_targets_log, full_baseline_log)
    model_cls = _classification_metrics(labels, probabilities)
    baseline_cls = _classification_metrics(labels, base_prob)
    rows = []
    macro_model_ap = []
    macro_baseline_ap = []
    macro_model_auc = []
    macro_baseline_auc = []
    # Group once rather than scan all N targets separately for each gene.
    order = np.argsort(gid, kind="stable")
    sorted_ids = gid[order]
    unique_ids, group_start, group_size = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    for gene, start, size in zip(unique_ids, group_start, group_size):
        group = order[start : start + size]
        gene_labels = labels[group]
        gene_model = probabilities[group]
        gene_baseline = base_prob[group]
        gene_ap = _safe_ap(gene_labels, gene_model)
        base_ap = _safe_ap(gene_labels, gene_baseline)
        gene_auc = _safe_auc(gene_labels, gene_model)
        base_auc = _safe_auc(gene_labels, gene_baseline)
        if gene_ap is not None:
            macro_model_ap.append(gene_ap)
        if base_ap is not None:
            macro_baseline_ap.append(base_ap)
        if gene_auc is not None:
            macro_model_auc.append(gene_auc)
        if base_auc is not None:
            macro_baseline_auc.append(base_auc)
        positive_group = group[gene_labels]
        gene_targets = np.log1p(raw[positive_group])
        positive_model_corr = _correlation_pair(gene_targets, predicted_log[positive_group])
        positive_baseline_corr = _correlation_pair(gene_targets, base_log[positive_group])
        full_model_corr = _correlation_pair(all_targets_log[group], full_model_log[group])
        full_baseline_corr = _correlation_pair(all_targets_log[group], full_baseline_log[group])
        rows.append({
            "gene_id": int(gene),
            "masked_targets": int(size),
            "positive_targets": int(gene_labels.sum()),
            "negative_targets": int(size - gene_labels.sum()),
            "positive_prevalence": float(gene_labels.mean()),
            "model_auroc": gene_auc,
            "baseline_auroc": base_auc,
            "model_auprc": gene_ap,
            "baseline_auprc": base_ap,
            "model_f1_at_0_5": float(f1_score(gene_labels, gene_model >= 0.5, zero_division=0)),
            "baseline_f1_at_0_5": float(f1_score(gene_labels, gene_baseline >= 0.5, zero_division=0)),
            "model_positive_mae_log1p": float(np.mean(np.abs(predicted_log[positive_group] - gene_targets))) if gene_targets.size else None,
            "baseline_positive_mae_log1p": float(np.mean(np.abs(base_log[positive_group] - gene_targets))) if gene_targets.size else None,
            "model_positive_pearson": positive_model_corr["pearson"],
            "model_positive_spearman": positive_model_corr["spearman"],
            "baseline_positive_pearson": positive_baseline_corr["pearson"],
            "baseline_positive_spearman": positive_baseline_corr["spearman"],
            "model_full_pearson": full_model_corr["pearson"],
            "model_full_spearman": full_model_corr["spearman"],
            "baseline_full_pearson": full_baseline_corr["pearson"],
            "baseline_full_spearman": full_baseline_corr["spearman"],
            "model_full_mae_log1p": _regression_error(all_targets_log[group], full_model_log[group])["mae"],
            "baseline_full_mae_log1p": _regression_error(all_targets_log[group], full_baseline_log[group])["mae"],
        })
    return {
        "masked_targets": int(labels.size),
        "positive_targets": positive_count,
        "model": {
            "presence": {
                **model_cls,
                "bce": bce_model,
                "macro_gene_auprc": _mean_or_none(macro_model_ap),
                "macro_gene_auroc": _mean_or_none(macro_model_auc),
                "genes_with_auprc": len(macro_model_ap),
                "genes_with_auroc": len(macro_model_auc),
            },
            "positive_intensity_log1p": {
                "mae": mae, "rmse": rmse, "huber": huber_model,
                **positive_pooled_model,
                **_macro_summary(rows, "model_positive"),
            },
            "full_reconstruction_log1p": {
                **_regression_error(all_targets_log, full_model_log),
                **full_pooled_model,
                **_macro_summary(rows, "model_full"),
                "prediction_rule": "presence_probability_times_positive_log1p_intensity",
            },
        },
        "baseline": {
            "presence": {
                **baseline_cls,
                "bce": bce_baseline,
                "macro_gene_auprc": _mean_or_none(macro_baseline_ap),
                "macro_gene_auroc": _mean_or_none(macro_baseline_auc),
                "genes_with_auprc": len(macro_baseline_ap),
                "genes_with_auroc": len(macro_baseline_auc),
            },
            "positive_intensity_log1p": {
                "mae": base_mae, "rmse": base_rmse, "huber": huber_baseline,
                **positive_pooled_baseline,
                **_macro_summary(rows, "baseline_positive"),
            },
            "full_reconstruction_log1p": {
                **_regression_error(all_targets_log, full_baseline_log),
                **full_pooled_baseline,
                **_macro_summary(rows, "baseline_full"),
                "prediction_rule": "presence_probability_times_positive_log1p_intensity",
            },
        },
    }, rows


def evaluate_frozen_reconstruction(
    *,
    model: GeneAwareMAE,
    dataset: SGMAECellDataset,
    cell_indices,
    baseline: dict,
    config: dict,
    loss_config: dict,
    device: str | torch.device = "cpu",
    show_progress: bool = True,
    scatter_sample_size: int = 0,
) -> tuple[dict, list[dict]]:
    """Evaluate a single frozen model on a fixed, measured-only mask protocol.

    To match the trainer's validation loss exactly, use the same 64-cell
    batching, ascending ShamL validation indices, and validation_mask_seed.
    """
    if not isinstance(dataset, SGMAECellDataset):
        raise TypeError("SGMAECellDataset is required.")
    indices = np.asarray(cell_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Evaluation must contain at least one cell.")
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError("Evaluation indices are invalid.")
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_sgmae_batch,
    )
    device = torch.device(device)
    model.to(device).eval()
    criterion = MaskedReconstructionLoss(**loss_config)
    rng = torch.Generator(device="cpu").manual_seed(int(config["validation_mask_seed"]))
    prevalence, mean_log = _baseline_by_id(baseline, model.vocab_size)
    mask_hasher = hashlib.sha256()
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in (
        "gene_ids", "counts", "logits", "intensity", "baseline_presence", "baseline_intensity"
    )}
    total_masked = total_positive = 0
    p_sum = i_sum = 0.0
    cells = 0

    if show_progress:
        from tqdm.auto import tqdm
        iterable = tqdm(loader, desc="Evaluate masked genes", unit="batch")
    else:
        iterable = loader
    with torch.inference_mode():
        for batch in iterable:
            if batch.get("vocabulary_fingerprint") != dataset.vocabulary_fingerprint:
                raise ValueError("DataLoader vocabulary fingerprint mismatch.")
            expression = batch["expression"].to(device)
            ids = batch["gene_ids"].to(device)
            measured = batch["measured_mask"].to(device)
            mask = generate_reconstruction_mask(
                measured,
                mask_ratio=float(config["mask_ratio"]),
                gene_ids=ids,
                generator=rng,
                ensure_visible=True,
            )
            mask_hasher.update(mask.detach().cpu().numpy().tobytes())
            outputs = model(
                expression=expression,
                gene_ids=ids,
                measured_mask=measured,
                reconstruction_mask=mask,
            )
            losses = criterion(outputs)
            supervised = outputs["target_reconstruction_mask"]
            ids_for_targets = ids
            selected_ids = ids_for_targets[supervised].detach().cpu().numpy().astype(np.int64)
            selected_counts = outputs["target_counts"][supervised].detach().cpu().numpy()
            selected_logits = outputs["presence_logits"][supervised].detach().cpu().numpy()
            selected_intensity = outputs["positive_intensity"][supervised].detach().cpu().numpy()
            if selected_ids.size and (np.isnan(prevalence[selected_ids]).any() or np.isnan(mean_log[selected_ids]).any()):
                raise ValueError("Evaluation contains genes absent from ShamL training baseline.")
            values = {
                "gene_ids": selected_ids,
                "counts": selected_counts,
                "logits": selected_logits,
                "intensity": selected_intensity,
                "baseline_presence": prevalence[selected_ids],
                "baseline_intensity": mean_log[selected_ids],
            }
            for name, array in values.items():
                chunks[name].append(array)
            total_masked += int(losses["num_masked"].item())
            total_positive += int(losses["num_positive"].item())
            p_sum += float(losses["presence_loss_sum"].item())
            i_sum += float(losses["intensity_loss_sum"].item())
            cells += expression.shape[0]

    if not total_masked:
        raise ValueError("Evaluation produced no masked measured targets.")
    packed = {name: np.concatenate(values) for name, values in chunks.items()}
    metrics, per_gene = summarize_predictions(**packed, huber_beta=criterion.huber_beta)
    if type(scatter_sample_size) is not int or scatter_sample_size < 0:
        raise ValueError("scatter_sample_size must be a nonnegative integer.")
    if scatter_sample_size > 0:
        # Draw a reproducible visualization subset without storing millions of points.
        sample_rng = np.random.default_rng(271828)
        sample_indices = sample_rng.choice(
            packed["counts"].size,
            min(scatter_sample_size, packed["counts"].size),
            replace=False,
        )
        sampled_counts = packed["counts"][sample_indices].astype(np.float64)
        sampled_probability = np.exp(-np.logaddexp(0.0, -packed["logits"][sample_indices].astype(np.float64)))
        metrics["plot_samples"] = {
            "target_log1p": np.log1p(sampled_counts).tolist(),
            "predicted_full_log1p": (sampled_probability * packed["intensity"][sample_indices]).tolist(),
            "predicted_positive_log1p": packed["intensity"][sample_indices].astype(np.float64).tolist(),
            "is_positive": (sampled_counts > 0).tolist(),
        }
    if total_masked != metrics["masked_targets"] or total_positive != metrics["positive_targets"]:
        raise AssertionError("Loss totals and prediction totals disagree.")
    p_mean = p_sum / total_masked
    i_mean = i_sum / total_positive if total_positive else 0.0
    weighted_loss = criterion.lambda_presence * p_mean + criterion.lambda_intensity * i_mean
    metrics.update({
        "num_cells": cells,
        "mask_sha256": mask_hasher.hexdigest(),
        "reconstruction_loss": {
            "total": weighted_loss,
            "presence_bce": p_mean,
            "positive_huber": i_mean,
            "baseline_total": criterion.lambda_presence * metrics["baseline"]["presence"]["bce"] + criterion.lambda_intensity * metrics["baseline"]["positive_intensity_log1p"]["huber"],
        },
    })
    return metrics, per_gene
