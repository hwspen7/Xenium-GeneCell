"""Quantitative diagnostics and matched expression baselines for frozen cell representations.

No clustering or cell-type prediction is performed here. Technical covariates
are descriptive, and neither UMAP separation nor reconstruction demonstrates
biological cell-type recognition.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA


def finite_pearson(x, y):
    """Return Pearson r or None when the coefficient is mathematically undefined."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Pearson requires at least two paired finite values.")
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return None
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def expression_transform(counts, *, method="library_normalized", target_sum=10000.0):
    """Transform cell-by-gene measured counts, preserving genuine zero counts."""
    x = np.asarray(counts, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Counts must be a finite, nonnegative cell-by-gene matrix.")
    if method == "raw_log1p":
        return np.log1p(x).astype(np.float32)
    if method != "library_normalized":
        raise ValueError("Unknown expression transform.")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be strictly positive.")
    totals = x.sum(axis=1, dtype=np.float64)
    scales = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(x * scales[:, None]).astype(np.float32)


def aligned_gene_columns(dataset, reference_gene_ids):
    """Return column indices in stable Gene-ID order, never by gene symbol."""
    actual = np.asarray(dataset.gene_ids.detach().cpu().numpy(), dtype=np.int64)
    reference = np.asarray(reference_gene_ids, dtype=np.int64)
    if actual.ndim != 1 or reference.ndim != 1 or len(actual) != len(reference):
        raise ValueError("Different measured-gene panel sizes.")
    if len(np.unique(actual)) != len(actual) or len(np.unique(reference)) != len(reference):
        raise ValueError("Duplicate Gene IDs.")
    mapping = {int(g): idx for idx, g in enumerate(actual)}
    if set(mapping) != set(reference.tolist()):
        raise ValueError("Different measured Gene IDs.")
    return np.asarray([mapping[int(g)] for g in reference], dtype=np.int64)


def matched_cell_rows(dataset, metadata):
    """Validate row_index and cell_id jointly before using expression counts."""
    required = {"row_index", "cell_id", "sample"}
    if not required.issubset(metadata.columns):
        raise ValueError("Missing fields required for cell-level count alignment.")
    indices = metadata["row_index"].to_numpy()
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
        raise ValueError("row_index must be nonempty integers.")
    if indices.min() < 0 or indices.max() >= len(dataset) or len(np.unique(indices)) != len(indices):
        raise ValueError("Invalid or duplicated row indices.")
    names = np.asarray(dataset.cell_ids, dtype=str)[indices]
    expected = metadata["cell_id"].astype(str).to_numpy()
    if not np.array_equal(names, expected):
        raise ValueError("Embedding metadata Cell IDs do not match source H5 rows.")
    return indices.astype(np.int64, copy=False)


def embedding_reference_pc1(reference):
    """Fit an unscaled PCA axis to a non-external reference representation."""
    vec = np.asarray(reference, dtype=np.float32)
    if vec.ndim != 2 or min(vec.shape) < 2 or not np.isfinite(vec).all():
        raise ValueError("Invalid reference embeddings.")
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(vec)
    return pca.mean_.astype(np.float64), pca.components_[0].astype(np.float64)


def embedding_diagnostics(embedding, total_counts, detected_genes, *, pc1_reference):
    """Compute collapse and technical-covariate diagnostics on all supplied cells."""
    vec = np.asarray(embedding, dtype=np.float64)
    counts = np.asarray(total_counts, dtype=np.float64).reshape(-1)
    detected = np.asarray(detected_genes, dtype=np.float64).reshape(-1)
    if vec.ndim != 2 or vec.shape[0] < 3 or vec.shape[1] < 2:
        raise ValueError("Embeddings require at least three cells and two dimensions.")
    if len(counts) != len(vec) or len(detected) != len(vec):
        raise ValueError("Embedding and QC row counts differ.")
    if not np.isfinite(vec).all() or not np.isfinite(counts).all() or not np.isfinite(detected).all():
        raise ValueError("Nonfinite embedding/QC values.")
    if (counts < 0).any() or (detected < 0).any():
        raise ValueError("Negative QC counts.")
    mean, axis = pc1_reference
    if len(mean) != vec.shape[1] or len(axis) != vec.shape[1]:
        raise ValueError("Reference principal axis and embeddings differ in width.")
    pc1 = (vec - mean) @ axis
    norms = np.linalg.norm(vec, axis=1)
    cov = np.cov(vec, rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0)[::-1]
    variance = float(eig.sum())
    proportions = eig / variance if variance > 0 else np.zeros_like(eig)
    positive = proportions[proportions > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0
    per_dim_sd = np.sqrt(np.maximum(np.diag(cov), 0))
    return {
        "cells": int(len(vec)),
        "dimensions": int(vec.shape[1]),
        "median_l2_norm": float(np.median(norms)),
        "mean_l2_norm": float(norms.mean()),
        "median_dimension_std": float(np.median(per_dim_sd)),
        "near_constant_dimensions_abs_std_lt_1e-6": int((per_dim_sd < 1e-6).sum()),
        "effective_rank": effective_rank,
        "top1_variance_fraction": float(proportions[:1].sum()),
        "top10_variance_fraction": float(proportions[:10].sum()),
        "reference_pc1_r_log1p_counts": finite_pearson(pc1, np.log1p(counts)),
        "reference_pc1_r_detected_genes": finite_pearson(pc1, detected),
        "embedding_norm_r_log1p_counts": finite_pearson(norms, np.log1p(counts)),
        "embedding_norm_r_detected_genes": finite_pearson(norms, detected),
        "positive_cell_fraction": float(np.mean(counts > 0)),
    }


def fit_train_reference_pca(train_counts, *, feature_dim: int, seed: int, method="library_normalized"):
    """Fit expression baseline PCA exclusively on ShamL TRAIN cells."""
    x = expression_transform(train_counts, method=method)
    if x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("Baseline PCA requires at least three cells and two genes.")
    components = min(int(feature_dim), x.shape[1], x.shape[0] - 1)
    if components < 2:
        raise ValueError("Baseline PCA needs at least two components.")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(x)
    return pca


def project_common_2d(matrix, *, seed: int = 42, skip_umap=False, n_neighbors=30, min_dist=0.3):
    """Fit a shared projection on the selected cells, separately for each representation."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid representation matrix.")
    pca = PCA(n_components=2, svd_solver="full")
    coords = pca.fit_transform(x).astype(np.float32)
    umap_coords = None
    if not skip_umap:
        from umap import UMAP
        if n_neighbors < 2 or n_neighbors >= len(x) or not 0 <= min_dist <= 1:
            raise ValueError("Invalid UMAP parameters.")
        umap_coords = UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            random_state=seed, n_jobs=1, low_memory=True, metric="euclidean",
        ).fit_transform(x).astype(np.float32)
    return coords, umap_coords, pca.explained_variance_ratio_.tolist()
