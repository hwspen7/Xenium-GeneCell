"""Read-only SG-MAE cell-embedding extraction and aligned Xenium metadata.

Extraction uses every measured gene as visible input and masks no genes.
This representation protocol is distinct from masked-reconstruction evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


EMBEDDING_SCHEMA_VERSION = 1


def _resolve_extraction_device(requested_device: str | torch.device, model_device: torch.device) -> torch.device:
    """Match the actual model device, allowing unindexed 'cuda' to mean any selected CUDA GPU."""
    requested = torch.device(requested_device)
    actual = torch.device(model_device)
    if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
        raise ValueError(f"Model is on {actual}, but extraction requested {requested}.")
    return actual


def verify_gene_panel(reference_ids, candidate_ids) -> None:
    """Require exactly the same measured gene identities; column order may differ."""
    a = np.asarray(reference_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        raise ValueError("Samples must have identical nonempty measured gene panels.")
    if np.unique(a).size != a.size or np.unique(b).size != b.size:
        raise ValueError("Duplicate measured Gene IDs are not allowed.")
    if not np.array_equal(np.sort(a), np.sort(b)):
        raise ValueError("External slice measured Gene IDs differ from ShamL.")


def _decode_cell_id(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Invalid cell_id in Xenium metadata.")
    return value


def aligned_spatial_coordinates(cell_ids, parquet_path: str | Path) -> np.ndarray:
    """Align Xenium centroids to H5 barcode order by exact cell identity."""
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing Xenium cells.parquet: {parquet_path}")
    frame = pd.read_parquet(parquet_path, columns=["cell_id", "x_centroid", "y_centroid"])
    metadata_ids = [_decode_cell_id(value) for value in frame["cell_id"].tolist()]
    query_ids = list(cell_ids)
    if len(metadata_ids) != len(query_ids):
        raise ValueError("H5 and Parquet contain different numbers of cells.")
    if len(set(metadata_ids)) != len(metadata_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate cell IDs encountered during spatial alignment.")
    lookup = pd.Index(metadata_ids).get_indexer(query_ids)
    if np.any(lookup < 0):
        raise ValueError("H5 Cell IDs cannot all be matched to cells.parquet.")
    coords = frame[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)[lookup]
    if coords.shape != (len(query_ids), 2) or not np.isfinite(coords).all():
        raise ValueError("Nonfinite or invalid spatial centroids.")
    return coords.astype(np.float32)


def validate_cell_indices(indices, n_cells: int) -> np.ndarray:
    """Require unique ordered cell indices, preserving their specified order."""
    selected = np.asarray(indices)
    if selected.ndim != 1 or selected.size == 0 or selected.dtype.kind not in "iu":
        raise ValueError("Cell indices must be a nonempty integer vector.")
    selected = selected.astype(np.int64, copy=False)
    if selected.min() < 0 or selected.max() >= n_cells or np.unique(selected).size != selected.size:
        raise ValueError("Cell indices must be unique and within the dataset bounds.")
    return selected


def encode_cells(
    model: torch.nn.Module,
    dataset,
    cell_indices,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract cell representations and QC without gradient or input masking.

    Returns embedding float32 [N, D], total raw counts [N], detected genes [N].
    Batches are sliced from the sample's cell-by-gene CSR matrix to avoid
    per-cell Python dispatch and never densify the entire sample.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    indices = validate_cell_indices(cell_indices, len(dataset))
    if not sparse.isspmatrix_csr(dataset.expression):
        raise TypeError("Dataset expression must be CSR sparse counts.")
    if dataset.expression.shape[0] != len(dataset):
        raise ValueError("Expression rows and dataset length disagree.")
    ids = dataset.gene_ids
    measured = dataset.measured_mask
    if ids.ndim != 1 or measured.shape != ids.shape or ids.dtype != torch.long:
        raise ValueError("Invalid dataset gene IDs or measured mask.")
    if dataset.expression.shape[1] != ids.numel() or measured.dtype != torch.bool:
        raise ValueError("Gene columns or measured mask do not match expression.")
    if not measured.any().item():
        raise ValueError("No measured genes available for embedding extraction.")

    location = _resolve_extraction_device(device, next(model.parameters()).device)
    if model.training:
        raise ValueError("Model must be in eval mode for frozen extraction.")

    gene_ids = ids.to(location)
    one_target_id = gene_ids[measured.to(location)][:1]
    embeddings = None
    total_counts = np.empty(len(indices), dtype=np.float32)
    detected_genes = np.empty(len(indices), dtype=np.int32)

    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            rows = indices[start:stop]
            matrix = dataset.expression[rows]
            batch_counts = np.asarray(matrix.toarray(), dtype=np.float32)
            if not np.isfinite(batch_counts).all() or (batch_counts < 0).any():
                raise ValueError("Invalid expression counts for selected cells.")
            batch_size_actual = stop - start
            total_counts[start:stop] = batch_counts[:, measured.numpy()].sum(axis=1)
            detected_genes[start:stop] = np.count_nonzero(batch_counts[:, measured.numpy()] > 0, axis=1)
            counts = torch.from_numpy(batch_counts).to(location)
            present = measured.to(location).unsqueeze(0).expand(batch_size_actual, -1)
            result = model(
                expression=counts,
                gene_ids=gene_ids,
                measured_mask=present,
                reconstruction_mask=torch.zeros_like(present),
                target_gene_ids=one_target_id,
            )
            part = result["cell_embeddings"].detach().float().cpu().numpy()
            if part.ndim != 2 or part.shape[0] != batch_size_actual or not np.isfinite(part).all():
                raise ValueError("Unexpected nonfinite model Cell Embeddings.")
            if embeddings is None:
                embeddings = np.empty((len(indices), part.shape[1]), dtype=np.float32)
            if part.shape[1] != embeddings.shape[1]:
                raise ValueError("Cell embedding dimension changed between batches.")
            embeddings[start:stop] = part
    return embeddings, total_counts, detected_genes


def save_sample_embedding(
    sample_dir: str | Path,
    *,
    sample_key: str,
    dataset,
    indices,
    coordinates: np.ndarray,
    embeddings: np.ndarray,
    total_counts: np.ndarray,
    detected_genes: np.ndarray,
    provenance: dict,
) -> None:
    """Save row-aligned embeddings and metadata; never overwrite prior reports."""
    indices = validate_cell_indices(indices, len(dataset))
    n = len(indices)
    if embeddings.ndim != 2 or embeddings.shape[0] != n or embeddings.shape[1] == 0:
        raise ValueError("Embedding array does not match cell selection.")
    if coordinates.shape != (len(dataset), 2):
        raise ValueError("Spatial coordinates do not match all source cells.")
    if total_counts.shape != (n,) or detected_genes.shape != (n,):
        raise ValueError("QC arrays do not match selected cells.")
    if not np.isfinite(embeddings).all() or not np.isfinite(total_counts).all():
        raise ValueError("Embedding or QC values are not finite.")
    path = Path(sample_dir)
    path.mkdir(parents=True, exist_ok=False)
    np.save(path / "embeddings.npy", np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    with (path / "cells.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "row_index", "cell_id", "x_centroid", "y_centroid", "total_counts", "detected_genes"])
        for local_idx, original_idx in enumerate(indices):
            x, y = coordinates[original_idx]
            writer.writerow([
                sample_key,
                int(original_idx),
                dataset.cell_ids[int(original_idx)],
                float(x), float(y),
                float(total_counts[local_idx]), int(detected_genes[local_idx]),
            ])
    info = {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "sample": sample_key,
        "cells": int(n),
        "cell_dim": int(embeddings.shape[1]),
        "extraction_protocol": "full_measured_visible_no_mask",
        "cell_order": "embeddings.npy row i corresponds to cells.csv row i (after header)",
        **provenance,
    }
    with (path / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(info, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
