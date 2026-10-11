"""Balanced joint PCA/UMAP visualization of four frozen SG-MAE embeddings.

From the project root:
  python -m scripts.v1.plot_embeddings --input-dir runs/sgmae_cell_embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


SAMPLES = (
    ("shaml", "ShamL", "o", "#0072B2"),
    ("shamr", "ShamR", "s", "#009E73"),
    ("hour12r", "12h Injury", "^", "#E69F00"),
    ("day2l", "Day 2 Injury", "D", "#D55E00"),
)


def _read_data(root: Path):
    """Return sample-local arrays and metadata with explicit provenance checks."""
    root_manifest = json.loads((root / "extraction_manifest.json").read_text(encoding="utf-8"))
    if root_manifest.get("encoding_protocol") != "full_measured_visible_no_mask":
        raise ValueError("Unknown extraction protocol; do not pool incompatible embeddings.")
    data = {}
    dim = None
    for sample, _, _, _ in SAMPLES:
        folder = root / sample
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("sample") != sample
            or manifest.get("checkpoint_sha256") != root_manifest["checkpoint_sha256"]
            or manifest.get("vocabulary_fingerprint") != root_manifest["vocabulary_fingerprint"]
            or manifest.get("split_fingerprint") != root_manifest["split_fingerprint"]
            or manifest.get("extraction_protocol") != root_manifest["encoding_protocol"]
        ):
            raise ValueError(f"Embedding provenance mismatch: {sample}")
        vec = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        meta = pd.read_csv(folder / "cells.csv", keep_default_na=False)
        if vec.ndim != 2 or vec.shape[0] < 2 or vec.shape[0] != len(meta):
            raise ValueError(f"Invalid embedding/cell metadata pairing: {sample}")
        if int(manifest["cells"]) != len(meta) or int(manifest["cell_dim"]) != vec.shape[1]:
            raise ValueError(f"Manifest dimensions mismatch: {sample}")
        if dim is None:
            dim = vec.shape[1]
        elif vec.shape[1] != dim:
            raise ValueError("Embedding dimensions differ among slices.")
        if not (meta["sample"] == sample).all() or not meta["cell_id"].is_unique:
            raise ValueError(f"Invalid sample labels or repeated Cell IDs in {sample}")
        if not np.isfinite(vec).all():
            raise ValueError(f"Nonfinite embeddings for sample {sample}")
        data[sample] = {"embedding": vec, "metadata": meta}
    return root_manifest, data


def select_balanced_sample(data: dict, *, per_sample: int, seed: int):
    """Draw equal numbers of distinct cells per slice, without replacement."""
    if not isinstance(per_sample, int) or per_sample < 2:
        raise ValueError("per_sample must be at least 2.")
    rng = np.random.default_rng(seed)
    smallest = min(len(data[key]["metadata"]) for key, *_ in SAMPLES)
    actual = min(per_sample, smallest)
    vectors = []
    frames = []
    for sample, *_ in SAMPLES:
        subset = np.sort(rng.choice(len(data[sample]["metadata"]), actual, replace=False))
        vectors.append(np.asarray(data[sample]["embedding"][subset], dtype=np.float32))
        frames.append(data[sample]["metadata"].iloc[subset].reset_index(drop=True))
    return np.concatenate(vectors, axis=0), pd.concat(frames, ignore_index=True), actual


def project_embeddings(embedding: np.ndarray, *, seed=42, n_neighbors=30, min_dist=0.3, skip_umap=False):
    """Fit one PCA and, when requested, one UMAP on all selected slices."""
    if embedding.ndim != 2 or embedding.shape[0] < 5 or embedding.shape[1] < 2:
        raise ValueError("Require at least five cells and two embedding dimensions.")
    if not np.isfinite(embedding).all():
        raise ValueError("Embeddings contain nonfinite coordinates.")
    pca = PCA(n_components=2, svd_solver="full")
    pca_xy = pca.fit_transform(embedding).astype(np.float32)
    umap_xy = None
    if not skip_umap:
        try:
            from umap import UMAP
        except ImportError as error:
            raise ImportError("UMAP visualization requires 'pip install umap-learn'.") from error
        if n_neighbors < 2 or n_neighbors >= embedding.shape[0]:
            raise ValueError("n_neighbors must be >= 2 and less than selected cells.")
        if not 0 <= min_dist <= 1:
            raise ValueError("min_dist must be in [0, 1].")
        umap_xy = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
            n_jobs=1,
            low_memory=True,
        ).fit_transform(embedding).astype(np.float32)
    return pca_xy, umap_xy, pca.explained_variance_ratio_.tolist()


def _draw_by_slice(frame, x_col, y_col, output: Path, *, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    for sample, caption, marker, color in SAMPLES:
        subset = frame[frame["sample"] == sample]
        ax.scatter(
            subset[x_col], subset[y_col], s=6, alpha=0.38,
            marker=marker, c=color, label=f"{caption} (n={len(subset):,})",
            linewidths=0, rasterized=True,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.15, linestyle=":")
    ax.legend(loc="best", frameon=False, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_qc(frame, x_col, y_col, output: Path, column: str, caption: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    values = frame[column].to_numpy(dtype=np.float64)
    if column == "total_counts":
        values = np.log1p(values)
    scatter = ax.scatter(
        frame[x_col], frame[y_col], c=values,
        cmap="viridis", s=6, alpha=0.65, linewidths=0, rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label=caption)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"SG-MAE | UMAP colored by {caption}")
    ax.grid(alpha=0.1, linestyle=":")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_facets(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (sample, caption, marker, color) in zip(axes.flat, SAMPLES):
        subset = frame[frame["sample"] == sample]
        ax.scatter(subset["umap1"], subset["umap2"], color=color, s=7, alpha=0.45, linewidths=0, rasterized=True)
        ax.set(title=f"{caption} | n={len(subset):,}", xlabel="UMAP 1", ylabel="UMAP 2")
        ax.grid(alpha=0.1, linestyle=":")
    fig.suptitle("SG-MAE V1.1 | Shared UMAP coordinates, displayed by slice", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="SG-MAE V1.1 balanced pooled PCA/UMAP plots")
    parser.add_argument("--input-dir", type=Path, default=root / "runs/v1/representations/embeddings")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--per-sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_dir or args.input_dir / "plots"
    if out.exists():
        raise FileExistsError(f"Plot folder already exists: {out}. Choose a new --output-dir.")
    source_manifest, data = _read_data(args.input_dir)
    embeddings, selected_meta, actual = select_balanced_sample(data, per_sample=args.per_sample, seed=args.seed)
    print(f"Pooled balanced projection: {len(embeddings):,} cells ({actual:,}/slice)")
    pca_xy, umap_xy, explained = project_embeddings(
        embeddings, seed=args.seed, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, skip_umap=args.skip_umap,
    )
    selected_meta["pca1"] = pca_xy[:, 0]
    selected_meta["pca2"] = pca_xy[:, 1]
    if umap_xy is not None:
        selected_meta["umap1"] = umap_xy[:, 0]
        selected_meta["umap2"] = umap_xy[:, 1]
    out.mkdir(parents=True, exist_ok=False)
    selected_meta.to_csv(out / "projection_cells.csv", index=False)
    _draw_by_slice(
        selected_meta, "pca1", "pca2", out / "pooled_pca_by_slice.png",
        title="SG-MAE V1.1 | Joint PCA by kidney slice",
        xlabel=f"PC1 ({100*explained[0]:.1f}% variance)",
        ylabel=f"PC2 ({100*explained[1]:.1f}% variance)",
    )
    if umap_xy is not None:
        _draw_by_slice(
            selected_meta, "umap1", "umap2", out / "pooled_umap_by_slice.png",
            title="SG-MAE V1.1 | Joint UMAP by kidney slice", xlabel="UMAP 1", ylabel="UMAP 2",
        )
        _draw_qc(selected_meta, "umap1", "umap2", out / "pooled_umap_by_counts.png", "total_counts", "log1p(total UMI counts)")
        _draw_qc(selected_meta, "umap1", "umap2", out / "pooled_umap_by_detected_genes.png", "detected_genes", "Detected genes")
        _draw_facets(selected_meta, out / "umap_by_slice_facets.png")
    details = {
        "projection_schema_version": 1,
        "embedding_checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "embedding_protocol": source_manifest["encoding_protocol"],
        "balance": "equal random subsample without replacement per slice",
        "per_slice_cells": int(actual),
        "projection_total_cells": int(len(embeddings)),
        "sample_seed": int(args.seed),
        "pca_explained_variance_ratio": explained,
        "umap": None if umap_xy is None else {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "seed": args.seed, "metric": "euclidean"},
        "important": "Joint visualization only; separation by slice cannot distinguish biological condition from sample/batch/quality confounding.",
    }
    with (out / "projection_manifest.json").open("x", encoding="utf-8") as file:
        json.dump(details, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(f"Saved plots and projection coordinates: {out}")
    return details


if __name__ == "__main__":
    main()
