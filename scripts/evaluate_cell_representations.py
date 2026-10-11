"""Frozen SG-MAE representation QC versus a ShamL-train-fitted expression PCA baseline.

Run from the project root. This command never fits the SG-MAE encoder, and
never fits the expression baseline on ShamL validation or external slices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_cell_embeddings import SAMPLES, _read_data, select_balanced_sample
from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import load_spatial_split
from src.dataset import find_xenium_sample_file, load_sgmae_sample
from src.evaluation.cell_embeddings import sha256_file
from src.evaluation.representation_diagnostics import (
    aligned_gene_columns,
    embedding_diagnostics,
    embedding_reference_pc1,
    expression_transform,
    fit_train_reference_pca,
    matched_cell_rows,
    project_common_2d,
)


SAMPLE_PREFIX = {
    "shaml": "ShamL", "shamr": "ShamR", "hour12r": "Hour12R", "day2l": "Day2L"
}
PLOT_COLORS = {
    "shaml": "#0072B2", "shamr": "#009E73", "hour12r": "#E69F00", "day2l": "#D55E00"
}


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty report.")
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _choose_rows(data, *, projection_csv, per_sample, seed):
    """Use the already-plotted cells when specified; otherwise balanced sampling."""
    if projection_csv is None:
        vectors, meta, actual = select_balanced_sample(data, per_sample=per_sample, seed=seed)
        return vectors, meta, actual
    meta = pd.read_csv(projection_csv, keep_default_na=False)
    if not {"sample", "row_index", "cell_id"}.issubset(meta.columns):
        raise ValueError("Projection CSV must include sample, row_index, cell_id.")
    if len(meta) < 5 or not meta["sample"].isin(SAMPLE_PREFIX).all():
        raise ValueError("Unknown or insufficient projection samples.")
    if meta[["sample", "cell_id"]].duplicated().any():
        raise ValueError("Repeated sample/cell IDs in projection CSV.")
    selected = []
    meta_out = []
    group_size = None
    for sample, _, _, _ in SAMPLES:
        wanted = meta[meta["sample"] == sample].reset_index(drop=True)
        original = data[sample]["metadata"]
        if len(wanted) < 2:
            raise ValueError(f"No comparable selected cells for {sample}.")
        if group_size is None:
            group_size = len(wanted)
        elif len(wanted) != group_size:
            raise ValueError("Existing projection must select equal numbers per slice.")
        ids = pd.Index(original["cell_id"].astype(str))
        offsets = ids.get_indexer(wanted["cell_id"].astype(str))
        if (offsets < 0).any():
            raise ValueError(f"Selected Cell IDs missing in stored embeddings: {sample}.")
        matching_rows = original.iloc[offsets].reset_index(drop=True)
        if not np.array_equal(
            wanted["row_index"].to_numpy(), matching_rows["row_index"].to_numpy()
        ):
            raise ValueError(f"Cell ID / row index mismatch for {sample}.")
        selected.append(np.asarray(data[sample]["embedding"][offsets], dtype=np.float32))
        meta_out.append(matching_rows)
    return np.concatenate(selected), pd.concat(meta_out, ignore_index=True), group_size


def _plot_qc(rows, filename):
    names = [entry[1] for entry in SAMPLES]
    labels = [entry[0] for entry in SAMPLES]
    colors = [PLOT_COLORS[k] for k in labels]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    settings = (
        ("effective_rank", "Effective covariance rank", None),
        ("top1_variance_fraction", "Largest covariance eigenvalue fraction", None),
        ("reference_pc1_r_log1p_counts", "|Pearson r|: reference PC1 vs log1p(total counts)", "abs"),
        ("embedding_norm_r_log1p_counts", "|Pearson r|: embedding norm vs log1p(total counts)", "abs"),
    )
    for ax, (field, title, mode) in zip(axes.flat, settings):
        values = [r[field] for r in rows]
        values = [abs(v) if v is not None and mode == "abs" else v for v in values]
        ax.bar(names, [np.nan if v is None else v for v in values], color=colors, edgecolor="black", linewidth=0.45)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        if mode == "abs" or field == "top1_variance_fraction":
            ax.set_ylim(0, 1.02)
    fig.suptitle("SG-MAE | Cell representation numerical and technical QC", fontweight="bold")
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_matched_qc(rows, filename):
    """Compare technical associations on exactly matched cells, not on different sets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    x = np.arange(len(SAMPLES))
    names = [item[1] for item in SAMPLES]
    for ax, field, title in (
        (axes[0], "reference_pc1_r_log1p_counts", "|PC1 correlation with log1p(total counts)|"),
        (axes[1], "reference_pc1_r_detected_genes", "|PC1 correlation with detected genes|"),
    ):
        for offset, label, color in ((-0.19, "SG-MAE", "#0072B2"),
                                     (0.19, "Expression PCA", "#E69F00")):
            vals = []
            for sample, *_ in SAMPLES:
                entry = next(r for r in rows if r["sample"] == sample and r["representation"] == label)
                val = entry[field]
                vals.append(np.nan if val is None else abs(val))
            ax.bar(x + offset, vals, width=0.37, color=color, edgecolor="black",
                   linewidth=0.45, label=label)
        ax.set_xticks(x, names)
        ax.tick_params(axis="x", labelrotation=14)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle=":", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Matched-cell technical associations | descriptive only", fontweight="bold")
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _scatter_by_sample(ax, points, meta, *, title, subtitle=""):
    for sample, name, _, _ in SAMPLES:
        take = (meta["sample"].to_numpy() == sample)
        ax.scatter(points[take, 0], points[take, 1], s=5, alpha=0.32,
                   c=PLOT_COLORS[sample], linewidths=0, rasterized=True,
                   label=f"{name} (n={int(take.sum()):,})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.13, linestyle=":")
    if subtitle:
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8,
                ha="left", va="bottom")


def _plot_compare(meta, model_pca, base_pca, model_umap, base_umap, output):
    if model_umap is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
        _scatter_by_sample(axes[0], model_pca, meta, title="SG-MAE cell embeddings | PCA")
        _scatter_by_sample(axes[1], base_pca, meta, title="ShamL-train normalized expression PCA | PCA")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12.7, 10.4), constrained_layout=True)
        _scatter_by_sample(axes[0, 0], model_pca, meta, title="SG-MAE cell embeddings | PCA")
        _scatter_by_sample(axes[0, 1], base_pca, meta, title="ShamL-train expression PCA | PCA")
        _scatter_by_sample(axes[1, 0], model_umap, meta, title="SG-MAE cell embeddings | UMAP")
        _scatter_by_sample(axes[1, 1], base_umap, meta, title="ShamL-train expression PCA | UMAP")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.025), frameon=False)
    fig.suptitle("Frozen representations | matched-cell exploratory comparison", y=1.065, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Frozen cell representation QC and matched expression PCA baseline")
    parser.add_argument("--input-dir", type=Path, default=root / "runs/sgmae_cell_embeddings_cuda_fixed")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs/gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs/splits/shaml_spatial_v1.json.gz")
    parser.add_argument("--projection-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--per-sample", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--expression-transform", choices=("library_normalized", "raw_log1p"), default="library_normalized")
    args = parser.parse_args(argv)

    output = args.output_dir or (args.input_dir / "representation_validation")
    if output.exists():
        raise FileExistsError(f"Output folder already exists: {output}. Use another directory.")
    if args.per_sample < 2:
        parser.error("--per-sample must be >= 2.")

    provenance, data = _read_data(args.input_dir)
    reference = embedding_reference_pc1(data["shaml"]["embedding"])
    qc = []
    for sample, *_ in SAMPLES:
        section = data[sample]
        meta = section["metadata"]
        print(f"Quantitative QC: {sample} | {len(meta):,} cells", flush=True)
        values = embedding_diagnostics(
            section["embedding"], meta["total_counts"].to_numpy(),
            meta["detected_genes"].to_numpy(), pc1_reference=reference,
        )
        qc.append({"sample": sample, **values})

    selected_embeddings, selected_meta, actual = _choose_rows(
        data, projection_csv=args.projection_csv, per_sample=args.per_sample, seed=args.seed
    )
    print(f"Matched representation comparison: {len(selected_meta):,} cells ({actual:,}/slice)", flush=True)

    vocabulary = GeneVocabulary.load(args.vocabulary)
    if vocabulary.species != "Mus musculus" or vocabulary.namespace != "ensembl_gene_id":
        raise ValueError("Expected mouse Ensembl Gene IDs.")
    if provenance["vocabulary_fingerprint"] != vocabulary.fingerprint:
        raise ValueError("Gene Vocabulary fingerprint differs from frozen extraction.")

    shaml_path = find_xenium_sample_file(args.data_root, "ShamL")
    if sha256_file(shaml_path) != provenance["shaml_h5_sha256"]:
        raise ValueError("ShamL source H5 does not match the frozen embedding extraction.")
    shaml = load_sgmae_sample(shaml_path, vocabulary)
    split = load_spatial_split(args.split, shaml.cell_ids, shaml_path.parent / "cells.parquet", expected_sample_name="ShamL")
    if split.fingerprint != provenance["split_fingerprint"]:
        raise ValueError("ShamL spatial split is inconsistent with extraction.")
    gene_order = np.sort(shaml.gene_ids.cpu().numpy())
    train_columns = aligned_gene_columns(shaml, gene_order)
    print(f"Fitting {args.expression_transform} expression PCA on {len(split.train_indices):,} ShamL training cells only", flush=True)
    train_matrix = shaml.expression[np.asarray(split.train_indices, dtype=np.int64)][:, train_columns].toarray()
    baseline = fit_train_reference_pca(train_matrix, feature_dim=selected_embeddings.shape[1],
                                       seed=args.seed, method=args.expression_transform)
    del train_matrix

    baseline_vectors = []
    for sample, *_ in SAMPLES:
        sample_meta = selected_meta[selected_meta["sample"] == sample].reset_index(drop=True)
        if sample == "shaml":
            dataset = shaml
            path = shaml_path
        else:
            path = find_xenium_sample_file(args.data_root, SAMPLE_PREFIX[sample])
            dataset = load_sgmae_sample(path, vocabulary)
        saved_manifest = json.loads((args.input_dir / sample / "manifest.json").read_text(encoding="utf-8"))
        if sha256_file(path) != saved_manifest["source_h5_sha256"]:
            raise ValueError(f"Source H5 differs from stored embeddings: {sample}")
        rows = matched_cell_rows(dataset, sample_meta)
        columns = aligned_gene_columns(dataset, gene_order)
        counts = dataset.expression[rows][:, columns].toarray()
        if not np.allclose(counts.sum(axis=1), sample_meta["total_counts"].to_numpy(), rtol=2e-6, atol=1e-3):
            raise ValueError(f"Selected raw counts differ from extracted QC totals: {sample}")
        if not np.array_equal(np.count_nonzero(counts, axis=1), sample_meta["detected_genes"].to_numpy()):
            raise ValueError(f"Selected detected-gene QC differs from extracted data: {sample}")
        transformed = expression_transform(counts, method=args.expression_transform)
        baseline_vectors.append(baseline.transform(transformed).astype(np.float32))
        print(f"  Matched expression baseline: {sample}, {len(rows):,} cells", flush=True)

    baseline_embeddings = np.concatenate(baseline_vectors, axis=0)
    matched_qc = []
    base_axis = (np.zeros(baseline_embeddings.shape[1], dtype=np.float64),
                 np.eye(1, baseline_embeddings.shape[1], dtype=np.float64)[0])
    for sample, *_ in SAMPLES:
        take = (selected_meta["sample"].to_numpy() == sample)
        sample_counts = selected_meta.loc[take, "total_counts"].to_numpy()
        sample_detected = selected_meta.loc[take, "detected_genes"].to_numpy()
        matched_qc.append({"sample": sample, "representation": "SG-MAE", **embedding_diagnostics(
            selected_embeddings[take], sample_counts, sample_detected, pc1_reference=reference
        )})
        matched_qc.append({"sample": sample, "representation": "Expression PCA", **embedding_diagnostics(
            baseline_embeddings[take], sample_counts, sample_detected, pc1_reference=base_axis
        )})
    model_pca, model_umap, model_variance = project_common_2d(
        selected_embeddings, seed=args.seed, skip_umap=args.skip_umap,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
    )
    base_pca, base_umap, base_variance = project_common_2d(
        baseline_embeddings, seed=args.seed, skip_umap=args.skip_umap,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
    )
    selected_meta = selected_meta.copy()
    selected_meta["sgmae_pca1"], selected_meta["sgmae_pca2"] = model_pca[:, 0], model_pca[:, 1]
    selected_meta["expression_pca1"], selected_meta["expression_pca2"] = base_pca[:, 0], base_pca[:, 1]
    if model_umap is not None:
        selected_meta["sgmae_umap1"], selected_meta["sgmae_umap2"] = model_umap[:, 0], model_umap[:, 1]
        selected_meta["expression_umap1"], selected_meta["expression_umap2"] = base_umap[:, 0], base_umap[:, 1]

    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "sample_qc.csv", qc)
    _write_csv(output / "matched_representation_qc.csv", matched_qc)
    selected_meta.to_csv(output / "matched_projection_cells.csv", index=False)
    _plot_qc(qc, output / "embedding_numerical_qc.png")
    _plot_matched_qc(matched_qc, output / "matched_technical_associations.png")
    _plot_compare(selected_meta, model_pca, base_pca, model_umap, base_umap,
                  output / "sgmae_vs_expression_pca.png")
    manifest = {
        "analysis": "frozen_cell_representation_qc",
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "split_fingerprint": split.fingerprint,
        "encoding_protocol": provenance["encoding_protocol"],
        "shaml_reference_pc1_fit": "ShamL spatial validation embeddings only; never external",
        "expression_pca_fit": "ShamL training cells only",
        "expression_transform": args.expression_transform,
        "expression_pca_components": int(baseline.n_components_),
        "per_slice_selected": int(actual),
        "total_selected": int(len(selected_meta)),
        "selection": "Existing validated projection CSV" if args.projection_csv else "equal seeded random subsample per slice",
        "seed": int(args.seed),
        "umap": None if args.skip_umap else {"n_neighbors": args.n_neighbors,
                                              "min_dist": args.min_dist, "seed": args.seed},
        "sgmae_joint_pca_2d_explained_variance": model_variance,
        "expression_joint_pca_2d_explained_variance": base_variance,
        "interpretation": "QC and exploratory matched-cell plots only; no biological labels or evidence of improved cell-type representation.",
    }
    with (output / "analysis_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Saved results to: {output}")
    print("No SG-MAE training or model weight updates were performed.")
    return manifest


if __name__ == "__main__":
    main()
