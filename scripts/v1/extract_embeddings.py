"""Extract frozen SG-MAE cell embeddings for four kidney Xenium slices.

From the project root:
  python -m scripts.v1.extract_embeddings --confirm-external
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import load_spatial_split
from src.dataset import find_xenium_sample_file, load_sgmae_sample
from src.evaluation.cell_embeddings import (
    EMBEDDING_SCHEMA_VERSION,
    aligned_spatial_coordinates,
    encode_cells,
    save_sample_embedding,
    sha256_file,
    verify_gene_panel,
)
from src.evaluation.evaluate_sgmae_v1 import load_frozen_best


SAMPLES = (("shaml", "ShamL"), ("shamr", "ShamR"), ("hour12r", "Hour12R"), ("day2l", "Day2L"))


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="SG-MAE V1.1 frozen 128D embeddings")
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/v1/training/scheduler/best.pt")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs/gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs/splits/shaml_spatial_v1.json.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/v1/representations/embeddings_new")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-external", action="store_true")
    args = parser.parse_args(argv)

    if args.checkpoint.name != "best.pt":
        parser.error("Only frozen best.pt checkpoints are accepted.")
    if not args.confirm_external:
        parser.error("Use --confirm-external to authorize extraction from three held-out slices.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Choose a new path.")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    vocabulary = GeneVocabulary.load(args.vocabulary)
    if vocabulary.species != "Mus musculus" or vocabulary.namespace != "ensembl_gene_id":
        raise ValueError("Expected Mus musculus Ensembl gene vocabulary.")
    sham_path = find_xenium_sample_file(args.data_root, "ShamL")
    shaml = load_sgmae_sample(sham_path, vocabulary)
    split = load_spatial_split(
        args.split, shaml.cell_ids, sham_path.parent / "cells.parquet", expected_sample_name="ShamL"
    )
    sham_sha = sha256_file(sham_path)
    model, checkpoint, metadata = load_frozen_best(
        args.checkpoint,
        vocabulary_fingerprint=vocabulary.fingerprint,
        split_fingerprint=split.fingerprint,
        sham_h5_sha256=sham_sha,
        vocabulary_size=len(vocabulary),
        device=device,
    )
    model.eval()
    checkpoint_hash = sha256_file(args.checkpoint)
    all_samples = []
    datasets = {"shaml": (sham_path, shaml)}
    for key, prefix in SAMPLES[1:]:
        path = find_xenium_sample_file(args.data_root, prefix)
        dataset = load_sgmae_sample(path, vocabulary)
        verify_gene_panel(shaml.gene_ids.numpy(), dataset.gene_ids.numpy())
        datasets[key] = (path, dataset)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for sample_key, _ in SAMPLES:
        path, dataset = datasets[sample_key]
        selected = (
            np.asarray(split.validation_indices, dtype=np.int64)
            if sample_key == "shaml"
            else np.arange(len(dataset), dtype=np.int64)
        )
        print(f"Extracting {sample_key}: {len(selected):,} cells ({device})", flush=True)
        coordinates = aligned_spatial_coordinates(dataset.cell_ids, path.parent / "cells.parquet")
        embeddings, counts, detected = encode_cells(
            model, dataset, selected, batch_size=args.batch_size, device=device
        )
        if embeddings.shape[1] != int(metadata["model_config"]["cell_dim"]):
            raise ValueError(f"Unexpected Cell Embedding width for {sample_key}.")
        sample_h5_sha = sham_sha if sample_key == "shaml" else sha256_file(path)
        save_sample_embedding(
            args.output_dir / sample_key,
            sample_key=sample_key,
            dataset=dataset,
            indices=selected,
            coordinates=coordinates,
            embeddings=embeddings,
            total_counts=counts,
            detected_genes=detected,
            provenance={
                "scope": "spatial_validation" if sample_key == "shaml" else "entire_external_slice",
                "checkpoint_sha256": checkpoint_hash,
                "vocabulary_fingerprint": vocabulary.fingerprint,
                "split_fingerprint": split.fingerprint,
                "source_h5_sha256": sample_h5_sha,
            },
        )
        all_samples.append({"sample": sample_key, "cells": int(len(selected)), "source_h5_sha256": sample_h5_sha})
        print(f"  Saved {embeddings.shape} -> {args.output_dir / sample_key}", flush=True)

    report = {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": int(checkpoint["best_epoch"]),
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "split_fingerprint": split.fingerprint,
        "shaml_h5_sha256": sham_sha,
        "encoding_protocol": "full_measured_visible_no_mask",
        "conditioning": "No model fine-tuning. All measured genes, including measured zero counts, are visible.",
        "samples": all_samples,
    }
    with (args.output_dir / "extraction_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved four-slice embeddings to {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
