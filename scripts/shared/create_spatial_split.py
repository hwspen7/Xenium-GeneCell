"""Create or verify the frozen ShamL-only spatial validation split."""

import argparse
from pathlib import Path

import h5py

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import (
    load_spatial_split,
    save_spatial_split,
    split_spatial_validation,
)
from src.dataset import (
    decode_strings,
    find_xenium_sample_file,
    load_xenium_gene_metadata,
)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create or verify a frozen ShamL-only spatial split."
    )
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument(
        "--vocabulary", type=Path,
        default=root / "configs" / "gene_vocabulary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "configs" / "splits" / "shaml_spatial_v1.json.gz",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--block-size-um", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-um", type=float, default=50.0)
    args = parser.parse_args()

    sham_path = find_xenium_sample_file(args.data_root, "ShamL")
    spatial_path = sham_path.parent / "cells.parquet"

    vocabulary = GeneVocabulary.load(args.vocabulary)
    if (
        vocabulary.species != "Mus musculus"
        or vocabulary.namespace != "ensembl_gene_id"
    ):
        raise ValueError("Expected a mouse Ensembl Gene ID vocabulary.")
    gene_ids, _ = load_xenium_gene_metadata(sham_path)
    missing = set(gene_ids) - set(vocabulary.gene_to_id)
    if missing:
        raise ValueError(f"Vocabulary does not cover {len(missing)} ShamL genes.")

    with h5py.File(sham_path, "r") as handle:
        cell_ids = decode_strings(handle["matrix"]["barcodes"][:])

    config = {
        "validation_fraction": args.validation_fraction,
        "block_size_um": args.block_size_um,
        "seed": args.seed,
        "buffer_um": args.buffer_um,
    }
    if args.output.exists():
        result = load_spatial_split(
            args.output, cell_ids, spatial_path,
            expected_sample_name="ShamL",
        )
        existing = {
            "validation_fraction": result.validation_fraction,
            "block_size_um": result.block_size_um,
            "seed": result.seed,
            "buffer_um": result.buffer_um,
        }
        if existing != config:
            raise ValueError(
                "Frozen split configuration differs from requested values. "
                "Choose a new output path for a new experiment."
            )
        status = "VERIFIED (no changes)"
    else:
        result = split_spatial_validation(
            cell_ids=cell_ids, cells_parquet=spatial_path,
            sample_name="ShamL", **config,
        )
        save_spatial_split(result, args.output)
        # Verify the on-disk representation before reporting success.
        loaded = load_spatial_split(args.output, cell_ids, spatial_path)
        if loaded.fingerprint != result.fingerprint:
            raise RuntimeError("Saved split differs from generated split.")
        status = "CREATED"

    print(f"ShamL Spatial Split: {status}")
    print(f"Source: {sham_path}")
    print(f"Manifest: {args.output}")
    print(f"Cells: {len(cell_ids):,}")
    print(f"Train: {result.counts['train']:,}")
    print(f"Validation: {result.counts['validation']:,}")
    print(f"Buffer-excluded: {result.counts['excluded']:,}")
    print(f"Block size: {result.block_size_um:g} um")
    print(f"Buffer: {result.buffer_um:g} um")
    print(f"Seed: {result.seed}")
    print(f"Split fingerprint: {result.fingerprint}")
    print(f"Vocabulary fingerprint: {vocabulary.fingerprint}")
    print("ShamR: untouched, reserved for held-out testing.")
    print("Hour12R / Day2L: untouched, reserved for later experiments.")


if __name__ == "__main__":
    main()
