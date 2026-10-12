"""Build immutable, partition-isolated Xenium spatial kNN graphs for SG-MAE V2.0.

Run from the project root:
    python -m scripts.v2.build_graphs --output-dir runs/v2/graphs/shaml_k8_r50um

All held-out slices require both --samples and --confirm-external.
This command reads only H5 Cell IDs and Parquet centroids, not gene counts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from src.data.splitting import load_spatial_split
from src.data.spatial_graph import (
    aligned_spatial_coordinates,
    build_spatial_knn_graph,
    build_frozen_shaml_graph,
    save_spatial_graph,
)
from src.dataset import find_xenium_sample_file


SAMPLE_PREFIXES = {
    "shaml": "ShamL",
    "shamr": "ShamR",
    "hour12r": "Hour12R",
    "day2l": "Day2L",
}


def _read_h5_cell_ids(path: Path) -> tuple[str, ...]:
    """Read just the cell identifiers, not the expression matrix."""
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle or "barcodes" not in handle["matrix"]:
            raise ValueError(f"H5 is missing matrix/barcodes: {path}")
        raw = handle["matrix/barcodes"][:]
        if "shape" not in handle["matrix"]:
            raise ValueError(f"H5 is missing matrix/shape: {path}")
        num_cells = int(handle["matrix/shape"][1])
    ids = tuple(
        item.decode("utf-8", errors="strict") if isinstance(item, bytes) else str(item)
        for item in raw
    )
    if len(ids) != num_cells or len(set(ids)) != len(ids) or any(
        not cid or cid != cid.strip() for cid in ids
    ):
        raise ValueError(f"H5 Cell IDs are duplicated, invalid, or mis-sized: {path}")
    return ids


def _make_graphs(args, directory: Path) -> dict:
    """Create graphs in a staging directory before publishing them."""
    sample_paths = {}
    for sample in args.samples:
        sample_paths[sample] = find_xenium_sample_file(args.data_root, SAMPLE_PREFIXES[sample])

    if "shaml" not in sample_paths:
        sample_paths["shaml"] = find_xenium_sample_file(args.data_root, "ShamL")
    sham_path = sample_paths["shaml"]
    sham_ids = _read_h5_cell_ids(sham_path)
    sham_coordinates_path = sham_path.parent / "cells.parquet"
    split = load_spatial_split(
        args.split,
        sham_ids,
        sham_coordinates_path,
        expected_sample_name="ShamL",
    )
    manifest = {
        "schema_version": 1,
        "purpose": "spatial_graph_construction_only",
        "edge_direction": "directed_knn",
        "graph_parameters": {
            "k": args.k,
            "max_distance_um": args.max_distance_um,
        },
        "shaml_split_fingerprint": split.fingerprint,
        "shaml_split_counts": split.counts,
        "external_slices_used_for_training": False,
        "source_files": {},
        "graphs": {},
    }

    for sample in args.samples:
        h5_path = sample_paths[sample]
        if sample == "shaml":
            cell_ids = sham_ids
            xy_ids, xy = aligned_spatial_coordinates(cell_ids, sham_coordinates_path)
            partitions = (
                ("train", split.train_indices),
                ("validation", split.validation_indices),
            )
        else:
            cell_ids = _read_h5_cell_ids(h5_path)
            xy_ids, xy = aligned_spatial_coordinates(cell_ids, h5_path.parent / "cells.parquet")
            partitions = (("all", np.arange(len(cell_ids), dtype=np.int64)),)

        if xy_ids != cell_ids:
            raise AssertionError("Spatial coordinates are not in the exact H5 order.")
        manifest["source_files"][sample] = {
            "h5_path": str(h5_path),
            "parquet_path": str(h5_path.parent / "cells.parquet"),
            "sample_cells": len(cell_ids),
        }
        for partition, indices in partitions:
            print(f"Building {sample}/{partition}: {len(indices):,} cells")
            if sample == "shaml":
                graph = build_frozen_shaml_graph(
                    split=split,
                    cell_ids=cell_ids,
                    coordinates_um=xy,
                    partition=partition,
                    k=args.k,
                    max_distance_um=args.max_distance_um,
                )
            else:
                graph = build_spatial_knn_graph(
                    cell_ids=cell_ids,
                    coordinates_um=xy,
                    node_indices=indices,
                    sample_name=SAMPLE_PREFIXES[sample],
                    partition="all",
                    k=args.k,
                    max_distance_um=args.max_distance_um,
                )
            graph_path = directory / sample / f"{partition}.npz"
            save_spatial_graph(graph, graph_path)
            manifest["graphs"][f"{sample}/{partition}"] = {
                "npz": str(graph_path.relative_to(directory).as_posix()),
                **graph.summary(),
            }
            print(
                f"  directed edges={graph.num_edges:,}, "
                f"mean degree={graph.degree.mean():.2f}, "
                f"isolated={int((graph.degree == 0).sum()):,}"
            )

    with (directory / "graph_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Partition-safe spatial kNN construction")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument(
        "--split", type=Path,
        default=root / "configs" / "splits" / "shaml_spatial_v1.json.gz",
    )
    parser.add_argument(
        "--samples", nargs="+", choices=tuple(SAMPLE_PREFIXES), default=["shaml"]
    )
    parser.add_argument("--confirm-external", action="store_true")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-distance-um", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "v2" / "graphs" / "shaml_k8_r50um")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be at least 1.")
    if not np.isfinite(args.max_distance_um) or args.max_distance_um <= 0:
        parser.error("--max-distance-um must be finite and positive.")
    if len(set(args.samples)) != len(args.samples):
        parser.error("--samples must not contain duplicates.")
    if any(sample != "shaml" for sample in args.samples) and not args.confirm_external:
        parser.error("External slices require --confirm-external.")
    if args.confirm_external and all(sample == "shaml" for sample in args.samples):
        parser.error("--confirm-external is only used for held-out samples.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Specify a fresh path; saved graphs are immutable."
        )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sgmae_v2_graph_stage_", dir=args.output_dir.parent))
    try:
        manifest = _make_graphs(args, stage)
        if args.output_dir.exists():
            raise FileExistsError(f"Graph output destination appeared during construction: {args.output_dir}")
        stage.rename(args.output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"\nGraphs published: {args.output_dir}")
    print("All V1 models, counts, and frozen split files are unchanged.")
    return manifest


if __name__ == "__main__":
    main()
