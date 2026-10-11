"""Regression tests for leakage-safe ShamL spatial splitting."""

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.splitting import (
    load_spatial_split,
    save_spatial_split,
    split_spatial_validation,
)


def expect_failure(callback, exception=ValueError):
    try:
        callback()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__} was not raised.")


def synthetic_cells():
    """Build varied grid blocks with neighboring cells near shared edges."""
    cell_ids, records = [], []
    for bx in range(7):
        for by in range(7):
            for dx, dy in ((5, 5), (5, 245), (245, 5), (245, 245)):
                i = len(cell_ids)
                cid = f"synthetic-cell-{i:05d}"
                cell_ids.append(cid)
                records.append({
                    "cell_id": cid,
                    "x_centroid": bx * 250.0 + dx,
                    "y_centroid": by * 250.0 + dy,
                })
    return cell_ids, pd.DataFrame.from_records(records)


def test_core():
    ids, frame = synthetic_cells()
    split = split_spatial_validation(
        ids, frame, validation_fraction=0.2,
        block_size_um=250, seed=42, buffer_um=25,
    )
    groups = split.cell_id_groups()
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert set(groups["train"]).isdisjoint(groups["excluded"])
    assert set(groups["validation"]).isdisjoint(groups["excluded"])
    assert sum(split.counts.values()) == len(ids)
    assert split.counts["train"] > 0 and split.counts["validation"] > 0
    assert split.counts["excluded"] > 0

    train_blocks = set(map(tuple, split.block_ids[split.train_indices]))
    valid_blocks = set(map(tuple, split.block_ids[split.validation_indices]))
    assert not train_blocks.intersection(valid_blocks)
    print("Disjoint spatial blocks and three-way partition: PASS")

    coordinate_matrix = frame[["x_centroid", "y_centroid"]].to_numpy()
    tree = cKDTree(coordinate_matrix[split.validation_indices])
    distances, _ = tree.query(coordinate_matrix[split.train_indices])
    assert (distances > 25.0).all()
    excluded_distances, _ = tree.query(coordinate_matrix[split.excluded_indices])
    assert (excluded_distances <= 25.0).all()
    print("Validation-cell buffer isolation: PASS")

    rng = np.random.default_rng(123)
    row_shuffle = rng.permutation(len(ids))
    shuffled_ids = [ids[i] for i in row_shuffle]
    shuffled_frame = frame.sample(frac=1.0, random_state=87).reset_index(drop=True)
    shuffled = split_spatial_validation(
        shuffled_ids, shuffled_frame, validation_fraction=0.2,
        block_size_um=250, seed=42, buffer_um=25,
    )
    assert shuffled.cell_id_groups() == split.cell_id_groups()
    assert shuffled.fingerprint == split.fingerprint
    print("Determinism under H5/Parquet row reordering: PASS")

    no_buffer = split_spatial_validation(ids, frame, buffer_um=0)
    assert len(no_buffer.excluded_indices) == 0
    assert sum(no_buffer.counts.values()) == len(ids)
    print("Optional zero-buffer baseline: PASS")

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "shaml_spatial_v1.json.gz"
        save_spatial_split(split, path)
        expect_failure(lambda: save_spatial_split(split, path), FileExistsError)
        restored = load_spatial_split(path, shuffled_ids, shuffled_frame)
        assert restored.cell_id_groups() == split.cell_id_groups()
        assert restored.fingerprint == split.fingerprint
        print("Frozen manifest round-trip and no overwrite: PASS")

        expect_failure(
            lambda: load_spatial_split(
                path, shuffled_ids, shuffled_frame, expected_sample_name="ShamR"
            )
        )
        corrupted_frame = shuffled_frame.copy()
        corrupted_frame.loc[0, "x_centroid"] += 1
        expect_failure(lambda: load_spatial_split(path, shuffled_ids, corrupted_frame))
        print("Sample and coordinate drift rejected: PASS")

        with gzip.open(path, "rt", encoding="utf-8") as stream:
            tampered = json.load(stream)
        tampered["cell_ids"]["train"][0] = "forged-cell-id"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(tampered, stream)
        expect_failure(lambda: load_spatial_split(path, ids, frame))
        print("Manifest tampering rejected: PASS")

    dup_ids = ids.copy()
    dup_ids[1] = dup_ids[0]
    expect_failure(lambda: split_spatial_validation(dup_ids, frame))
    duplicated_frame = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    expect_failure(lambda: split_spatial_validation(ids, duplicated_frame))
    incomplete_frame = frame.iloc[:-1]
    expect_failure(lambda: split_spatial_validation(ids, incomplete_frame))
    nan_frame = frame.copy()
    nan_frame.loc[0, "y_centroid"] = np.nan
    expect_failure(lambda: split_spatial_validation(ids, nan_frame))
    print("Duplicate, missing and invalid coordinate rejection: PASS")

    for kwargs in (
        {"validation_fraction": 0.0},
        {"validation_fraction": 1.0},
        {"block_size_um": 0.0},
        {"buffer_um": -1.0},
        {"seed": -1},
    ):
        expect_failure(lambda options=kwargs: split_spatial_validation(ids, frame, **options))
    expect_failure(lambda: split_spatial_validation(ids, frame, block_size_um=100000))
    expect_failure(lambda: split_spatial_validation(ids, frame, buffer_um=100000))
    print("Invalid configuration and degenerate split rejected: PASS")


def test_binary_cell_ids():
    """Xenium Parquet stores bytes; H5 barcodes are typically strings."""
    ids, frame = synthetic_cells()
    baseline = split_spatial_validation(ids, frame, buffer_um=25)

    binary = frame.copy()
    binary["cell_id"] = [cid.encode("utf-8") for cid in binary["cell_id"]]
    result = split_spatial_validation(ids, binary, buffer_um=25)
    assert result.cell_id_groups() == baseline.cell_id_groups()
    assert result.fingerprint == baseline.fingerprint

    # Read-only manifest validation must accept the binary Parquet IDs too.
    with TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "split.json.gz"
        save_spatial_split(baseline, manifest)
        loaded = load_spatial_split(manifest, ids, binary)
        assert loaded.fingerprint == baseline.fingerprint

    invalid_encoding = binary.copy()
    invalid_encoding.at[0, "cell_id"] = b"\xff"
    expect_failure(lambda: split_spatial_validation(ids, invalid_encoding))

    decoded_duplicate = binary.copy()
    decoded_duplicate.at[1, "cell_id"] = ids[0]
    expect_failure(lambda: split_spatial_validation(ids, decoded_duplicate))

    invalid_identifier = binary.copy()
    invalid_identifier.at[0, "cell_id"] = 123
    expect_failure(lambda: split_spatial_validation(ids, invalid_identifier))
    print("Binary Xenium cell IDs and strict decoding: PASS")


def test_real_shaml():
    """Read only ShamL H5 barcodes and cells.parquet; never write official files."""
    root = Path(__file__).resolve().parents[2]
    data_root = root / "data"
    if not data_root.is_dir():
        print("Real ShamL spatial metadata: SKIP (no data directory)")
        return
    import h5py
    from src.dataset import find_xenium_sample_file, decode_strings

    try:
        h5_path = find_xenium_sample_file(data_root, "ShamL")
    except ValueError:
        print("Real ShamL spatial metadata: SKIP (ShamL unavailable)")
        return

    with h5py.File(h5_path, "r") as handle:
        ids = decode_strings(handle["matrix"]["barcodes"][:])
    parquet_path = h5_path.parent / "cells.parquet"
    result = split_spatial_validation(
        ids, parquet_path,
        validation_fraction=0.20, block_size_um=250,
        seed=42, buffer_um=50, sample_name="ShamL",
    )
    assert sum(result.counts.values()) == len(ids)
    assert len(result.train_indices) > 0 and len(result.validation_indices) > 0
    with TemporaryDirectory() as tmp:
        temp_manifest = Path(tmp) / "split.json.gz"
        save_spatial_split(result, temp_manifest)
        read_back = load_spatial_split(temp_manifest, ids, parquet_path)
        assert read_back.fingerprint == result.fingerprint
    print(
        f"Real ShamL spatial metadata: PASS "
        f"({result.counts['train']:,} train / "
        f"{result.counts['validation']:,} validation / "
        f"{result.counts['excluded']:,} buffer-excluded)"
    )


def main():
    test_core()
    test_binary_cell_ids()
    test_real_shaml()
    print("\nAll SG-MAE spatial splitting tests passed.")


if __name__ == "__main__":
    main()
