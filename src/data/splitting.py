"""Reproducible spatial-block splitting and frozen manifests for SG-MAE.

Only the ShamL experiment launcher should create V1 training/validation splits.
The core splitter is sample-agnostic so its invariants can be unit-tested.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCHEMA_VERSION = 1
_REQUIRED_COLUMNS = ("cell_id", "x_centroid", "y_centroid")
_SPLIT_KEYS = ("train", "validation", "excluded")


def _validate_config(validation_fraction, block_size_um, seed, buffer_um):
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (float, int))
        or not math.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    for name, value, allow_zero in (
        ("block_size_um", block_size_um, False),
        ("buffer_um", buffer_um, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)
        ):
            raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")


def _canonical_json(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _read_coordinates(cells_parquet: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read only the three essential spatial columns from Parquet or a test frame."""
    if isinstance(cells_parquet, pd.DataFrame):
        if not set(_REQUIRED_COLUMNS).issubset(cells_parquet.columns):
            raise ValueError("Spatial metadata is missing required columns.")
        table = cells_parquet.loc[:, list(_REQUIRED_COLUMNS)].copy()
    else:
        path = Path(cells_parquet)
        if not path.is_file():
            raise FileNotFoundError(f"cells.parquet not found: {path}")
        try:
            table = pd.read_parquet(path, columns=list(_REQUIRED_COLUMNS))
        except ImportError as error:
            raise ImportError("Reading cells.parquet requires pyarrow or fastparquet.") from error
    if table.empty:
        raise ValueError("No cells found in spatial metadata.")
    return table


def _decode_cell_id(value) -> str:
    """Decode Xenium Parquet binary Cell IDs without changing their identity."""
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("Invalid UTF-8 cell_id in spatial metadata.") from error
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Invalid cell_id in spatial metadata.")
    return value


def _aligned_coordinates(cell_ids: Sequence[str], cells_parquet):
    ids = list(cell_ids)
    if not ids or any(not isinstance(cid, str) or not cid or cid != cid.strip() for cid in ids):
        raise ValueError("cell_ids must be non-empty valid strings.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cell IDs in expression matrix.")

    frame = _read_coordinates(cells_parquet)
    if frame["cell_id"].isna().any():
        raise ValueError("Missing cell_id in spatial metadata.")
    # Xenium cells.parquet commonly stores cell_id as binary (bytes),
    # while H5 barcode readers return decoded strings. Decode strictly
    # before checking uniqueness or matching the expression matrix.
    table_ids = [_decode_cell_id(cid) for cid in frame["cell_id"].tolist()]
    if len(set(table_ids)) != len(table_ids):
        raise ValueError("Duplicate cell IDs in spatial metadata.")
    if len(table_ids) != len(ids) or set(table_ids) != set(ids):
        missing = len(set(ids).difference(table_ids))
        extra = len(set(table_ids).difference(ids))
        raise ValueError(f"Expression/spatial Cell ID mismatch: missing={missing}, extra={extra}.")

    index = pd.Index(table_ids).get_indexer(ids)
    if (index < 0).any():
        raise ValueError("Some expression Cell IDs could not be aligned.")
    try:
        coords = frame.loc[:, ["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)[index]
    except (TypeError, ValueError) as error:
        raise ValueError("Spatial coordinates must be numeric.") from error
    if coords.shape != (len(ids), 2) or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be finite x/y values.")
    return ids, coords


def _coordinate_fingerprint(ids: Sequence[str], coords: np.ndarray) -> str:
    """Bind each Cell ID to its coordinates independently of source row order."""
    digest = hashlib.sha256()
    for i in sorted(range(len(ids)), key=ids.__getitem__):
        record = [ids[i], float(coords[i, 0]).hex(), float(coords[i, 1]).hex()]
        digest.update(_canonical_json(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _spatial_block_ids(coords: np.ndarray, block_size_um: float) -> np.ndarray:
    scaled = np.floor(coords / float(block_size_um))
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= np.iinfo(np.int64).max:
        raise ValueError("Coordinate / block size is outside supported range.")
    return scaled.astype(np.int64)


@dataclass(frozen=True)
class SpatialSplit:
    """Split indices refer to the exact cell_ids order supplied by the caller."""

    sample_name: str
    cell_ids: tuple[str, ...]
    coordinates_fingerprint: str
    block_ids: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    excluded_indices: np.ndarray
    validation_fraction: float
    block_size_um: float
    seed: int
    buffer_um: float

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train_indices),
            "validation": len(self.validation_indices),
            "excluded": len(self.excluded_indices),
        }

    def cell_id_groups(self) -> dict[str, list[str]]:
        """Return sorted cell IDs for portable, row-order-independent storage."""
        return {
            "train": sorted(self.cell_ids[i] for i in self.train_indices),
            "validation": sorted(self.cell_ids[i] for i in self.validation_indices),
            "excluded": sorted(self.cell_ids[i] for i in self.excluded_indices),
        }

    def manifest(self) -> dict:
        """Create a fingerprinted, self-contained immutable split definition."""
        content = {
            "schema_version": SCHEMA_VERSION,
            "sample_name": self.sample_name,
            "coordinates_fingerprint": self.coordinates_fingerprint,
            "configuration": {
                "validation_fraction": self.validation_fraction,
                "block_size_um": self.block_size_um,
                "seed": self.seed,
                "buffer_um": self.buffer_um,
            },
            "counts": self.counts,
            "cell_ids": self.cell_id_groups(),
        }
        content["fingerprint"] = _fingerprint(content)
        return content

    @property
    def fingerprint(self) -> str:
        return self.manifest()["fingerprint"]


def split_spatial_validation(
    cell_ids: Sequence[str],
    cells_parquet: str | Path | pd.DataFrame,
    validation_fraction: float = 0.20,
    block_size_um: float = 250.0,
    seed: int = 42,
    buffer_um: float = 50.0,
    sample_name: str = "ShamL",
) -> SpatialSplit:
    """Select validation spatial blocks, then buffer against validation cells.

    Blocks are fixed in the absolute x/y coordinate reference frame (floor(x/size)).
    Random block ordering is independent of Parquet/H5 row ordering.
    Validation occupancy approaches the target fraction before buffer exclusion;
    it cannot be guaranteed exactly because complete blocks are selected.
    """
    _validate_config(validation_fraction, block_size_um, seed, buffer_um)
    if not isinstance(sample_name, str) or not sample_name.strip():
        raise ValueError("sample_name must be a nonempty string.")
    ids, coords = _aligned_coordinates(cell_ids, cells_parquet)
    block_ids = _spatial_block_ids(coords, block_size_um)

    unique_blocks, inverse, counts = np.unique(
        block_ids, axis=0, return_inverse=True, return_counts=True,
    )
    if len(unique_blocks) < 2:
        raise ValueError("At least two occupied spatial blocks are required.")

    # Deterministic block permutation and cell-count-aware validation allocation.
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(unique_blocks))
    cumulative = np.cumsum(counts[permutation], dtype=np.int64)
    target = len(ids) * float(validation_fraction)
    # Consider 1..num_blocks-1 prefixes so both groups remain non-empty.
    best_k = int(np.argmin(np.abs(cumulative[:-1] - target))) + 1
    selected = np.zeros(len(unique_blocks), dtype=bool)
    selected[permutation[:best_k]] = True
    validation_mask = selected[inverse]
    train_mask = ~validation_mask
    excluded_mask = np.zeros(len(ids), dtype=bool)

    if buffer_um > 0:
        tree = cKDTree(coords[validation_mask])
        distance, _ = tree.query(
            coords[train_mask], k=1, distance_upper_bound=float(buffer_um),
        )
        candidate_train = np.flatnonzero(train_mask)
        too_close = np.isfinite(distance)
        excluded_mask[candidate_train[too_close]] = True
        train_mask[excluded_mask] = False

    if not train_mask.any() or not validation_mask.any():
        raise ValueError(
            "Spatial split has no training or validation cells; "
            "reduce buffer_um or adjust the block configuration."
        )

    return SpatialSplit(
        sample_name=sample_name,
        cell_ids=tuple(ids),
        coordinates_fingerprint=_coordinate_fingerprint(ids, coords),
        block_ids=block_ids,
        train_indices=np.flatnonzero(train_mask),
        validation_indices=np.flatnonzero(validation_mask),
        excluded_indices=np.flatnonzero(excluded_mask),
        validation_fraction=float(validation_fraction),
        block_size_um=float(block_size_um),
        seed=int(seed),
        buffer_um=float(buffer_um),
    )


def save_spatial_split(split: SpatialSplit, file_path: str | Path) -> Path:
    """Atomically save one frozen manifest; never overwrite an existing path."""
    path = Path(file_path)
    if path.exists():
        raise FileExistsError(
            f"Frozen spatial split already exists: {path}. "
            "Load it instead of regenerating it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split.manifest()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as gz:
                gz.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        # Refuse overwriting on all supported platforms, even for race-created files.
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(f"Frozen spatial split already exists: {path}.") from None
        except OSError:
            # os.link is unavailable on some filesystems. Fail closed.
            raise RuntimeError("Filesystem does not support safe create-only split publication.") from None
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_spatial_split(
    file_path: str | Path,
    cell_ids: Sequence[str],
    cells_parquet: str | Path | pd.DataFrame,
    expected_sample_name: str = "ShamL",
) -> SpatialSplit:
    """Verify manifest integrity, sample coordinates and all split invariants."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen spatial split not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)

    if not isinstance(saved, dict) or set(saved) != {
        "schema_version", "sample_name", "coordinates_fingerprint",
        "configuration", "counts", "cell_ids", "fingerprint",
    }:
        raise ValueError("Invalid spatial split manifest structure.")
    if type(saved["schema_version"]) is not int or saved["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported spatial split manifest version.")
    supplied_fingerprint = saved["fingerprint"]
    if not isinstance(supplied_fingerprint, str) or supplied_fingerprint != _fingerprint(
        {key: value for key, value in saved.items() if key != "fingerprint"}
    ):
        raise ValueError("Spatial split manifest fingerprint mismatch.")
    if saved["sample_name"] != expected_sample_name:
        raise ValueError("Spatial split sample identity mismatch.")

    config = saved["configuration"]
    if not isinstance(config, dict) or set(config) != {
        "validation_fraction", "block_size_um", "seed", "buffer_um"
    }:
        raise ValueError("Invalid spatial split configuration.")
    _validate_config(**config)
    groups = saved["cell_ids"]
    counts = saved["counts"]
    if not isinstance(groups, dict) or set(groups) != set(_SPLIT_KEYS):
        raise ValueError("Invalid spatial split cell ID groups.")
    if not isinstance(counts, dict) or set(counts) != set(_SPLIT_KEYS):
        raise ValueError("Invalid spatial split counts.")
    for key in _SPLIT_KEYS:
        group = groups[key]
        if (
            not isinstance(group, list)
            or not all(isinstance(cid, str) and cid for cid in group)
            or group != sorted(set(group))
            or type(counts[key]) is not int
            or counts[key] != len(group)
        ):
            raise ValueError(f"Invalid {key} IDs or counts in split manifest.")
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Manifest must have nonempty train and validation groups.")

    aligned_ids, coords = _aligned_coordinates(cell_ids, cells_parquet)
    if _coordinate_fingerprint(aligned_ids, coords) != saved["coordinates_fingerprint"]:
        raise ValueError("Coordinates or cell identity changed since split creation.")
    joined = groups["train"] + groups["validation"] + groups["excluded"]
    if len(set(joined)) != len(joined) or set(joined) != set(aligned_ids):
        raise ValueError("Split groups must form a disjoint complete cell partition.")
    membership = {cid: label for label in _SPLIT_KEYS for cid in groups[label]}
    index_arrays = {
        label: np.fromiter(
            (index for index, cid in enumerate(aligned_ids) if membership[cid] == label),
            dtype=np.int64,
        )
        for label in _SPLIT_KEYS
    }
    blocks = _spatial_block_ids(coords, config["block_size_um"])
    train_blocks = set(map(tuple, blocks[index_arrays["train"]]))
    validation_blocks = set(map(tuple, blocks[index_arrays["validation"]]))
    excluded_blocks = set(map(tuple, blocks[index_arrays["excluded"]]))
    if train_blocks & validation_blocks:
        raise ValueError("Frozen split has spatial block leakage.")
    if excluded_blocks & validation_blocks:
        raise ValueError("Excluded cells must not lie within validation blocks.")
    if config["buffer_um"] == 0 and len(index_arrays["excluded"]) > 0:
        raise ValueError("A zero-buffer split cannot contain excluded cells.")
    if config["buffer_um"] > 0:
        tree = cKDTree(coords[index_arrays["validation"]])
        distance, _ = tree.query(
            coords[index_arrays["train"]], k=1,
            distance_upper_bound=config["buffer_um"],
        )
        if np.isfinite(distance).any():
            raise ValueError("Frozen split violates its validation buffer.")
        # All excluded cells must genuinely lie within the buffer radius.
        if len(index_arrays["excluded"]):
            distances, _ = tree.query(
                coords[index_arrays["excluded"]], k=1,
                distance_upper_bound=config["buffer_um"],
            )
            if not np.isfinite(distances).all():
                raise ValueError("Excluded cells do not match validation buffer.")

    split = SpatialSplit(
        sample_name=saved["sample_name"], cell_ids=tuple(aligned_ids),
        coordinates_fingerprint=saved["coordinates_fingerprint"], block_ids=blocks,
        train_indices=index_arrays["train"],
        validation_indices=index_arrays["validation"],
        excluded_indices=index_arrays["excluded"],
        **config,
    )
    if split.fingerprint != supplied_fingerprint:
        raise ValueError("Frozen spatial split content is not canonical.")
    return split
