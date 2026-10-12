"""Cell-ID-aligned, partition-isolated spatial kNN graphs for SG-MAE V2.0.

Graph edges use spatial coordinates only. They never inspect gene counts.
CSR row i describes outgoing neighbors of local node i, and node_indices[i]
identifies the corresponding row in the original Xenium H5 expression matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from src.data.splitting import _aligned_coordinates, _coordinate_fingerprint


SCHEMA_VERSION = 1


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def aligned_spatial_coordinates(cell_ids: Sequence[str], cells_parquet):
    """Return (cell_ids, [N,2] centroids) in the exact H5 cell order.

    This reuses the strict identity checks from the frozen V1 spatial splitter,
    including UTF-8 decoding of Xenium Parquet byte-string cell IDs.
    Coordinates are expected in micrometers, as in Xenium cells.parquet.
    """
    ids, xy = _aligned_coordinates(cell_ids, cells_parquet)
    return tuple(ids), np.ascontiguousarray(xy, dtype=np.float64)


def _as_indices(node_indices, n_total: int) -> np.ndarray:
    indices = np.asarray(node_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("node_indices must contain at least one index.")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise TypeError("node_indices must contain integers.")
    if indices.dtype.kind == "u" and np.any(indices > np.iinfo(np.int64).max):
        raise ValueError("node_indices exceed the supported integer range.")
    indices = indices.astype(np.int64, copy=True)
    if np.any(indices < 0) or np.any(indices >= n_total):
        raise IndexError("node_indices contain a row outside the H5 sample.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Repeated node_indices would duplicate graph nodes.")
    return indices


def _graph_fingerprint(graph: "SpatialGraph") -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "schema_version": SCHEMA_VERSION,
        "sample": graph.sample_name,
        "partition": graph.partition,
        "k": graph.k,
        "max_distance_um": graph.max_distance_um,
        "split_fingerprint": graph.split_fingerprint,
        "coordinates_fingerprint": graph.coordinates_fingerprint,
        "cell_ids": graph.cell_ids,
    }))
    for array, dtype in (
        (graph.node_indices, "<i8"),
        (graph.xy_um, "<f8"),
        (graph.indptr, "<i8"),
        (graph.neighbors, "<i8"),
        (graph.distances_um, "<f8"),
    ):
        canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
        digest.update(_canonical_json(list(canonical.shape)))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialGraph:
    """Directed sparse graph, with local CSR indices and global H5 row mapping."""

    sample_name: str
    partition: str
    cell_ids: tuple[str, ...]
    node_indices: np.ndarray
    xy_um: np.ndarray
    indptr: np.ndarray
    neighbors: np.ndarray
    distances_um: np.ndarray
    k: int
    max_distance_um: float
    coordinates_fingerprint: str
    split_fingerprint: str | None = None

    def __post_init__(self):
        n = len(self.cell_ids)
        if not self.sample_name or self.partition not in ("train", "validation", "all"):
            raise ValueError("Invalid graph sample or partition.")
        if n == 0 or len(set(self.cell_ids)) != n or any(
            not isinstance(cid, str) or not cid or cid.strip() != cid for cid in self.cell_ids
        ):
            raise ValueError("Graph cell IDs must be nonempty and unique.")
        if type(self.k) is not int or self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if not math.isfinite(self.max_distance_um) or self.max_distance_um <= 0:
            raise ValueError("max_distance_um must be finite and positive.")
        if not isinstance(self.coordinates_fingerprint, str) or not self.coordinates_fingerprint:
            raise ValueError("Missing spatial coordinate fingerprint.")
        if self.partition in ("train", "validation") and not self.split_fingerprint:
            raise ValueError("A train/validation graph requires a frozen split fingerprint.")
        if self.partition == "all" and self.split_fingerprint is not None:
            raise ValueError("An external all-cells graph must not claim a training split.")
        arrays = (
            (self.node_indices, (n,)),
            (self.xy_um, (n, 2)),
            (self.indptr, (n + 1,)),
        )
        for array, shape in arrays:
            if not isinstance(array, np.ndarray) or array.shape != shape:
                raise ValueError("Graph array dimensions are inconsistent.")
        if any(array.dtype.kind not in "iu" for array in (self.node_indices, self.indptr, self.neighbors)):
            raise TypeError("Graph node and CSR indices must be integers.")
        if self.xy_um.dtype.kind != "f" or self.distances_um.dtype.kind != "f":
            raise TypeError("Coordinates and distances must be floating-point.")
        if np.unique(self.node_indices).size != n or np.any(self.node_indices < 0):
            raise ValueError("Graph node_indices must be nonnegative and unique.")
        if not np.isfinite(self.xy_um).all() or not np.isfinite(self.distances_um).all():
            raise ValueError("Graph coordinates and distances must be finite.")
        if self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0):
            raise ValueError("Invalid CSR indptr.")
        e = int(self.indptr[-1])
        if self.neighbors.shape != (e,) or self.distances_um.shape != (e,):
            raise ValueError("CSR indices and edge distances disagree.")
        if np.any(self.neighbors < 0) or np.any(self.neighbors >= n):
            raise ValueError("Graph contains an out-of-range neighbor.")
        if np.any(self.distances_um < 0) or np.any(self.distances_um > self.max_distance_um + 1e-8):
            raise ValueError("An edge violates the maximum spatial distance.")
        degree = np.diff(self.indptr)
        if np.any(degree > self.k):
            raise ValueError("A graph node has more than k neighbors.")
        sources = np.repeat(np.arange(n, dtype=np.int64), degree)
        if np.any(sources == self.neighbors):
            raise ValueError("Graph contains an unexpected self-edge.")
        if e:
            measured = np.linalg.norm(
                self.xy_um[sources] - self.xy_um[self.neighbors], axis=1
            )
            if not np.allclose(measured, self.distances_um, atol=1e-7, rtol=1e-7):
                raise ValueError("Stored edge distances disagree with cell coordinates.")
        for i in range(n):
            a, b = int(self.indptr[i]), int(self.indptr[i + 1])
            if np.unique(self.neighbors[a:b]).size != b - a:
                raise ValueError("Graph contains duplicate neighbors.")

    @property
    def num_nodes(self) -> int:
        return len(self.cell_ids)

    @property
    def num_edges(self) -> int:
        return int(self.neighbors.size)

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    @property
    def fingerprint(self) -> str:
        return _graph_fingerprint(self)

    def summary(self) -> dict:
        degrees = self.degree
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_name": self.sample_name,
            "partition": self.partition,
            "num_nodes": self.num_nodes,
            "num_directed_edges": self.num_edges,
            "num_isolated_nodes": int((degrees == 0).sum()),
            "mean_out_degree": float(degrees.mean()),
            "max_out_degree": int(degrees.max()),
            "k": self.k,
            "max_distance_um": self.max_distance_um,
            "split_fingerprint": self.split_fingerprint,
            "coordinates_fingerprint": self.coordinates_fingerprint,
            "graph_fingerprint": self.fingerprint,
            "edge_direction": "directed_knn",
            "node_indices_semantics": "zero_based_original_h5_cell_row",
            "neighbor_indices_semantics": "zero_based_local_graph_node_row",
            "unit": "micrometer",
        }


def build_spatial_knn_graph(
    *,
    cell_ids: Sequence[str],
    coordinates_um: np.ndarray,
    node_indices,
    sample_name: str,
    partition: str,
    k: int = 8,
    max_distance_um: float = 50.0,
    split_fingerprint: str | None = None,
    coordinates_fingerprint: str | None = None,
) -> SpatialGraph:
    """Build an exact-ID-aligned, directed kNN graph for one allowed partition.

    The caller must select only train OR only validation rows for ShamL.
    Never concatenate train/validation or cells from different slices here.
    Neighbors are sorted by distance, then stable Cell ID for tie-breaking.
    """
    ids = tuple(cell_ids)
    n_total = len(ids)
    if n_total == 0 or len(set(ids)) != n_total or any(
        not isinstance(cid, str) or not cid or cid.strip() != cid for cid in ids
    ):
        raise ValueError("Expected a nonempty sample with unique valid Cell IDs.")
    xy = np.asarray(coordinates_um)
    if xy.shape != (n_total, 2):
        raise ValueError("Coordinates must have shape [all_sample_cells, 2].")
    try:
        xy = np.ascontiguousarray(xy, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Coordinates must be numeric.") from error
    if not np.isfinite(xy).all():
        raise ValueError("Coordinates must be finite.")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")
    if isinstance(max_distance_um, bool) or not isinstance(max_distance_um, (float, int)) or not math.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("max_distance_um must be finite and positive.")
    selected = _as_indices(node_indices, n_total)
    sub_ids = tuple(ids[int(i)] for i in selected)
    sub_ids_array = np.asarray(sub_ids)
    sub_xy = np.ascontiguousarray(xy[selected], dtype=np.float64)
    num_nodes = len(sub_ids)

    order = np.argsort(sub_ids_array, kind="stable")
    canonical_xy = sub_xy[order]
    tree = cKDTree(canonical_xy)
    query_k = min(num_nodes, k + 2)
    dists, candidate = tree.query(
        sub_xy, k=query_k,
        distance_upper_bound=float(np.nextafter(float(max_distance_um), np.inf)),
        workers=1
    )
    dists = np.asarray(dists).reshape(num_nodes, query_k)
    candidate = np.asarray(candidate).reshape(num_nodes, query_k)

    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    neighbors = []
    distances = []
    for local_i in range(num_nodes):
        valid = (
            (candidate[local_i] < num_nodes)
            & np.isfinite(dists[local_i])
            & (dists[local_i] <= float(max_distance_um))
        )
        dest = order[candidate[local_i, valid]]
        dist = dists[local_i, valid]
        keep = dest != local_i
        dest, dist = dest[keep], dist[keep]
        if dest.size >= k + 1:
            # cKDTree need not include all tied neighbors in its truncated query.
            # Expand only rows whose k-th distance ties with the next candidate.
            ordered = np.sort(dist)
            boundary = ordered[k - 1]
            tie_tol = 1e-12 * max(1.0, float(boundary))
            if ordered[k] - boundary <= tie_tol:
                canonical_candidates = tree.query_ball_point(
                    sub_xy[local_i],
                    r=min(float(max_distance_um), float(boundary) + tie_tol),
                )
                expanded_dest = order[np.asarray(canonical_candidates, dtype=np.int64)]
                expanded_dist = np.linalg.norm(
                    sub_xy[expanded_dest] - sub_xy[local_i], axis=1
                )
                allowed = (
                    (expanded_dest != local_i)
                    & (expanded_dist <= float(max_distance_um))
                )
                dest, dist = expanded_dest[allowed], expanded_dist[allowed]
        if dest.size:
            # Sort by exact distance and then by stable Cell ID.
            sort_order = np.lexsort((sub_ids_array[dest], dist))
            dest = dest[sort_order][:k]
            dist = dist[sort_order][:k]
            neighbors.extend(dest.tolist())
            distances.extend(dist.tolist())
        indptr[local_i + 1] = len(neighbors)

    if coordinates_fingerprint is None:
        coordinates_fingerprint = _coordinate_fingerprint(ids, xy)
    return SpatialGraph(
        sample_name=sample_name,
        partition=partition,
        cell_ids=sub_ids,
        node_indices=selected,
        xy_um=sub_xy,
        indptr=indptr,
        neighbors=np.asarray(neighbors, dtype=np.int64),
        distances_um=np.asarray(distances, dtype=np.float64),
        k=k,
        max_distance_um=float(max_distance_um),
        coordinates_fingerprint=coordinates_fingerprint,
        split_fingerprint=split_fingerprint,
    )


def save_spatial_graph(graph: SpatialGraph, path: str | Path) -> None:
    """Write an unpickled NPZ graph and its provenance JSON; never overwrite."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Graph output must be an .npz file.")
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"Graph artifacts already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            node_indices=graph.node_indices,
            cell_ids=np.asarray(graph.cell_ids, dtype=np.str_),
            xy_um=graph.xy_um,
            indptr=graph.indptr,
            neighbors=graph.neighbors,
            distances_um=graph.distances_um,
        )
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(graph.summary(), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_spatial_graph(path: str | Path) -> SpatialGraph:
    """Load and validate a graph without pickle, including provenance hash."""
    path = Path(path)
    with path.with_suffix(".json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported spatial graph schema.")
    with np.load(path, allow_pickle=False) as archive:
        required = {"node_indices", "cell_ids", "xy_um", "indptr", "neighbors", "distances_um"}
        if set(archive.files) != required:
            raise ValueError("Spatial graph NPZ fields differ from the schema.")
        arrays = {key: archive[key] for key in required}
    graph = SpatialGraph(
        sample_name=meta["sample_name"],
        partition=meta["partition"],
        cell_ids=tuple(arrays["cell_ids"].tolist()),
        node_indices=arrays["node_indices"],
        xy_um=arrays["xy_um"],
        indptr=arrays["indptr"],
        neighbors=arrays["neighbors"],
        distances_um=arrays["distances_um"],
        k=int(meta["k"]),
        max_distance_um=float(meta["max_distance_um"]),
        coordinates_fingerprint=meta["coordinates_fingerprint"],
        split_fingerprint=meta["split_fingerprint"],
    )
    if graph.summary() != meta:
        raise ValueError("Graph data/metadata fingerprint or statistics mismatch.")
    return graph


def build_frozen_shaml_graph(
    *,
    split,
    cell_ids: Sequence[str],
    coordinates_um: np.ndarray,
    partition: str,
    k: int = 8,
    max_distance_um: float = 50.0,
) -> SpatialGraph:
    """Fail-closed ShamL train/validation builder tied to the frozen split.

    The caller cannot supply arbitrary training rows. Excluded buffer cells are
    never included, and each graph contains exactly one frozen partition.
    """
    if partition not in ("train", "validation"):
        raise ValueError("Only train or validation is allowed for a frozen ShamL graph.")
    if tuple(cell_ids) != tuple(split.cell_ids) or split.sample_name != "ShamL":
        raise ValueError("Frozen split and requested ShamL Cell IDs disagree.")
    actual_fingerprint = _coordinate_fingerprint(tuple(cell_ids), np.asarray(coordinates_um))
    if actual_fingerprint != split.coordinates_fingerprint:
        raise ValueError("ShamL coordinates differ from the frozen spatial split.")
    indices = split.train_indices if partition == "train" else split.validation_indices
    return build_spatial_knn_graph(
        cell_ids=cell_ids,
        coordinates_um=coordinates_um,
        node_indices=indices,
        sample_name="ShamL",
        partition=partition,
        k=k,
        max_distance_um=max_distance_um,
        split_fingerprint=split.fingerprint,
        coordinates_fingerprint=actual_fingerprint,
    )
