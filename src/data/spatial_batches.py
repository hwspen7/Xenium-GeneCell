"""Deterministic, one-hop graph batches aligned to Xenium H5 cell rows.

A batch carries center nodes first, followed by unique neighbor (halo) nodes.
Only centers receive incoming edges. Halo expressions must be loaded using
``h5_row_indices`` and masked before passing to the V1 encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.data.spatial_graph import SpatialGraph


@dataclass(frozen=True)
class SpatialOneHopBatch:
    """Sparse one-hop edge list for center nodes, in local batch indexing.

    edge_index[0, e] is the receiving *center* row of this batch.
    edge_index[1, e] is its sending neighbor row of this batch.
    h5_row_indices identifies the H5 expression matrix rows to load in order.
    """

    center_graph_indices: np.ndarray
    graph_node_indices: np.ndarray
    h5_row_indices: np.ndarray
    cell_ids: tuple[str, ...]
    edge_index: np.ndarray
    edge_distances_um: np.ndarray
    graph_fingerprint: str
    sample_name: str
    partition: str

    @property
    def num_centers(self) -> int:
        return int(self.center_graph_indices.size)

    @property
    def num_nodes(self) -> int:
        return int(self.graph_node_indices.size)

    @property
    def num_edges(self) -> int:
        return int(self.edge_distances_um.size)

    def to_torch(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Return topology tensors. Centers always occupy local rows 0..C-1."""
        return {
            "center_index": torch.arange(self.num_centers, dtype=torch.long, device=device),
            "edge_index": torch.as_tensor(self.edge_index.copy(), dtype=torch.long, device=device),
            "edge_distances_um": torch.as_tensor(
                self.edge_distances_um.copy(), dtype=torch.float32, device=device
            ),
            "h5_row_indices": torch.as_tensor(
                self.h5_row_indices.copy(), dtype=torch.long, device=device
            ),
        }


def make_one_hop_batch(
    graph: SpatialGraph,
    center_graph_indices: Sequence[int] | np.ndarray,
) -> SpatialOneHopBatch:
    """Select a deterministic center/halo batch without crossing graph partitions.

    Center indices are zero-based *graph-local* rows, not H5 cell rows.
    Edges are copied only for those centers; halo nodes have no outgoing
    update requests. The batch contains no expression counts or labels.
    """
    if not isinstance(graph, SpatialGraph):
        raise TypeError("graph must be a validated SpatialGraph.")
    centers = np.asarray(center_graph_indices)
    if centers.ndim != 1 or centers.size == 0 or centers.dtype.kind not in "iu":
        raise ValueError("Center indices must be a nonempty 1D integer array.")
    if centers.dtype.kind == "u" and np.any(centers > np.iinfo(np.int64).max):
        raise ValueError("Center indices exceed int64 range.")
    centers = centers.astype(np.int64, copy=True)
    if np.any(centers < 0) or np.any(centers >= graph.num_nodes):
        raise IndexError("Center indices lie outside this graph partition.")
    if np.unique(centers).size != centers.size:
        raise ValueError("Center indices must be unique.")

    source_rows = []
    receiving_rows = []
    edge_distances = []
    for batch_row, graph_row in enumerate(centers):
        start = int(graph.indptr[graph_row])
        stop = int(graph.indptr[graph_row + 1])
        neighbor_rows = graph.neighbors[start:stop]
        source_rows.extend(neighbor_rows.tolist())
        receiving_rows.extend([batch_row] * (stop - start))
        edge_distances.extend(graph.distances_um[start:stop].tolist())

    center_set = set(centers.tolist())
    halo = np.asarray(sorted(set(source_rows) - center_set), dtype=np.int64)
    nodes = np.concatenate((centers, halo))
    local_lookup = {int(node): i for i, node in enumerate(nodes)}
    sources_local = np.fromiter(
        (local_lookup[int(row)] for row in source_rows), dtype=np.int64, count=len(source_rows)
    )
    receiving_local = np.asarray(receiving_rows, dtype=np.int64)
    edge_index = np.stack((receiving_local, sources_local), axis=0)
    return SpatialOneHopBatch(
        center_graph_indices=centers,
        graph_node_indices=nodes,
        h5_row_indices=np.asarray(graph.node_indices[nodes], dtype=np.int64),
        cell_ids=tuple(graph.cell_ids[i] for i in nodes),
        edge_index=edge_index,
        edge_distances_um=np.asarray(edge_distances, dtype=np.float32),
        graph_fingerprint=graph.fingerprint,
        sample_name=graph.sample_name,
        partition=graph.partition,
    )


def validate_one_hop_batch(batch: SpatialOneHopBatch, graph: SpatialGraph) -> None:
    """Check a batch against its source graph before cached CUDA execution.

    Validates exact node, H5-row, Cell-ID, distance and edge correspondences.
    Call again if either source graph or batch arrays might have been mutated.
    """
    if not isinstance(batch, SpatialOneHopBatch) or not isinstance(graph, SpatialGraph):
        raise TypeError("Expected a SpatialOneHopBatch and SpatialGraph.")
    if (batch.sample_name, batch.partition) != (graph.sample_name, graph.partition):
        raise ValueError("Batch belongs to another sample or partition.")
    if batch.graph_fingerprint != graph.fingerprint:
        raise ValueError("Batch provenance does not match the spatial graph.")
    c, n, e = batch.num_centers, batch.num_nodes, batch.num_edges
    if not (0 < c <= n) or batch.edge_index.shape != (2, e):
        raise ValueError("Batch has inconsistent center or edge dimensions.")
    if batch.graph_node_indices.shape != (n,) or batch.h5_row_indices.shape != (n,):
        raise ValueError("Batch graph and H5 row mappings are malformed.")
    centers = batch.center_graph_indices
    nodes = batch.graph_node_indices
    if not np.array_equal(nodes[:c], centers):
        raise ValueError("Center nodes are not first in the batch.")
    if np.any(nodes < 0) or np.any(nodes >= graph.num_nodes) or np.unique(nodes).size != n:
        raise ValueError("Graph node mapping contains invalid or repeated indices.")
    if not np.array_equal(batch.h5_row_indices, graph.node_indices[nodes]):
        raise ValueError("Batch H5 row indices do not match source graph.")
    if batch.cell_ids != tuple(graph.cell_ids[int(i)] for i in nodes):
        raise ValueError("Batch Cell IDs do not match source graph.")
    if (batch.edge_index.dtype.kind not in 'iu'
            or batch.edge_distances_um.dtype.kind != 'f'
            or not np.isfinite(batch.edge_distances_um).all()):
        raise ValueError("Invalid edge dtype or distance.")
    if (np.any(batch.edge_index < 0) or np.any(batch.edge_index[0] >= c)
            or np.any(batch.edge_index[1] >= n)):
        raise ValueError("An edge endpoint lies outside the batch.")
    lookup = {int(row): i for i, row in enumerate(nodes)}
    expected_receivers = []
    expected_sources = []
    expected_distances = []
    for local_center, graph_center in enumerate(centers):
        a, b = int(graph.indptr[graph_center]), int(graph.indptr[graph_center + 1])
        expected_receivers.extend([local_center] * (b - a))
        expected_sources.extend(lookup[int(i)] for i in graph.neighbors[a:b])
        expected_distances.extend(graph.distances_um[a:b])
    if not np.array_equal(batch.edge_index, np.asarray(
        [expected_receivers, expected_sources], dtype=np.int64).reshape(2, -1
    )):
        raise ValueError("Batch edges do not match source graph.")
    if not np.allclose(batch.edge_distances_um, expected_distances, atol=1e-5, rtol=1e-6):
        raise ValueError("Batch distances do not match source graph.")
