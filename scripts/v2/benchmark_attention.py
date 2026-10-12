"""Measure spatial-attention forward/backward cost with a V2-A graph.

This is a kernel-level benchmark using synthetic latent vectors, NOT a V1
encoder throughput measurement and NOT an end-to-end Xenium training run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.spatial_batches import make_one_hop_batch, validate_one_hop_batch
from src.data.spatial_graph import build_spatial_knn_graph, load_spatial_graph
from src.models.spatial_attention import DistanceAwareSpatialAttention


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="SG-MAE V2-B.1 attention kernel benchmark")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--centers", type=int, default=128)
    parser.add_argument("--cell-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--distance-gate", choices=("none", "exponential"), default="none")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.centers <= 0 or args.steps < 1 or args.warmup < 0:
        parser.error("--centers/--steps must be positive, --warmup nonnegative")
    if args.cell_dim <= 0 or args.heads <= 0 or args.cell_dim % args.heads:
        parser.error("--cell-dim must be positive and divisible by --heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if args.graph is None:
        ids = [f"cell_{i}" for i in range(1500)]
        xy = np.stack((np.arange(len(ids), dtype=np.float64) * 5.0, np.zeros(len(ids))), axis=1)
        graph = build_spatial_knn_graph(
            cell_ids=ids, coordinates_um=xy, node_indices=np.arange(len(ids)),
            sample_name="Synthetic", partition="all", k=8, max_distance_um=50,
        )
        source = "synthetic_graph"
    else:
        graph = load_spatial_graph(args.graph)
        source = str(args.graph)
    rng = np.random.default_rng(args.seed)
    centers = rng.choice(graph.num_nodes, size=min(args.centers, graph.num_nodes), replace=False)
    batch = make_one_hop_batch(graph, centers)
    validate_one_hop_batch(batch, graph)
    topology = batch.to_torch(device)
    torch.manual_seed(args.seed)
    embeddings = torch.randn(batch.num_nodes, args.cell_dim, device=device, requires_grad=True)
    layer = DistanceAwareSpatialAttention(
        cell_dim=args.cell_dim, num_heads=args.heads,
        max_distance_um=graph.max_distance_um,
        distance_gate=args.distance_gate, max_neighbors=graph.k,
    ).to(device)

    def step(*, strict: bool) -> None:
        layer.zero_grad(set_to_none=True)
        embeddings.grad = None
        result = layer(
            embeddings, topology["edge_index"], topology["edge_distances_um"],
            topology["center_index"], validate_topology=strict,
        )
        result.square().mean().backward()

    # Check every tensor once before timing repeated calls on immutable topology.
    step(strict=True)
    for _ in range(args.warmup):
        step(strict=False)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(args.steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        step(strict=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)
    result = {
        "status": "passed", "graph_source": source,
        "device": device, "distance_gate": args.distance_gate,
        "centers": batch.num_centers, "nodes_with_halo": batch.num_nodes,
        "edges": batch.num_edges, "warmup": args.warmup, "steps": args.steps,
        "median_step_ms": float(np.median(times_ms)),
        "mean_step_ms": float(np.mean(times_ms)),
        "centers_per_second": float(batch.num_centers * 1000 / np.mean(times_ms)),
        "cuda_peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device == "cuda" else None
        ),
        "scope": "spatial_attention_forward_and_backward_only_excludes_v1_encoder_and_data_loading",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
