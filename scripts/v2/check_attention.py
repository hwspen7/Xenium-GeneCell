"""V2.0-B sparse attention smoke test, with synthetic latent inputs.

The --graph option loads a real V2-A graph but does not load Xenium expression
or a model checkpoint. Results are engineering diagnostics, not predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.spatial_batches import make_one_hop_batch, validate_one_hop_batch
from src.data.spatial_graph import build_spatial_knn_graph, load_spatial_graph
from src.models.spatial_attention import DistanceAwareSpatialAttention
from src.models.spatial_diagnostics import summarize_spatial_attention


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="SG-MAE V2.0-B graph attention smoke test")
    parser.add_argument("--graph", type=Path, default=None, help="Optional V2-A .npz graph")
    parser.add_argument("--centers", type=int, default=16)
    parser.add_argument("--cell-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--distance-gate", choices=("none", "exponential"), default="none")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.centers <= 0:
        parser.error("--centers must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if args.graph is None:
        ids = [f"cell_{i:04d}" for i in range(40)]
        coordinates = np.stack((np.arange(40, dtype=np.float64) * 5, np.zeros(40)), axis=1)
        graph = build_spatial_knn_graph(
            cell_ids=ids,
            coordinates_um=coordinates,
            node_indices=np.arange(40),
            sample_name="Synthetic",
            partition="all",
            k=8,
            max_distance_um=50,
        )
        source = "synthetic_graph"
    else:
        graph = load_spatial_graph(args.graph)
        source = str(args.graph)

    rng = np.random.default_rng(args.seed)
    centers = rng.choice(graph.num_nodes, size=min(args.centers, graph.num_nodes), replace=False)
    batch = make_one_hop_batch(graph, centers)
    validate_one_hop_batch(batch, graph)
    tensors = batch.to_torch(device)
    torch.manual_seed(args.seed)
    features = torch.randn(batch.num_nodes, args.cell_dim, device=device, requires_grad=True)
    attention = DistanceAwareSpatialAttention(
        cell_dim=args.cell_dim,
        num_heads=args.heads,
        max_distance_um=graph.max_distance_um,
        distance_gate=args.distance_gate,
        max_neighbors=graph.k,
    ).to(device)
    attention.eval()
    context, weights = attention(
        features,
        tensors["edge_index"],
        tensors["edge_distances_um"],
        tensors["center_index"],
        return_attention=True,
    )
    (context.square().sum() + features.square().sum() * 0).backward()
    assert context.shape == (batch.num_centers, args.cell_dim)
    assert torch.isfinite(context).all() and torch.isfinite(weights).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    diagnostics = summarize_spatial_attention(
        weights, tensors["edge_index"], tensors["edge_distances_um"],
        tensors["center_index"], num_nodes=batch.num_nodes,
    )
    active = diagnostics["active_center_mask"]
    result = {
        "status": "passed",
        "input_type": "random_cell_embeddings_no_checkpoint",
        "graph_source": source,
        "sample": graph.sample_name,
        "partition": graph.partition,
        "graph_fingerprint": graph.fingerprint,
        "centers": batch.num_centers,
        "total_nodes_with_halo": batch.num_nodes,
        "edges": batch.num_edges,
        "spatial_context_shape": list(context.shape),
        "spatial_context_norm_mean": float(context.detach().norm(dim=1).mean().cpu()),
        "device": device,
        "distance_gate": args.distance_gate,
        "isolated_centers": int((~active).sum().cpu()),
        "mean_effective_neighbors": (
            float(diagnostics["effective_neighbors"][active].mean().cpu())
            if active.any() else 0.0
        ),
        "mean_attended_distance_um": (
            float(diagnostics["attended_distance_um"][active].mean().cpu())
            if active.any() else 0.0
        ),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
