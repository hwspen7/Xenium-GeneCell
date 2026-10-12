"""Read-only V2.0-B integration check with a frozen ShamL V1 checkpoint.

This checks alignment, V1 checkpoint provenance and sparse attention backward.
It does not train V2, change V1 logits, or save any biological predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.gene_vocabulary import GeneVocabulary
from src.data.masking import generate_reconstruction_mask
from src.data.spatial_batches import make_one_hop_batch, validate_one_hop_batch
from src.data.spatial_graph import load_spatial_graph
from src.data.splitting import load_spatial_split
from src.dataset import find_xenium_sample_file, load_sgmae_sample
from src.evaluation.cell_embeddings import sha256_file
from src.evaluation.evaluate_sgmae_v1 import load_frozen_best
from src.models.spatial_attention import DistanceAwareSpatialAttention
from src.models.spatial_backbone import SpatialContextBackbone


def main(argv: list[str] | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="SG-MAE V2.0-B frozen V1 ShamL integration check")
    parser.add_argument("--graph", type=Path, default=root / "runs/v2/graphs/shaml_k8_r50um/shaml/train.npz")
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/v1/training/scheduler/best.pt")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs/gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs/splits/shaml_spatial_v1.json.gz")
    parser.add_argument("--centers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--distance-gate", choices=("none", "exponential"), default="none")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.centers < 1 or not (0 < args.mask_ratio < 1):
        parser.error("--centers must be positive and --mask-ratio must lie in (0,1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    graph = load_spatial_graph(args.graph)
    if graph.sample_name != "ShamL" or graph.partition not in ("train", "validation"):
        raise ValueError("The V2-B checkpoint smoke test accepts frozen ShamL graphs only")

    vocabulary = GeneVocabulary.load(args.vocabulary)
    sample_path = find_xenium_sample_file(args.data_root, "ShamL")
    dataset = load_sgmae_sample(sample_path, vocabulary)
    split = load_spatial_split(
        args.split, dataset.cell_ids, sample_path.parent / "cells.parquet",
        expected_sample_name="ShamL",
    )
    expected = split.train_indices if graph.partition == "train" else split.validation_indices
    if not np.array_equal(graph.node_indices, expected):
        raise ValueError("Graph node indices differ from the frozen ShamL partition")
    if tuple(dataset.cell_ids[i] for i in graph.node_indices) != graph.cell_ids:
        raise ValueError("Graph Cell IDs do not align with H5 counts")
    if graph.split_fingerprint != split.fingerprint or graph.coordinates_fingerprint != split.coordinates_fingerprint:
        raise ValueError("Graph spatial/split fingerprints differ from the frozen ShamL split")
    if len(dataset.gene_ids) < 2:
        raise ValueError("At least two measured genes are needed for masked reconstruction")

    model, checkpoint, metadata = load_frozen_best(
        args.checkpoint,
        vocabulary_fingerprint=vocabulary.fingerprint,
        split_fingerprint=split.fingerprint,
        sham_h5_sha256=sha256_file(sample_path),
        vocabulary_size=len(vocabulary),
        device=device,
    )
    model.eval().requires_grad_(False)
    torch.manual_seed(args.seed)
    attention = DistanceAwareSpatialAttention(
        cell_dim=model.cell_projection.out_features,
        num_heads=args.heads,
        max_distance_um=graph.max_distance_um,
        distance_gate=args.distance_gate,
        max_neighbors=graph.k,
    ).to(device)
    backbone = SpatialContextBackbone(model, attention).eval()
    rng = np.random.default_rng(args.seed)
    centers = rng.choice(graph.num_nodes, size=min(args.centers, graph.num_nodes), replace=False)
    batch = make_one_hop_batch(graph, centers)
    validate_one_hop_batch(batch, graph)
    topology = batch.to_torch(device)
    raw = dataset.expression[batch.h5_row_indices, :].toarray().astype(np.float32, copy=False)
    expression = torch.from_numpy(raw).to(device)
    gene_ids = dataset.gene_ids.to(device)
    measured = torch.ones_like(expression, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    masked = generate_reconstruction_mask(
        measured,
        mask_ratio=args.mask_ratio,
        gene_ids=gene_ids,
        generator=generator,
        ensure_visible=True,
    )
    outputs = backbone(
        expression=expression,
        gene_ids=gene_ids,
        measured_mask=measured,
        reconstruction_mask=masked,
        center_index=topology["center_index"],
        edge_index=topology["edge_index"],
        edge_distances_um=topology["edge_distances_um"],
    )
    context = outputs["spatial_context_embeddings"]
    if not torch.isfinite(context).all().item():
        raise ValueError("Nonfinite spatial context in the integrated V1/V2-B forward")
    context.square().mean().backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in attention.parameters()):
        raise ValueError("Nonfinite spatial attention gradients")
    report = {
        "status": "passed",
        "mode": "frozen_v1_real_shaml_checkpoint_no_v2_training",
        "graph": str(args.graph),
        "graph_fingerprint": graph.fingerprint,
        "graph_partition": graph.partition,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "centers": batch.num_centers,
        "nodes_including_halo": batch.num_nodes,
        "edges": batch.num_edges,
        "intrinsic_embedding_shape": list(outputs["intrinsic_cell_embeddings"].shape),
        "spatial_context_shape": list(context.shape),
        "device": device,
        "distance_gate": args.distance_gate,
        "mask_protocol": "src.data.masking.generate_reconstruction_mask",
        "mask_ratio": args.mask_ratio,
        "masked_gene_counts": masked.sum(dim=1).tolist(),
        "note": "V1 reconstruction logits remain intrinsic-only; V2-C will add fused decoding",
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
