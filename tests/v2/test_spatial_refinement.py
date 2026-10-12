"""Regression tests for V2.0-B.1 hardening without changing frozen V1."""

import math

import numpy as np
import pytest
import torch

from src.data.masking import generate_reconstruction_mask
from src.data.spatial_batches import make_one_hop_batch, validate_one_hop_batch
from src.data.spatial_graph import build_spatial_knn_graph
from src.models.spatial_attention import DistanceAwareSpatialAttention
from src.models.spatial_backbone import SpatialContextBackbone
from src.models.spatial_diagnostics import summarize_spatial_attention
from src.models.gene_aware_mae import GeneAwareMAE
from scripts.v2.benchmark_attention import main as benchmark_main
from scripts.v2.check_attention import main as check_attention_main


def small_graph():
    return build_spatial_knn_graph(
        cell_ids=("a", "b", "c", "d", "e"),
        coordinates_um=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [100., 100.]]),
        node_indices=np.array([3, 1, 4, 0, 2]),
        sample_name="Synthetic", partition="all", k=2, max_distance_um=3,
    )


def test_default_distance_gate_is_backward_compatible():
    baseline = DistanceAwareSpatialAttention(8, 2, 50)
    explicit = DistanceAwareSpatialAttention(8, 2, 50, distance_gate="none")
    explicit.load_state_dict(baseline.state_dict(), strict=True)
    assert "raw_distance_gate_decay" not in baseline.state_dict()
    embeddings = torch.randn(4, 8)
    edge = torch.tensor([[0, 0], [1, 2]])
    distances = torch.tensor([2., 19.])
    torch.testing.assert_close(
        baseline(embeddings, edge, distances, torch.tensor([0])),
        explicit(embeddings, edge, distances, torch.tensor([0])),
    )


def test_exponential_gate_attenuates_one_neighbor_by_absolute_distance():
    attn = DistanceAwareSpatialAttention(8, 2, 50, distance_gate="exponential").eval()
    features = torch.randn(2, 8)
    edge = torch.tensor([[0], [1]])
    center = torch.tensor([0])
    near, near_weights = attn(features, edge, torch.tensor([5.]), center, return_attention=True)
    far, far_weights = attn(features, edge, torch.tensor([45.]), center, return_attention=True)
    torch.testing.assert_close(near_weights, far_weights)
    torch.testing.assert_close(near_weights, torch.ones_like(near_weights))
    assert near.norm() > far.norm()
    ratio = math.exp(-(45 - 5) / 50)
    torch.testing.assert_close(far, near * ratio, atol=1e-6, rtol=1e-5)


def test_exponential_gate_has_finite_gradient_and_zero_for_isolated():
    attn = DistanceAwareSpatialAttention(8, 2, 50, distance_gate="exponential")
    feats = torch.randn(4, 8, requires_grad=True)
    edge = torch.tensor([[0, 0], [1, 2]])
    context = attn(feats, edge, torch.tensor([5., 40.]), torch.tensor([0, 3]))
    torch.testing.assert_close(context[1], torch.zeros_like(context[1]), atol=0, rtol=0)
    context.square().sum().backward()
    assert attn.raw_distance_gate_decay.grad is not None
    assert torch.isfinite(attn.raw_distance_gate_decay.grad).all()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


def test_duplicate_spatial_edges_fail_closed():
    attn = DistanceAwareSpatialAttention(8, 2, 50)
    with pytest.raises(ValueError, match="Duplicate spatial edges"):
        attn(torch.randn(3, 8), torch.tensor([[0, 0], [1, 1]]),
             torch.tensor([3., 3.]), torch.tensor([0]))


def test_over_degree_graph_fails_closed_when_k_configured():
    attn = DistanceAwareSpatialAttention(8, 2, 50, max_neighbors=1)
    with pytest.raises(ValueError, match="max_neighbors"):
        attn(torch.randn(3, 8), torch.tensor([[0, 0], [1, 2]]),
             torch.tensor([3., 4.]), torch.tensor([0]))


@pytest.mark.parametrize("kwargs", [
    {"distance_gate": "quadratic"},
    {"max_neighbors": 0},
    {"max_neighbors": True},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        DistanceAwareSpatialAttention(8, 2, 50, **kwargs)


def test_fast_path_matches_strict_path_output_and_gradients():
    torch.manual_seed(12)
    attn = DistanceAwareSpatialAttention(8, 2, 50, distance_gate="exponential")
    x = torch.randn(4, 8, requires_grad=True)
    e = torch.tensor([[0, 0, 2], [1, 3, 1]])
    d = torch.tensor([2., 4., 8.])
    c = torch.tensor([0, 2])
    result1 = attn(x, e, d, c, validate_topology=True)
    grad1 = torch.autograd.grad(result1.square().sum(), x, retain_graph=True)[0]
    result2 = attn(x, e, d, c, validate_topology=False)
    grad2 = torch.autograd.grad(result2.square().sum(), x)[0]
    torch.testing.assert_close(result1, result2)
    torch.testing.assert_close(grad1, grad2)


def test_diagnostics_known_distribution_and_empty_center():
    weights = torch.tensor([[0.5, 0.75], [0.5, 0.25], [1.0, 1.0]])
    edge = torch.tensor([[2, 2, 0], [1, 3, 2]])
    centers = torch.tensor([0, 2, 4])
    d = torch.tensor([10., 30., 5.])
    diag = summarize_spatial_attention(weights, edge, d, centers, num_nodes=5)
    assert diag["degree"].tolist() == [1, 2, 0]
    torch.testing.assert_close(diag["entropy"][0], torch.zeros(2))
    torch.testing.assert_close(diag["effective_neighbors"][0], torch.ones(2))
    torch.testing.assert_close(diag["effective_neighbors"][2], torch.zeros(2))
    torch.testing.assert_close(diag["normalized_entropy"][1, 0], torch.tensor(1.))
    torch.testing.assert_close(diag["attended_distance_um"][1], torch.tensor([20., 15.]))
    assert not diag["entropy"].requires_grad


def test_diagnostics_no_edges_are_zero():
    d = summarize_spatial_attention(
        torch.empty(0, 2), torch.empty(2, 0, dtype=torch.long),
        torch.empty(0), torch.tensor([1, 0]), num_nodes=2,
    )
    assert torch.equal(d["degree"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(d["entropy"], torch.zeros(2, 2))
    assert torch.equal(d["effective_neighbors"], torch.zeros(2, 2))


def test_diagnostics_are_detached_from_gradients():
    w = torch.tensor([[0.25], [0.75]], requires_grad=True)
    d = summarize_spatial_attention(w, torch.tensor([[0, 0], [1, 2]]),
                                    torch.tensor([2., 4.]), torch.tensor([0]), num_nodes=3)
    assert all(not tensor.requires_grad for tensor in d.values())


def test_graph_batch_exact_alignment():
    g = small_graph()
    b = make_one_hop_batch(g, [3, 2, 0])
    validate_one_hop_batch(b, g)
    assert b.h5_row_indices.tolist() == g.node_indices[b.graph_node_indices].tolist()


@pytest.mark.parametrize("field", ["h5_row_indices", "edge_index", "edge_distances_um", "cell_ids", "graph_fingerprint"])
def test_graph_batch_tampering_is_detected(field):
    from dataclasses import replace
    g = small_graph()
    b = make_one_hop_batch(g, [0, 1])
    if field == "cell_ids":
        tampered = replace(b, cell_ids=("wrong",) + b.cell_ids[1:])
    elif field == "graph_fingerprint":
        tampered = replace(b, graph_fingerprint="incorrect")
    else:
        data = getattr(b, field).copy()
        if field == "h5_row_indices":
            data[0] += 200
        elif field == "edge_index":
            data[1, 0] = 0
        else:
            data[0] += 1
        tampered = replace(b, **{field: data})
    with pytest.raises(ValueError):
        validate_one_hop_batch(tampered, g)


def test_v1_mask_protocol_covers_center_and_halo():
    measured = torch.tensor([[True, True, True, False], [True, True, True, True],
                             [True, True, False, False]])
    ids = torch.tensor([0, 1, 2, 3])
    seed = 47
    m1 = generate_reconstruction_mask(measured, mask_ratio=0.2, gene_ids=ids,
                                      generator=torch.Generator().manual_seed(seed))
    m2 = generate_reconstruction_mask(measured, mask_ratio=0.2, gene_ids=ids,
                                      generator=torch.Generator().manual_seed(seed))
    assert torch.equal(m1, m2)
    assert not (m1 & ~measured).any()
    assert all(0 < m1[i].sum() < measured[i].sum() for i in range(3))


def test_hidden_halo_counts_cannot_change_spatial_context():
    torch.manual_seed(20)
    v1 = GeneAwareMAE(vocab_size=6, token_dim=8, cell_dim=8,
                      num_latents=2, num_heads=2, dropout=0.)
    backbone = SpatialContextBackbone(
        v1, DistanceAwareSpatialAttention(8, 2, 50, distance_gate="exponential")
    ).eval()
    expression = torch.rand(3, 5)
    measured = torch.ones_like(expression, dtype=torch.bool)
    mask = generate_reconstruction_mask(
        measured, mask_ratio=0.2, gene_ids=torch.arange(5),
        generator=torch.Generator().manual_seed(7),
    )
    params = dict(gene_ids=torch.arange(5), measured_mask=measured,
                  reconstruction_mask=mask, center_index=torch.tensor([0]),
                  edge_index=torch.tensor([[0, 0], [1, 2]]),
                  edge_distances_um=torch.tensor([2., 3.]))
    a = backbone(expression=expression, **params)
    changed = expression.clone()
    changed[mask] += 900
    b = backbone(expression=changed, **params)
    torch.testing.assert_close(a["spatial_context_embeddings"], b["spatial_context_embeddings"], atol=0, rtol=0)
    torch.testing.assert_close(a["intrinsic_cell_embeddings"], b["intrinsic_cell_embeddings"], atol=0, rtol=0)


def test_cpu_benchmark_smoke():
    result = benchmark_main(["--centers", "4", "--steps", "2", "--warmup", "0",
                             "--cell-dim", "8", "--heads", "2", "--device", "cpu"])
    assert result["status"] == "passed"
    assert result["mean_step_ms"] > 0
    assert result["cuda_peak_allocated_mib"] is None


def test_smoke_cli_with_optional_gate():
    result = check_attention_main(["--centers", "4", "--cell-dim", "8",
                                   "--heads", "2", "--distance-gate", "exponential", "--device", "cpu"])
    assert result["status"] == "passed"
    assert result["distance_gate"] == "exponential"
    assert result["mean_effective_neighbors"] > 0
