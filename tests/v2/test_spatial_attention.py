"""Numerical, graph-integrity, masking, and V1-compatibility tests for V2-B."""

import numpy as np
import pytest
import torch

from src.data.spatial_batches import make_one_hop_batch
from src.data.spatial_graph import build_spatial_knn_graph
from src.models.gene_aware_mae import GeneAwareMAE
from src.models.spatial_attention import DistanceAwareSpatialAttention
from src.models.spatial_backbone import SpatialContextBackbone
from scripts.v2.check_attention import main as smoke_main


def sample_graph():
    return build_spatial_knn_graph(
        cell_ids=("A", "B", "C", "isolated"),
        coordinates_um=np.asarray([[0, 0], [3, 0], [0, 4], [500, 0]], dtype=float),
        node_indices=np.asarray([0, 1, 2, 3]),
        sample_name="ShamR",
        partition="all",
        k=2,
        max_distance_um=5,
    )


def test_one_hop_batch_exact_h5_alignment():
    graph = build_spatial_knn_graph(
        cell_ids=("A", "B", "C", "D"),
        coordinates_um=np.asarray([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=float),
        node_indices=[2, 0, 3],
        sample_name="ShamL",
        partition="train",
        split_fingerprint="frozen",
        k=2,
        max_distance_um=3,
    )
    batch = make_one_hop_batch(graph, [1])
    assert batch.cell_ids == ("A", "C", "D")
    assert batch.h5_row_indices.tolist() == [0, 2, 3]
    assert batch.center_graph_indices.tolist() == [1]
    assert batch.edge_index.tolist() == [[0, 0], [1, 2]]
    assert batch.graph_fingerprint == graph.fingerprint
    t = batch.to_torch()
    assert t["center_index"].tolist() == [0]


def test_one_hop_batch_invalid_center_rejected():
    g = sample_graph()
    for selection in ([0, 0], [4], [-1], [], [0.5]):
        with pytest.raises((ValueError, IndexError)):
            make_one_hop_batch(g, selection)
    with pytest.raises(TypeError):
        make_one_hop_batch(object(), [0])


def test_batch_preserves_edges_and_isolation():
    g = sample_graph()
    batch = make_one_hop_batch(g, [3, 0])
    assert batch.num_centers == 2
    assert batch.num_nodes == 4
    assert batch.num_edges == 2
    assert batch.edge_index.tolist() == [[1, 1], [2, 3]]
    assert batch.edge_distances_um.tolist() == [3.0, 4.0]


def test_attention_segment_weights_are_normalized():
    torch.manual_seed(7)
    g = sample_graph()
    b = make_one_hop_batch(g, [0, 1, 3])
    attn = DistanceAwareSpatialAttention(cell_dim=12, num_heads=3, max_distance_um=5)
    attn.eval()
    t = b.to_torch()
    context, weights = attn(torch.randn(b.num_nodes, 12), t["edge_index"], t["edge_distances_um"], t["center_index"], return_attention=True)
    assert context.shape == (3, 12)
    assert weights.shape == (b.num_edges, 3)
    for i in range(b.num_centers):
        chosen = t["edge_index"][0] == i
        if chosen.any():
            torch.testing.assert_close(weights[chosen].sum(0), torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(context[2], torch.zeros(12), atol=0, rtol=0)


def test_distance_bias_prefers_nearer_equal_content():
    attn = DistanceAwareSpatialAttention(cell_dim=8, num_heads=2, max_distance_um=50)
    attn.eval()
    feats = torch.ones(3, 8)
    centers = torch.tensor([0], dtype=torch.long)
    edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    _, weights = attn(feats, edges, torch.tensor([5., 40.]), centers, return_attention=True)
    assert torch.all(weights[0] > weights[1])
    assert torch.all(weights[0] > 0.5)


def test_attention_one_hop_no_non_neighbor_influence():
    torch.manual_seed(1)
    attn = DistanceAwareSpatialAttention(8, 2, 50)
    attn.eval()
    features = torch.randn(4, 8)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    distances = torch.tensor([3.0])
    centers = torch.tensor([0])
    baseline = attn(features, edges, distances, centers)
    modified = features.clone()
    modified[2:] += 1000
    torch.testing.assert_close(attn(modified, edges, distances, centers), baseline)


def test_attention_gradient_and_extreme_logits_finite():
    torch.manual_seed(2)
    attention = DistanceAwareSpatialAttention(8, 2, 50)
    features = (torch.randn(6, 8) * 1000).requires_grad_(True)
    edges = torch.tensor([[0, 0, 0, 3, 3], [1, 2, 4, 2, 5]], dtype=torch.long)
    dist = torch.tensor([1., 40., 50., 0., 10.])
    centers = torch.tensor([0, 3], dtype=torch.long)
    out = attention(features, edges, dist, centers)
    out.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters())


def test_all_isolated_context_zero_and_differentiable():
    attention = DistanceAwareSpatialAttention(8, 2, 50)
    features = torch.randn(3, 8, requires_grad=True)
    centers = torch.tensor([2, 0])
    context, weights = attention(features, torch.empty(2, 0, dtype=torch.long), torch.empty(0), centers, return_attention=True)
    assert weights.shape == (0, 2)
    torch.testing.assert_close(context, torch.zeros(2, 8), atol=0, rtol=0)
    context.sum().backward()
    assert features.grad is not None


def test_malformed_graph_inputs_fail_closed():
    attn = DistanceAwareSpatialAttention(8, 2, 50)
    feats = torch.randn(3, 8)
    center = torch.tensor([0])
    edge = torch.tensor([[0], [1]])
    for bad_edge, bad_dist, bad_centers in (
        (torch.tensor([[0], [3]]), torch.tensor([2.]), center),
        (torch.tensor([[1], [2]]), torch.tensor([2.]), center),
        (torch.tensor([[0], [0]]), torch.tensor([0.]), center),
        (edge, torch.tensor([51.]), center),
        (edge, torch.tensor([float('nan')]), center),
        (edge, torch.tensor([-1.]), center),
        (edge, torch.tensor([1.]), torch.tensor([0, 0])),
    ):
        with pytest.raises((ValueError, IndexError)):
            attn(feats, bad_edge, bad_dist, bad_centers)


def test_batch_local_permutation_invariance():
    torch.manual_seed(10)
    attn = DistanceAwareSpatialAttention(12, 3, 50)
    attn.eval()
    feats = torch.randn(5, 12)
    edge = torch.tensor([[0, 0, 2], [1, 3, 4]])
    center = torch.tensor([0, 2])
    dists = torch.tensor([1., 20., 4.])
    first = attn(feats, edge, dists, center)
    order = torch.tensor([3, 1, 4, 0, 2])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    remapped = inverse[edge]
    second = attn(feats[order], remapped, dists, inverse[center])
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_masked_target_values_do_not_affect_spatial_context():
    torch.manual_seed(21)
    v1 = GeneAwareMAE(vocab_size=8, token_dim=16, cell_dim=16, num_latents=2, num_heads=4, dropout=0.0)
    attention = DistanceAwareSpatialAttention(16, 4, 50)
    backbone = SpatialContextBackbone(v1, attention).eval()
    expression = torch.tensor([
        [0., 1., 5., 2., 0., 3.],
        [1., 3., 0., 0., 6., 2.],
        [2., 5., 8., 1., 0., 0.],
    ])
    measured = torch.ones_like(expression, dtype=torch.bool)
    reconstruction = torch.tensor([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.bool)
    base_args = dict(
        gene_ids=torch.arange(6), measured_mask=measured, reconstruction_mask=reconstruction,
        center_index=torch.tensor([0, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_distances_um=torch.tensor([5., 5.]),
    )
    first = backbone(expression=expression, **base_args)
    altered = expression.clone()
    altered[reconstruction] += 100
    second = backbone(expression=altered, **base_args)
    torch.testing.assert_close(first["intrinsic_cell_embeddings"], second["intrinsic_cell_embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(first["spatial_context_embeddings"], second["spatial_context_embeddings"], rtol=0, atol=0)
    torch.testing.assert_close(first["v1_presence_logits"], second["v1_presence_logits"], rtol=0, atol=0)
    assert not torch.equal(first["target_counts"], second["target_counts"])


def test_spatial_backbone_keeps_v1_logits_unchanged():
    torch.manual_seed(34)
    v1 = GeneAwareMAE(vocab_size=7, token_dim=8, cell_dim=8, num_latents=2, num_heads=2, dropout=0.0)
    backbone = SpatialContextBackbone(v1, DistanceAwareSpatialAttention(8, 2, 50)).eval()
    expression = torch.rand(3, 5)
    masks = torch.ones(3, 5, dtype=torch.bool)
    hide = torch.zeros_like(masks)
    hide[:, 1] = True
    base = v1(expression, torch.arange(5), masks, hide)
    answer = backbone(
        expression=expression, gene_ids=torch.arange(5), measured_mask=masks,
        reconstruction_mask=hide, center_index=torch.tensor([0, 2]),
        edge_index=torch.tensor([[0, 2], [1, 1]]), edge_distances_um=torch.tensor([5., 4.]),
    )
    torch.testing.assert_close(answer["v1_presence_logits"], base["presence_logits"][[0, 2]])
    torch.testing.assert_close(answer["v1_positive_intensity"], base["positive_intensity"][[0, 2]])
    torch.testing.assert_close(answer["intrinsic_cell_embeddings"], base["cell_embeddings"][[0, 2]])


def test_attention_validation_configuration():
    for kwargs in (dict(cell_dim=7, num_heads=2), dict(max_distance_um=0), dict(num_heads=0), dict(attention_dropout=1)):
        with pytest.raises(ValueError):
            DistanceAwareSpatialAttention(**kwargs)
    with pytest.raises(ValueError):
        SpatialContextBackbone(
            GeneAwareMAE(vocab_size=5, token_dim=8, cell_dim=8, num_latents=2, num_heads=2),
            DistanceAwareSpatialAttention(12, 3),
        )


def test_cli_synthetic_smoke():
    result = smoke_main(["--centers", "5", "--cell-dim", "16", "--heads", "4", "--device", "cpu"])
    assert result["status"] == "passed"
    assert result["spatial_context_shape"] == [5, 16]


def test_cuda_smoke_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    g = sample_graph()
    b = make_one_hop_batch(g, [0, 1])
    tensors = b.to_torch("cuda")
    attn = DistanceAwareSpatialAttention(16, 4, 5).cuda()
    x = torch.randn(b.num_nodes, 16, device="cuda", requires_grad=True)
    output = attn(x, tensors["edge_index"], tensors["edge_distances_um"], tensors["center_index"])
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_read_only_backbone_cli_integration_with_mocked_frozen_inputs(tmp_path, monkeypatch):
    """Exercise checkpoint-to-context wiring while keeping all real data untouched."""
    from types import SimpleNamespace
    from scipy import sparse
    from src.data.spatial_graph import save_spatial_graph
    import scripts.v2.check_backbone as script

    graph = build_spatial_knn_graph(
        cell_ids=("a", "b", "c", "d"),
        coordinates_um=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        node_indices=np.arange(4),
        sample_name="ShamL", partition="train", split_fingerprint="split123",
        k=2, max_distance_um=5,
    )
    graph_path = tmp_path / "graph" / "train.npz"
    save_spatial_graph(graph, graph_path)
    class MockVocabulary:
        fingerprint = "vocab456"

        def __len__(self):
            return 5

    vocabulary = MockVocabulary()
    sample = SimpleNamespace(
        cell_ids=list(graph.cell_ids),
        expression=sparse.csr_matrix(np.array([
            [1., 2., 0., 1., 4.], [0., 1., 2., 1., 0.],
            [2., 0., 1., 3., 1.], [3., 1., 0., 2., 1.],
        ], dtype=np.float32)),
        gene_ids=torch.arange(5, dtype=torch.long),
    )
    split = SimpleNamespace(
        train_indices=np.arange(4), validation_indices=np.empty(0, dtype=np.int64),
        fingerprint="split123", coordinates_fingerprint=graph.coordinates_fingerprint,
    )
    model = GeneAwareMAE(vocab_size=5, token_dim=8, cell_dim=8, num_latents=2, num_heads=2, dropout=0)
    monkeypatch.setattr(script.GeneVocabulary, "load", lambda *args: vocabulary)
    monkeypatch.setattr(script, "find_xenium_sample_file", lambda *args: tmp_path / "ShamL" / "cell_feature_matrix.h5")
    monkeypatch.setattr(script, "load_sgmae_sample", lambda *args: sample)
    monkeypatch.setattr(script, "load_spatial_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(script, "sha256_file", lambda *args: "mock_h5_sha")
    monkeypatch.setattr(script, "load_frozen_best", lambda *args, **kwargs: (model, {"best_epoch": 10}, {}))
    result = script.main([
        "--graph", str(graph_path), "--centers", "2", "--heads", "2", "--device", "cpu",
    ])
    assert result["status"] == "passed"
    assert result["intrinsic_embedding_shape"] == [2, 8]
    assert result["spatial_context_shape"] == [2, 8]
