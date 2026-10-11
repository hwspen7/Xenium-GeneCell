"""CPU regression checks for SG-MAE V1.1 embedding extraction and plotting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
import types

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from src.evaluation.cell_embeddings import (
    _resolve_extraction_device,
    aligned_spatial_coordinates,
    encode_cells,
    save_sample_embedding,
    validate_cell_indices,
    verify_gene_panel,
)
from src.models.gene_aware_mae import GeneAwareMAE
from scripts.plot_cell_embeddings import main as plot_main, select_balanced_sample


class TinyDataset:
    def __init__(self, expression, gene_ids, *, label="test"):
        self.expression = csr_matrix(np.asarray(expression, dtype=np.float32))
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        self.measured_mask = torch.ones(len(gene_ids), dtype=torch.bool)
        self.cell_ids = [f"{label}-{i}" for i in range(self.expression.shape[0])]

    def __len__(self):
        return self.expression.shape[0]


def test_cuda_device_normalization():
    assert _resolve_extraction_device("cpu", torch.device("cpu")) == torch.device("cpu")
    assert _resolve_extraction_device("cuda", torch.device("cuda:0")) == torch.device("cuda:0")
    assert _resolve_extraction_device("cuda", torch.device("cuda:1")) == torch.device("cuda:1")
    assert _resolve_extraction_device("cuda:1", torch.device("cuda:1")) == torch.device("cuda:1")
    for requested, actual in (("cpu", "cuda:0"), ("cuda", "cpu"), ("cuda:0", "cuda:1")):
        try:
            _resolve_extraction_device(requested, torch.device(actual))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Device mismatch was accepted: {requested} vs {actual}")


def test_extract_exact_embeddings_and_row_alignment():
    torch.manual_seed(42)
    model = GeneAwareMAE(vocab_size=5, token_dim=16, cell_dim=12, num_latents=3, num_heads=4, dropout=0.0)
    model.eval()
    dataset = TinyDataset([[0, 1, 3], [2, 0, 0], [4, 6, 1], [0, 0, 0], [2, 3, 0]], [1, 4, 2])
    indices = np.array([4, 0, 3, 1], dtype=np.int64)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output, counts, genes = encode_cells(model, dataset, indices, batch_size=2)
    again, _, _ = encode_cells(model, dataset, indices, batch_size=3)
    assert output.shape == (4, 12)
    assert np.allclose(output, again, atol=1e-6, rtol=1e-6)
    assert np.array_equal(counts, [5, 4, 0, 2])
    assert np.array_equal(genes, [2, 2, 0, 1])
    with torch.inference_mode():
        raw = torch.as_tensor(dataset.expression[indices].toarray(), dtype=torch.float32)
        ids = dataset.gene_ids
        measured = torch.ones_like(raw, dtype=torch.bool)
        expected = model(raw, ids, measured, torch.zeros_like(measured), target_gene_ids=ids[:1])["cell_embeddings"]
        assert np.allclose(expected.numpy(), output, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"Model changed during inference: {key}"


def test_panel_permutation_and_validation():
    torch.manual_seed(123)
    model = GeneAwareMAE(vocab_size=5, token_dim=16, cell_dim=12, num_latents=3, num_heads=4, dropout=0.0).eval()
    counts = np.asarray([[1, 2, 0], [3, 0, 7], [0, 0, 0]], dtype=np.float32)
    original = TinyDataset(counts, [4, 0, 2])
    reordered = TinyDataset(counts[:, [2, 0, 1]], [2, 4, 0])
    verify_gene_panel(original.gene_ids, reordered.gene_ids)
    a, _, _ = encode_cells(model, original, np.arange(3), batch_size=2)
    b, _, _ = encode_cells(model, reordered, np.arange(3), batch_size=2)
    assert np.allclose(a, b, atol=1e-5, rtol=1e-5)
    for invalid in ([0, 0], [-1], [3], [0.5]):
        try:
            validate_cell_indices(invalid, 3)
            raise AssertionError(f"Invalid cell indices accepted: {invalid}")
        except ValueError:
            pass
    try:
        verify_gene_panel([0, 1, 2], [0, 1, 3])
        raise AssertionError("Different panels accepted")
    except ValueError:
        pass


def test_coordinates_and_output():
    dataset = TinyDataset([[0, 1], [3, 0], [1, 1]], [0, 1], label="cells")
    frame = pd.DataFrame({
        "cell_id": [b"cells-1", b"cells-2", b"cells-0"],
        "x_centroid": [21., 32., 10.],
        "y_centroid": [2., 3., 1.],
    })
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        (p / "cells.parquet").touch()
        with patch("src.evaluation.cell_embeddings.pd.read_parquet", return_value=frame):
            coords = aligned_spatial_coordinates(dataset.cell_ids, p / "cells.parquet")
        assert coords.tolist() == [[10, 1], [21, 2], [32, 3]]
        model = GeneAwareMAE(vocab_size=3, token_dim=16, cell_dim=12, num_latents=2, num_heads=4, dropout=0).eval()
        indices = np.asarray([2, 0])
        embedding, counts, detected = encode_cells(model, dataset, indices, batch_size=2)
        save_sample_embedding(
            p / "test", sample_key="shaml", dataset=dataset,
            indices=indices, coordinates=coords, embeddings=embedding,
            total_counts=counts, detected_genes=detected, provenance={"test": True},
        )
        recovered = np.load(p / "test/embeddings.npy")
        metadata = pd.read_csv(p / "test/cells.csv")
        assert np.array_equal(embedding, recovered)
        assert metadata["cell_id"].tolist() == ["cells-2", "cells-0"]
        assert metadata["x_centroid"].tolist() == [32, 10]
        assert json.loads((p / "test/manifest.json").read_text())["extraction_protocol"] == "full_measured_visible_no_mask"


def test_balanced_projection_pipeline():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "embeddings"
        root.mkdir()
        reference = {
            "checkpoint_sha256": "checkpoint",
            "vocabulary_fingerprint": "vocabulary",
            "split_fingerprint": "split",
            "encoding_protocol": "full_measured_visible_no_mask",
        }
        (root / "extraction_manifest.json").write_text(json.dumps(reference))
        rng = np.random.default_rng(73)
        sources = {}
        for i, sample in enumerate(("shaml", "shamr", "hour12r", "day2l")):
            folder = root / sample
            folder.mkdir()
            embeddings = rng.normal(size=(17 + i, 12)).astype(np.float32) + i * 0.05
            np.save(folder / "embeddings.npy", embeddings)
            metadata = pd.DataFrame({
                "sample": [sample] * len(embeddings),
                "row_index": np.arange(len(embeddings)),
                "cell_id": [f"{sample}-{j}" for j in range(len(embeddings))],
                "x_centroid": np.arange(len(embeddings)),
                "y_centroid": np.arange(len(embeddings)),
                "total_counts": np.ones(len(embeddings)) * (i + 1),
                "detected_genes": np.ones(len(embeddings)) * (i + 2),
            })
            metadata.to_csv(folder / "cells.csv", index=False)
            (folder / "manifest.json").write_text(json.dumps({
                "sample": sample, "cells": len(embeddings), "cell_dim": 12,
                "checkpoint_sha256": "checkpoint", "vocabulary_fingerprint": "vocabulary",
                "split_fingerprint": "split", "extraction_protocol": "full_measured_visible_no_mask",
            }))
            sources[sample] = {"metadata": metadata, "embedding": embeddings}
        _, balanced, k = select_balanced_sample(sources, per_sample=8, seed=42)
        assert k == 8 and len(balanced) == 32
        report = plot_main(["--input-dir", str(root), "--per-sample", "8", "--skip-umap"])
        plot_dir = root / "plots"
        assert (plot_dir / "pooled_pca_by_slice.png").is_file()
        assert not (plot_dir / "pooled_umap_by_slice.png").exists()
        plotted = pd.read_csv(plot_dir / "projection_cells.csv")
        assert len(plotted) == 32
        assert plotted.groupby("sample").size().to_dict() == {k: 8 for k in sources}
        assert report["umap"] is None
        assert sum(report["pca_explained_variance_ratio"]) <= 1.0000001

        class FakeUMAP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_transform(self, values):
                return np.asarray(values[:, :2], dtype=np.float32)

        with patch.dict(sys.modules, {"umap": types.SimpleNamespace(UMAP=FakeUMAP)}):
            with_umap = plot_main([
                "--input-dir", str(root), "--per-sample", "8",
                "--output-dir", str(root / "plots_with_umap"),
            ])
        second = root / "plots_with_umap"
        for name in (
            "pooled_umap_by_slice.png", "umap_by_slice_facets.png",
            "pooled_umap_by_counts.png", "pooled_umap_by_detected_genes.png",
        ):
            assert (second / name).is_file(), name
        assert with_umap["umap"]["n_neighbors"] == 30


def main():
    test_cuda_device_normalization()
    test_extract_exact_embeddings_and_row_alignment()
    test_panel_permutation_and_validation()
    test_coordinates_and_output()
    test_balanced_projection_pipeline()
    print("SG-MAE V1.1 embedding tests: all passed")


if __name__ == "__main__":
    main()
